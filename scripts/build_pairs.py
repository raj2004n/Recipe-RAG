"""Build the candidate pool the chooser reads, and calibrate its repeat guard.

The pool is the build-time half of pair selection. For every recipe in the
registered corpus it stores the top-ranked neighbours in the whole-recipe
embedding space, so the chooser can be handed a primary's candidates without
anything being re-encoded. Ranks 10 to 30 are the band a partner is drawn from,
ranks 1 to 3 are the plausibility anchor the chooser sees for a primary whose
band holds nothing workable, and ranks 4 to 9 are kept only so the band's edge
can be re-derived later.

The repeat guard is a cosine cut against the primary's own vector, and a flagged
candidate is never offered to the chooser: it is left out of the band, and the
fallback anchor is the closest candidate that is not flagged. The rows stay in
the pool so the flagging can be audited after the fact.

Two runs. `calibrate` measures what a cut flags across the pool and prints its
reach, writing nothing. `build` writes the pool and its manifest at a cut passed
on the command line, so the guard's value is chosen with its consequences in
front of it rather than buried in the code.

The corpus is read through the snapshot's registered dataset key and is never
written to. It is the only input, and the pair set this pool feeds stays
independent of every other dataset.
"""

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.rag.core.datasets.resolver import LocalDatasetResolver
from src.rag.core.logging import configure_logging
from src.rag.core.snapshot import registry
from src.rag.core.snapshot.paths import SnapshotPaths
from src.rag.evaluation.harness import load_snapshot, resolve_snapshot_folder
from src.rag.retrieval.chunker import chunk_recipes
from src.rag.retrieval.dense.encoder import Encoder

logger = logging.getLogger(__name__)

# The band a partner is drawn from, the ranks used as the plausibility anchor,
# and how many neighbours the pool keeps per primary.
BAND_START = 10
BAND_END = 30
ANCHOR_END = 3
TOP_K = BAND_END
SPACE = "whole_recipe"

# The cuts the calibration prints a full reach for.
REPORT_THRESHOLDS = [0.80, 0.82, 0.85, 0.87, 0.90, 0.92]

# How many of the highest-scoring rows the calibration names, so a cut's first
# casualties can be read rather than only counted.
PREVIEW_ROWS = 12

OUTPUT_JSONL = Path("data/pairs/candidate_pool.jsonl")
OUTPUT_MANIFEST = Path("data/pairs/candidate_pool_manifest.json")
VECTOR_CACHE = Path("/tmp/pairs_vectors.npz")


def role_for(rank: int) -> str:
    """Which part of the pool a rank belongs to."""
    if rank <= ANCHOR_END:
        return "anchor"
    return "band" if rank >= BAND_START else "gap"


def resolve_corpus(snapshot):
    """The registered corpus behind the snapshot's dataset key, and its registry entry.

    The registry entry is read alongside the resolved path so the manifest can
    carry the size the corpus was registered at, which the resolver does not
    expose.
    """
    paths = SnapshotPaths.from_paths_config({})
    resolver = LocalDatasetResolver(
        registry_path=paths.registry_file,
        registry_get_dataset=registry.get_dataset,
    )
    return resolver.resolve(snapshot.dataset_key), registry.get_dataset(paths.registry_file, snapshot.dataset_key)


def load_space(args: argparse.Namespace) -> dict:
    """The corpus, the encoder's identity, and one whole-recipe vector per recipe.

    Encoding the corpus takes about a minute, so the vectors are cached and
    reused while the cache still belongs to the same corpus and the same encoder.
    """
    snapshot = load_snapshot(resolve_snapshot_folder(args.snapshot))
    resolved, entry = resolve_corpus(snapshot)
    encoder_cfg = snapshot.base_config["encoder"]
    recipes = json.load(open(resolved.path))
    fingerprints = (encoder_cfg["model_id"], resolved.checksum)

    ids = vecs = None
    if VECTOR_CACHE.exists():
        cache = np.load(VECTOR_CACHE, allow_pickle=False)
        if tuple(str(x) for x in cache["fingerprints"]) == fingerprints:
            logger.info("reusing vectors from %s", VECTOR_CACHE)
            ids, vecs = cache["ids"], cache["vecs"]
    if ids is None:
        encoder = Encoder(
            encoder_cfg["model_id"],
            encoder_cfg["batch_size"],
            encoder_cfg["normalize_embeddings"],
        )
        encoded = encoder.encode_chunks(chunk_recipes(SPACE, resolved.path))[0].encoded_chunks
        ids = np.array([c.recipe_id for c in encoded])
        vecs = np.array([c.vector for c in encoded], dtype=np.float32)
        np.savez(VECTOR_CACHE, ids=ids, vecs=vecs, fingerprints=np.array(fingerprints))
        logger.info("encoded %d recipes with %s", len(ids), encoder_cfg["model_id"])

    primaries = recipes
    if args.limit is not None:
        primaries = sorted(recipes, key=lambda r: r["recipe_id"])[:args.limit]
    return {
        "recipes": recipes,
        "primaries": primaries,
        "ids": ids,
        "vecs": vecs,
        "encoder": encoder_cfg,
        "dataset_key": snapshot.dataset_key,
        "dataset_path": str(resolved.path),
        "dataset_checksum": resolved.checksum,
        "dataset_size_bytes": entry.get("size_bytes"),
        "dataset_registered_at": entry.get("created_at"),
    }


def neighbours(ids: np.ndarray, vecs: np.ndarray, primary: int, k: int) -> list[tuple[int, float]]:
    """The k nearest recipes to a primary, by cosine, ties broken on the lower id."""
    sims = vecs @ vecs[ids == primary][0]
    order = np.lexsort((ids, -sims))
    found: list[tuple[int, float]] = []
    for i in order:
        rid = int(ids[i])
        if rid == primary:
            continue
        found.append((rid, float(sims[i])))
        if len(found) == k:
            break
    return found


def pool_rows(primaries: list[dict], ids: np.ndarray, vecs: np.ndarray) -> list[dict]:
    """Every candidate the pool would hold, flattened."""
    rows: list[dict] = []
    for recipe in primaries:
        primary = recipe["recipe_id"]
        for rank, (rid, cos) in enumerate(neighbours(ids, vecs, primary, TOP_K), start=1):
            rows.append({
                "primary": primary,
                "recipe_id": rid,
                "rank": rank,
                "role": role_for(rank),
                "cosine": cos,
            })
    return rows


def threshold_impact(rows: list[dict], threshold: float) -> dict:
    """What one cut flags, and what that takes away from both uses of a candidate."""
    flagged = [r for r in rows if r["cosine"] >= threshold]

    band_totals: Counter = Counter()
    band_flagged: Counter = Counter()
    anchors: dict[int, list[bool]] = {}
    for row in rows:
        hit = row["cosine"] >= threshold
        if row["role"] == "band":
            band_totals[row["primary"]] += 1
            band_flagged[row["primary"]] += int(hit)
        elif row["role"] == "anchor":
            anchors.setdefault(row["primary"], []).append(not hit)

    return {
        "threshold": threshold,
        "flagged": len(flagged),
        "flagged_in_band": sum(1 for r in flagged if r["role"] == "band"),
        "flagged_anchors": sum(1 for r in flagged if r["role"] == "anchor"),
        "primaries_losing_a_band_candidate": sum(1 for p, n in band_flagged.items() if n),
        "primaries_with_no_usable_band": sum(1 for p, n in band_flagged.items() if n == band_totals[p]),
        "primaries_with_no_usable_anchor": sum(1 for flags in anchors.values() if not any(flags)),
    }


def print_calibration(rows: list[dict], names: dict, space: dict) -> None:
    """Print where the pool's cosines sit and what each cut would take away."""
    print(f"\ncorpus {len(space['recipes'])} recipes, {len(space['primaries'])} primaries | "
          f"encoder {space['encoder']['model_id']} | {space['dataset_checksum']}")
    print(f"pool rows: {len(rows)}")

    print("\ncosine inside the pool, by role")
    spans = {"anchor": f"rank 1 to {ANCHOR_END}", "gap": f"rank {ANCHOR_END + 1} to {BAND_START - 1}",
             "band": f"rank {BAND_START} to {BAND_END}"}
    for role in ("anchor", "gap", "band"):
        cos = np.array([r["cosine"] for r in rows if r["role"] == role])
        if not cos.size:
            print(f"  {role:<6} ({spans[role]}): none")
            continue
        print(f"  {role:<6} ({spans[role]}): {cos.min():.3f} to {cos.max():.3f}, median {np.median(cos):.3f}")

    print("\nhighest-scoring rows in the pool, the ones a cut flags first")
    for row in sorted(rows, key=lambda r: -r["cosine"])[:PREVIEW_ROWS]:
        print(f"  {row['cosine']:.3f}  {row['role']:<6} rank {row['rank']:>2}  primary {row['primary']:>5} "
              f"{names[row['primary']][:28]:<28} <-> {row['recipe_id']:>5} {names[row['recipe_id']][:32]}")

    print("\nwhat each cut flags, and what it takes away")
    print(f"  {'cut':>6}  {'flagged':>8}  {'in band':>8}  {'anchors':>8}  {'primaries losing':>17}  "
          f"{'no usable band':>15}  {'no usable anchor':>17}")
    for threshold in REPORT_THRESHOLDS:
        impact = threshold_impact(rows, threshold)
        print(f"  {threshold:>6.3f}  {impact['flagged']:>8}  {impact['flagged_in_band']:>8}  "
              f"{impact['flagged_anchors']:>8}  {impact['primaries_losing_a_band_candidate']:>17}  "
              f"{impact['primaries_with_no_usable_band']:>15}  "
              f"{impact['primaries_with_no_usable_anchor']:>17}")
    print()


def run_calibrate(args: argparse.Namespace) -> int:
    space = load_space(args)
    rows = pool_rows(space["primaries"], space["ids"], space["vecs"])
    names = {r["recipe_id"]: r["recipe_name"] for r in space["recipes"]}
    print_calibration(rows, names, space)
    return 0


def run_build(args: argparse.Namespace) -> int:
    space = load_space(args)
    rows = pool_rows(space["primaries"], space["ids"], space["vecs"])
    names = {r["recipe_id"]: r["recipe_name"] for r in space["recipes"]}
    threshold = args.threshold

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    candidates = flagged = flagged_in_band = primaries_with_flagged_band = 0
    primaries_without_anchor = 0
    with open(OUTPUT_JSONL, "w") as handle:
        for recipe in space["primaries"]:
            primary = recipe["recipe_id"]
            ranked = []
            band_flagged = False
            fallback = None
            for rank, (rid, cos) in enumerate(neighbours(space["ids"], space["vecs"], primary, TOP_K), start=1):
                role = role_for(rank)
                repeat = cos >= threshold
                flagged += int(repeat)
                if repeat and role == "band":
                    flagged_in_band += 1
                    band_flagged = True
                if role == "anchor" and not repeat and fallback is None:
                    fallback = rid
                candidates += 1
                ranked.append({
                    "rank": rank,
                    "recipe_id": rid,
                    "name": names[rid],
                    "cosine": round(cos, 6),
                    "role": role,
                    "repeat": repeat,
                })
            primaries_with_flagged_band += int(band_flagged)
            primaries_without_anchor += int(fallback is None)
            handle.write(json.dumps({
                "primary_id": primary,
                "primary_name": recipe["recipe_name"],
                "space": SPACE,
                "encoder": space["encoder"]["model_id"],
                "dataset_key": space["dataset_key"],
                "band": [BAND_START, BAND_END],
                "anchor": [1, ANCHOR_END],
                "fallback_anchor": fallback,
                "candidates": ranked,
            }) + "\n")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "space": SPACE,
        "encoder": space["encoder"],
        "dataset_key": space["dataset_key"],
        "dataset_path": space["dataset_path"],
        "dataset_checksum": space["dataset_checksum"],
        "dataset_size_bytes": space["dataset_size_bytes"],
        "dataset_registered_at": space["dataset_registered_at"],
        "corpus_recipes": len(space["recipes"]),
        "primaries": len(space["primaries"]),
        "limit": args.limit,
        "band": [BAND_START, BAND_END],
        "anchor": [1, ANCHOR_END],
        "top_k": TOP_K,
        "output": str(OUTPUT_JSONL),
        "repeat_guard": {
            "threshold": threshold,
            "rule": "a candidate is flagged when its cosine to the primary is at or above the threshold",
            "applied": (
                "a flagged candidate is not offered to the chooser: it is left out of the band, and the "
                "fallback anchor is the closest candidate in ranks 1 to 3 that is not flagged. The row stays "
                "in the pool so the flagging can be audited after the fact."
            ),
            "reach": threshold_impact(rows, threshold),
            "tradeoff": [threshold_impact(rows, t) for t in REPORT_THRESHOLDS],
        },
        "counts": {
            "candidates": candidates,
            "flagged_candidates": flagged,
            "flagged_in_band": flagged_in_band,
            "primaries_with_flagged_band_candidate": primaries_with_flagged_band,
            "primaries_with_no_usable_anchor": primaries_without_anchor,
        },
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("wrote %d primaries and %d candidates to %s", len(space["primaries"]), candidates, OUTPUT_JSONL)
    logger.info("%d candidates flagged as repeats, %d of them inside a band, %d primaries without an un-flagged anchor",
                flagged, flagged_in_band, primaries_without_anchor)
    return 0


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("calibrate", "build"), default="calibrate",
                        help="Calibrate the repeat guard, or write the pool at a given cut.")
    parser.add_argument("--snapshot", default="latest",
                        help="Snapshot target ('latest' or 'current') or folder name.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="The repeat guard's cosine cut. Required by --run build.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of primaries (for smoke runs).")
    args = parser.parse_args()
    if args.run == "build" and args.threshold is None:
        parser.error("--run build needs --threshold")
    raise SystemExit(run_calibrate(args) if args.run == "calibrate" else run_build(args))


if __name__ == "__main__":
    main()
