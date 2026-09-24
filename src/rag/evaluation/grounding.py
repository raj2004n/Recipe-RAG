"""Flag prompts that ask for a way of cooking the pair gives the teacher no basis for.

A teacher reads a pair and a prompt, and invents one recipe from them. It can only build on what
the pair shows it, so a request that names a technique neither recipe uses is asking for something
the pair cannot teach: the answer comes from the model's own prior rather than from the corpus, and
nothing in the answer can be checked against the pair afterwards. The prompts that do this are the
rows a fine-tuning set cannot use, which is why they are worth finding before the teacher runs.

Each recipe's instructions carry the techniques it demonstrates. This reads both sides of a pair,
collects the techniques the pair shows, and compares them with the techniques the prompt names.
A technique a parent demonstrates is grounded. One that belongs to the same family as something a
parent demonstrates is recorded but not flagged, because a simmer against a boil is a step the pair
already shows the way to. One in no family the pair touches is the flag.

The comparison is a reading of words, so it is exact about vocabulary and blind to paraphrase: a
prompt can name a technique in words neither parent happens to use for the same work, and a parent
can demonstrate a technique without naming it. It is a filter for a review list, not a verdict, so
a pass reports the rate at which it fires and leaves the row in the file.

A pass writes an annotated copy of the file beside the original and never touches the file it read,
so a set that has already been designed can be measured without being designed again. The summary
is what says whether the flag is worth acting on: a flag that fires on most of the set is telling
the reader about the corpus rather than about the prompts.
"""

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

from ..data.pairs.rendering import instructions_text
from ..data.pairs.selection import (
    dataset_key_of,
    load_pairs,
    load_recipes,
    resolve_corpus,
)
from ..data.pairs.types import PAIRS_JSONL

logger = logging.getLogger(__name__)

# The file a pass measures when it is not pointed at one: the designed prompt set in use, whose
# annotated copy lands beside it.
DEFAULT_PROMPTS = Path("data/pairs/recipe_prompts_20260918.jsonl")

# The ways of cooking a prompt can ask for, as the words a recipe uses for them. Each carries the
# family it belongs to, which is what decides whether a technique a parent does not name is still
# within reach of what the pair shows. The patterns carry the inflections a recipe page uses, and
# the ones that mean something else in a kitchen are narrowed: "reduce" is read as reducing a
# liquid and not as turning down a burner, and "air fry" is an appliance rather than a fat.
#
# The families come in two tiers. A structural family is a way of applying heat or setting a dish,
# and a prompt that asks for one the pair never uses is asking for the pair's craft to be replaced.
# A detail family is how a component is handled on the way, so a prompt that asks for a pickled
# garnish or a toasted nut from a pair that does neither is asking for a garnish, not a dish. Both
# are reported, and the structural ones are what a review list leads with.
#
# The families are drawn where a cook would draw them rather than where the words sit. Boiling,
# simmering and braising are one family, because a page that simmers shows the way to a braise.
# Steaming is its own, because nothing about a bake or a simmer shows the way to it.
TECHNIQUES: dict[str, tuple[str, str]] = {
    "boil": ("wet heat", r"\bboil(?:s|ed|ing)?\b"),
    "simmer": ("wet heat", r"\bsimmer(?:s|ed|ing)?\b"),
    "poach": ("wet heat", r"\bpoach(?:es|ed|ing)?\b"),
    "blanch": ("wet heat", r"\bblanch(?:es|ed|ing)?\b"),
    "steam": ("steam", r"\bsteam(?:s|ed|ing|er|ers)?\b"),
    "braise": ("wet heat", r"\bbrais(?:e|es|ed|ing)\b"),
    "stew": ("wet heat", r"\bstew(?:s|ed|ing)?\b"),
    "slow cook": ("wet heat", r"\bslow[\s-]?cook(?:s|ed|er|ing)?\b"),
    "pressure cook": ("wet heat", r"\bpressure[\s-]?cook(?:s|ed|er|ing)?\b"),
    "bake": ("oven", r"\bbak(?:e|es|ed|ing)\b"),
    "roast": ("oven", r"\broast(?:s|ed|ing|er)?\b"),
    "broil": ("oven", r"\bbroil(?:s|ed|ing|er)?\b"),
    "toast": ("oven", r"\btoast(?:s|ed|ing)?\b"),
    "fry": ("fat", r"\bfry(?:ing)?\b|\bfri(?:ed|es|er|ers)\b|\bfritter(?:s)?\b"),
    "saute": ("fat", r"\bsaut(?:e|é|es|ed|ing)\b"),
    "sear": ("fat", r"\bsear(?:s|ed|ing)?\b"),
    "stir fry": ("fat", r"\bstir[\s-]?fr(?:y|ies|ied|ying)\b"),
    "deep fry": ("fat", r"\bdeep[\s-]?fr(?:y|ies|ied|ying)\b"),
    "grill": ("direct fire", r"\bgrill(?:s|ed|ing)?\b"),
    "barbecue": ("direct fire", r"\bbarbecu(?:e|es|ed|ing)\b|\bbbq\b"),
    "char": ("direct fire", r"\bcharred\b|\bcharring\b|\bchar[\s-]?grill\b"),
    "smoke": ("direct fire", r"\bsmok(?:e|es|ed|ing)\b"),
    "chill": ("cold set", r"\bchill(?:s|ed|ing)?\b|\brefrigerat(?:e|es|ed|ing|or)\b"),
    "freeze": ("cold set", r"\bfreez(?:e|es|ing)\b|\bfrozen\b"),
    "no bake": ("cold set", r"\bno[\s-]?bak(?:e|ing)\b"),
    "microwave": ("appliance", r"\bmicrowav(?:e|es|ed|ing)\b"),
    "air fry": ("appliance", r"\bair[\s-]?fr(?:y|ies|ied|ying|er)\b"),
    "knead": ("hand work", r"\bknead(?:s|ed|ing)?\b"),
    "whisk": ("hand work", r"\bwhisk(?:s|ed|ing)?\b"),
    "whip": ("hand work", r"\bwhip(?:s|ped|ping)?\b"),
    "fold": ("hand work", r"\bfold(?:s|ed|ing)?\b"),
    "blend": ("hand work", r"\bblend(?:s|ed|ing|er)?\b"),
    "puree": ("hand work", r"\bpur(?:e|é)e(?:s|d)?\b"),
    "mash": ("hand work", r"\bmash(?:es|ed|ing)?\b"),
    "layer": ("assembly", r"\blay(?:er|ers|ered|ering)\b"),
    "stuff": ("assembly", r"\bstuff(?:s|ed|ing)?\b"),
    "wrap": ("assembly", r"\bwrap(?:s|ped|ping)?\b"),
    "fill": ("assembly", r"\bfill(?:s|ed|ing)?\b"),
    "assemble": ("assembly", r"\bassembl(?:e|es|ed|ing|y)\b"),
    "marinate": ("flavour work", r"\bmarinat(?:e|es|ed|ing)\b"),
    "brine": ("flavour work", r"\bbrin(?:e|es|ed|ing)\b"),
    "pickle": ("flavour work", r"\bpickl(?:e|es|ed|ing)\b"),
    "reduce": ("flavour work", r"\breduc(?:e|es|ed|ing)\b(?!\s+the\s+heat)"),
    "caramelise": ("flavour work", r"\bcaramelis(?:e|es|ed|ing)\b|\bcarameliz(?:e|es|ed|ing)\b"),
    "deglaze": ("flavour work", r"\bdeglaz(?:e|es|ed|ing)\b"),
    "proof": ("flavour work", r"\bproof(?:s|ed|ing)?\b"),
    "temper": ("flavour work", r"\btemper(?:s|ed|ing)?\b"),
}

STRUCTURAL_FAMILIES = {
    "wet heat", "steam", "oven", "fat", "direct fire", "cold set", "appliance",
}

# A technique named inside a refusal is not a technique being asked for. "skips the deep-fryer",
# "rather than baking", "no steaming" all name a method in order to rule it out, and reading them
# as requests would flag the prompts that have learned to say what they do not want.
NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|skip(?:s|ped|ping)?|avoid(?:s|ed|ing)?|instead\s+of|rather\s+than"
    r"|other\s+than|free|don'?t|doesn'?t|isn'?t|aren'?t|won'?t|can'?t|un)\b",
    re.IGNORECASE,
)
NEGATION_WINDOW = 60

TECHNIQUE_PATTERNS = {
    name: (family, re.compile(pattern, re.IGNORECASE))
    for name, (family, pattern) in TECHNIQUES.items()
}
FAMILY_OF = {name: family for name, (family, _) in TECHNIQUE_PATTERNS.items()}


def techniques_in(text: str, honour_negation: bool = False) -> set[str]:
    """The techniques a piece of kitchen text names.

    A refusal is read as a refusal when `honour_negation` is set, which is what a prompt is read
    with. A recipe's instructions are read without it, because a step that says "do not boil" is
    still a page that has thought about boiling.
    """
    found = set()
    for name, (_, pattern) in TECHNIQUE_PATTERNS.items():
        for match in pattern.finditer(text):
            if honour_negation and NEGATION_RE.search(
                text[max(0, match.start() - NEGATION_WINDOW):match.start()]
            ):
                continue
            found.add(name)
            break
    return found


def ground(prompt: str, parents: list[tuple[str, str]]) -> dict:
    """How the techniques a prompt names sit against the techniques the pair shows.

    The pair is read as each recipe's name and its instructions, because a dish called a stir fry
    teaches stir-frying whether or not the steps spell the word out. A technique a parent shows is
    grounded. One whose family a parent touches is recorded under `nearby`, since the pair shows the
    way to that kind of heat even where it does not use the word. The rest are the gaps, split into
    the ones that take the pair's craft somewhere it never goes and the ones that touch a component.
    """
    requested = techniques_in(prompt, honour_negation=True)
    shown = set()
    for name, instructions in parents:
        shown |= techniques_in(name) | techniques_in(instructions)
    families = {FAMILY_OF[name] for name in shown}
    absent = sorted(requested - shown)
    structural = [name for name in absent if FAMILY_OF[name] in STRUCTURAL_FAMILIES]
    return {
        "requested": sorted(requested),
        "shown": sorted(shown),
        "families_shown": sorted(families),
        "grounded": sorted(requested & shown),
        "nearby": sorted(name for name in absent if FAMILY_OF[name] in families
                         and FAMILY_OF[name] not in STRUCTURAL_FAMILIES),
        "structural_gap": sorted(
            name for name in structural if FAMILY_OF[name] not in families),
        "detail_gap": sorted(
            name for name in absent
            if FAMILY_OF[name] not in STRUCTURAL_FAMILIES and FAMILY_OF[name] not in families),
        "unsupported": sorted(
            set(structural) | {name for name in absent if FAMILY_OF[name] not in families}),
    }


def unsupported_flag(result: dict) -> str:
    """The flag a review list carries for one row, or an empty string when there is nothing to say."""
    if not result["unsupported"]:
        return ""
    named = result["structural_gap"] + [
        f"{name} (a component)" for name in result["detail_gap"]
    ]
    return (
        "asks for what the pair never shows: "
        + ", ".join(named)
        + f"; it shows {', '.join(result['shown']) or 'nothing this check knows'}"
    )


def annotate(rows: list[dict], recipes: dict) -> tuple[list[dict], list[dict]]:
    """Every row with its grounding attached, and the rows whose pair was missing a recipe."""
    annotated, missing = [], []
    for row in rows:
        primary = recipes.get(row["primary_id"])
        partner = recipes.get(row["partner_id"])
        if primary is None or partner is None:
            missing.append(row)
            continue
        result = ground(
            row["prompt"],
            [
                (primary.recipe_name, instructions_text(primary)),
                (partner.recipe_name, instructions_text(partner)),
            ],
        )
        flag = unsupported_flag(result)
        annotated.append({**row, "grounding": result, "grounding_flags": [flag] if flag else []})
    return annotated, missing


def summarise(rows: list[dict]) -> dict:
    """What a reader needs to judge the flag: how often it fires and on what."""
    fired = [row for row in rows if row["grounding"]["unsupported"]]
    structural = [row for row in rows if row["grounding"]["structural_gap"]]
    unsupported = Counter(name for row in rows for name in row["grounding"]["unsupported"])
    structural_counts = Counter(name for row in rows for name in row["grounding"]["structural_gap"])
    requested = Counter(name for row in rows for name in row["grounding"]["requested"])
    shown = Counter(name for row in rows for name in row["grounding"]["shown"])
    demonstrated = Counter(name for row in rows for name in row["grounding"]["grounded"])
    return {
        "rows": len(rows),
        "rows_flagged": len(fired),
        "flagged_share": round(len(fired) / len(rows), 3) if rows else 0.0,
        "rows_with_a_gap_in_the_craft": len(structural),
        "craft_gap_share": round(len(structural) / len(rows), 3) if rows else 0.0,
        "per_row": dict(sorted(Counter(
            len(row["grounding"]["unsupported"]) for row in rows
        ).items())),
        "unsupported": dict(unsupported.most_common()),
        "structural_gaps": dict(structural_counts.most_common()),
        "requested": dict(requested.most_common()),
        "demonstrated_by_pair": dict(demonstrated.most_common()),
        "shown_by_pair": dict(shown.most_common()),
        "rows_with_no_technique_named": sum(1 for row in rows if not row["grounding"]["requested"]),
    }


def print_report(summary: dict, show: int) -> None:
    """The counts, then the techniques the flag is riding on."""
    print(f"rows {summary['rows']}  flagged {summary['rows_flagged']} "
          f"({summary['flagged_share']:.0%})  of which a gap in the craft "
          f"{summary['rows_with_a_gap_in_the_craft']} ({summary['craft_gap_share']:.0%})  "
          f"no technique named at all {summary['rows_with_no_technique_named']}")
    print("unsupported per row: " + "  ".join(
        f"{count}={rows}" for count, rows in summary["per_row"].items()))
    print("\nthe craft a pair does not show, by rows flagged for it:")
    for name, count in list(summary["structural_gaps"].items())[:show or 20]:
        print(f"  {count:>5}  {name} ({FAMILY_OF[name]})")
    print("\ntechniques the prompts ask for, and how often the pair showed them:")
    for name, count in list(summary["requested"].items())[:show or 20]:
        shown = summary["demonstrated_by_pair"].get(name, 0)
        print(f"  asked {count:>5}  pair shows it {shown:>5}  {name}")


def print_rows(rows: list[dict], show: int) -> None:
    """The flagged rows themselves, with what the pair behind each one shows."""
    flagged = [row for row in rows if row["grounding"]["structural_gap"]]
    for row in flagged[:show]:
        ground_result = row["grounding"]
        print(f"\n{row['primary_name']} / {row['partner_name']}"
              f"  ({row['primary_id']} + {row['partner_id']})")
        print(f"  the pair does not show: {', '.join(ground_result['structural_gap'])}"
              + (f"  |  nor {', '.join(ground_result['detail_gap'])} as a component"
                 if ground_result["detail_gap"] else ""))
        print(f"  the pair shows: {', '.join(ground_result['shown']) or 'nothing this check knows'}")
        print(f"  prompt: {row['prompt']}")
    if len(flagged) > show:
        print(f"\n... {len(flagged) - show} more rows with a gap in the craft in the written file")


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write the annotated file whole, by rename, so a reader never sees it half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(temp_path, path)


def check_grounding(
    prompts_path: Path = DEFAULT_PROMPTS,
    pairs_path: Path = PAIRS_JSONL,
    recipes_override: str | None = None,
    out_path: Path | None = None,
    no_write: bool = False,
    show: int = 15,
    limit: int | None = None,
) -> int:
    """Measure the designed prompts and write the annotated rows beside them.

    The prompts default to the sample whose rows have already been read by hand, and the corpus is
    the one the pair set was drawn from unless `recipes_override` names a file. `show` sets how many
    techniques and rows the printed report carries, and `limit` measures a prefix of the file for a
    smoke run. The annotated copy lands beside the prompts unless `out_path` names somewhere else,
    and `no_write` reports without writing it at all.
    """
    prompts_path = Path(prompts_path)
    rows = [json.loads(line) for line in prompts_path.open() if line.strip()]
    pairs = load_pairs(Path(pairs_path))
    dataset_key = dataset_key_of(pairs)
    corpus = resolve_corpus(dataset_key, recipes_override)
    recipes = load_recipes(Path(corpus["path"]))
    logger.info("read %d prompts, %d pairs and %d recipes", len(rows), len(pairs), len(recipes))

    if limit is not None:
        rows = rows[:limit]

    annotated, missing = annotate(rows, recipes)
    if missing:
        logger.warning("%d prompt rows name a pair whose recipes are not in the corpus", len(missing))

    summary = summarise(annotated)
    print_report(summary, show)
    print_rows(annotated, show)

    if no_write:
        print(f"\nnothing written ({len(annotated)} rows measured)")
        return 0
    out_path = Path(out_path) if out_path else prompts_path.with_name(
        f"{prompts_path.stem}_grounding.jsonl")
    write_rows(out_path, annotated)
    print(f"\nwrote {len(annotated)} annotated rows to {out_path}")
    return 0
