import json
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DVC_FILE = REPO_ROOT / "dvc.yaml"

# The stages the graph is allowed to hold, in the order they run: the build
# pipeline from the crawl to the seal, and nothing else.
_BUILD_PIPELINE = (
    "collect",
    "preprocess",
    "set_strategy",
    "chunk",
    "transform",
    "index",
    "finalise_build",
)

# Each build stage, the output it declares, and the output of the stage before
# it that it must take as an input.
_BUILD_CHAIN = (
    ("set_strategy", "build/staging/metadata.json", None),
    ("chunk", "build/staging/chunks/", "build/staging/metadata.json"),
    ("transform", "build/staging/transformed_chunks/", "build/staging/chunks/"),
    ("index", "build/staging/index/", "build/staging/transformed_chunks/"),
)

_STAGING_DIRS = (
    "build/staging/chunks/",
    "build/staging/transformed_chunks/",
    "build/staging/index/",
)

_CONFIG_FILES = (
    "configs/base_config.yaml",
    "configs/paths_config.yaml",
    "configs/prompts_config.yaml",
)

# Each build stage's config keys live twice: in the `params:` block it declares
# here, and in its row of `STAGE_CONFIG_KEYS`, the table its code reads config
# through. A row spells a file by its own name and dvc.yaml spells it by its path,
# which is what this joins.
_CONFIG_DIR = "configs"

# The one build stage that cannot name keys instead of files: it copies the config
# bytes into the sealed folder, so every byte of them is part of what it produces.
_WHOLE_CONFIG_STAGE = "finalise_build"


def _stages() -> dict:
    return yaml.safe_load(DVC_FILE.read_text())["stages"]


def test_the_graph_is_exactly_the_build_pipeline():
    """The stage set is the seven stages of the build, and nothing else.

    A stage for a pair, distillation, training or eval pass would put off-graph
    work inside the graph DVC skips and reproduces, which is the boundary this
    pins: adding one is a deliberate edit here rather than a silent addition to
    `dvc.yaml`.
    """
    assert set(_stages()) == set(_BUILD_PIPELINE)


def test_the_build_stages_declare_the_staging_chain():
    """Each stage owns one output under the staging root and reads the last one.

    The output is what DVC hashes to decide whether to skip the stage, and the
    input is what orders the four into a chain rather than four stages that
    could run in any order.
    """
    stages = _stages()
    for name, out, previous_out in _BUILD_CHAIN:
        stage = stages[name]
        assert stage["outs"] == [out], f"{name} must declare {out} as its output"
        if previous_out is not None:
            assert previous_out in stage["deps"], (
                f"{name} must depend on {previous_out} so DVC orders the chain"
            )


def test_finalise_build_takes_the_staging_dirs_and_declares_no_outputs():
    """The seal declares no output, because the folder it writes is named at seal time.

    The versioned name is claimed when the build seals, so it cannot appear in
    a static graph. The stage therefore carries no output at all and is re-run
    on every repro; what it reads is the three staging directories and the
    configs it copies into the sealed folder.
    """
    stage = _stages()["finalise_build"]
    assert not stage.get("outs")
    assert stage["always_changed"] is True
    for dep in (*_STAGING_DIRS, *_CONFIG_FILES):
        assert dep in stage["deps"]


def test_the_params_blocks_name_the_keys_each_stage_declares():
    """Each stage's `params:` entries are the row its code reads config through.

    A stage that started reading a key with no entry here would change its output
    while DVC still saw an unchanged dependency, and a skipped re-run is quieter
    than an over-eager one. `STAGE_CONFIG_KEYS` holds the rows and the reader
    refuses a key outside them; this holds the yaml to the same rows from the other
    side, so the two cannot drift apart.
    """
    from src.rag.core.config import STAGE_CONFIG_KEYS

    stages = _stages()
    for stage, row in STAGE_CONFIG_KEYS.items():
        if stage == _WHOLE_CONFIG_STAGE:
            assert "params" not in stages[stage], (
                f"{stage} copies the config bytes into the seal, so it keeps whole-file deps"
            )
            continue
        declared = {
            f"{_CONFIG_DIR}/{name}": sorted(keys) for name, keys in row.items()
        }
        assert _named_params(stages[stage]) == declared, (
            f"{stage}'s params: block and its STAGE_CONFIG_KEYS row name different keys"
        )


def test_a_directory_output_skips_when_unchanged_and_reruns_when_partial(tmp_path: Path):
    """The behaviour the graph relies on, pinned against the installed DVC.

    A stage whose directory output still matches the hash in `dvc.lock` is not
    re-run, which is what lets an unchanged build skip the four build stages. A
    directory output with a file missing no longer matches, so the stage re-runs
    rather than leaving the build to trust a tree that lost bytes. This runs in
    a throwaway repository: the project's own graph needs the corpus and the
    encoder models to do the work for real.
    """
    (tmp_path / "dvc.yaml").write_text(
        textwrap.dedent(f"""
            stages:
              build:
                cmd: {sys.executable} -c "import pathlib; d = pathlib.Path('out'); d.mkdir(exist_ok=True); (d / 'a.txt').write_text('a'); (d / 'b.txt').write_text('b')"
                outs:
                - out/
        """).lstrip()
    )
    _dvc(tmp_path, "init", "--no-scm")
    _dvc(tmp_path, "repro", "build")

    assert "didn't change, skipping" in _dvc(tmp_path, "repro", "build")

    (tmp_path / "out" / "b.txt").unlink()
    assert "Running stage" in _dvc(tmp_path, "repro", "build")


def test_the_declared_code_deps_cover_what_the_build_imports():
    """Every src.rag module the build entry point imports is covered by a declared dep.

    The per-stage dep lists are hand-maintained, so this is what keeps them
    from drifting: an import added to the build's path that no stage declares
    fails here, rather than a stage quietly reading code that no edit to it
    would re-run.
    """
    declared = {
        dep
        for stage in _stages().values()
        for dep in stage.get("deps", [])
        if dep.startswith("src/")
    }
    uncovered = sorted(
        module for module in _imported_src_rag_files() if not _covered(module, declared)
    )
    assert uncovered == [], (
        f"{uncovered} are imported by the build but declared by no stage's deps"
    )


def _dvc(cwd: Path, *args: str) -> str:
    """Run a dvc subcommand in `cwd` and return its combined output."""
    result = subprocess.run(
        [sys.executable, "-m", "dvc", *args], cwd=cwd, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def _named_params(stage: dict) -> dict[str, list[str]]:
    """A stage's `params:` entries as {file path: sorted keys}.

    Every entry in this graph is file-scoped, the form that names keys inside a
    file other than the default `params.yaml`.
    """
    return {
        path: sorted(keys)
        for entry in stage.get("params", [])
        for path, keys in entry.items()
    }


def _imported_src_rag_files() -> list[str]:
    """Import the build entry point and list the src/rag files it loaded.

    The import happens in its own process: the entry point pulls in the builder
    stack and the config singletons, and a test process should not have to hold
    either.
    """
    code = textwrap.dedent("""
        import json, sys
        import scripts.build_rag
        print(json.dumps(sorted({
            module.__file__
            for name, module in sys.modules.items()
            if name.startswith("src.rag") and getattr(module, "__file__", None)
        })))
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    return [str(Path(path).resolve().relative_to(REPO_ROOT)) for path in loaded]


def _covered(module_path: str, declared: set[str]) -> bool:
    """True when a declared dep names the module, or a directory holding it.

    A package's own `__init__.py` counts as covered when a module inside that
    package is declared, since importing a declared module imports its package
    root first and the root directory itself (`src/rag/`) is deliberately too
    coarse for a stage's deps to name.
    """
    if module_path.endswith("/__init__.py"):
        package_dir = module_path[: -len("__init__.py")]
        if any(dep.startswith(package_dir) for dep in declared):
            return True
    return any(
        module_path == dep or (dep.endswith("/") and module_path.startswith(dep))
        for dep in declared
    )
