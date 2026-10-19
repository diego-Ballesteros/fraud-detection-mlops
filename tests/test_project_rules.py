"""Tests that enforce CLAUDE.md rules on the source tree instead of relying on memory.

Each test first runs its pattern against synthetic strings and a synthetic file: if the
pattern cannot catch a known violation, a clean scan of the repository proves nothing.
"""

import json
import re
from pathlib import Path

from fraud_detection import settings

# tests/ is deliberately excluded: literal seeds are legitimate there.
SCANNED_DIRS = [settings.ROOT_DIR / name for name in ("src", "notebooks", "scripts", "app")]
SCANNED_SUFFIXES = (".py", ".ipynb")
CONFIG_FILE = settings.ROOT_DIR / "src" / "fraud_detection" / "config.py"

HARDCODED_SEED = re.compile(
    r"""
    \b(?:random_state|seed)\s*=\s*[-+]?\d           # keyword or assignment: seed=42
    | ["'](?:random_state|seed)["']\s*:\s*[-+]?\d   # dictionary key: {"random_state": 42}
    | \bdefault_rng\(\s*[-+]?\d                     # numpy generator: default_rng(42)
    """,
    re.VERBOSE,
)
SYS_PATH = re.compile(
    r"""
    \bsys\.path\b                                   # sys.path.append(...), .insert(...)
    | \bfrom\s+sys\s+import\b[^#\n]*\bpath\b        # from sys import path
    """,
    re.VERBOSE,
)
NOTEBOOKS_DIR = settings.ROOT_DIR / "notebooks"
READS_PRODUCTION = re.compile(
    r"""
    \bread_production\b                             # loader.read_production()
    | \bPRODUCTION_PATH\b                           # pd.read_parquet(loader.PRODUCTION_PATH)
    | \bproduction\.parquet\b                       # the file, named directly
    """,
    re.VERBOSE,
)


def _files_under(roots: list[Path]) -> list[Path]:
    return sorted(
        path for root in roots for path in root.rglob("*") if path.suffix in SCANNED_SUFFIXES
    )


def _violations(pattern: re.Pattern[str], files: list[Path]) -> list[str]:
    return [
        f"{path}:{number}: {line.strip()}"
        for path in files
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line)
    ]


def _scanned_files() -> list[Path]:
    missing = [root for root in SCANNED_DIRS if not root.is_dir()]
    assert not missing, f"Scanned directories missing, the scan would pass vacuously: {missing}"
    files = _files_under(SCANNED_DIRS)
    assert CONFIG_FILE in files, "The scan does not reach the package source"
    return files


def test_no_hardcoded_seed(tmp_path: Path) -> None:
    violations = [
        "LogisticRegression(random_state=42)",
        "train_test_split(x, y, random_state = 0)",
        "TPESampler(seed=7)",
        'params = {"random_state": 42}',
        "params = {'seed': 3}",
        "rng = np.random.default_rng(42)",
    ]
    sanctioned = [
        "fit(random_state=settings.RANDOM_STATE)",
        "TPESampler(seed=settings.RANDOM_STATE)",
        'params = {"random_state": settings.RANDOM_STATE}',
        "rng = np.random.default_rng(settings.RANDOM_STATE)",
        "if random_state == 42:",
        "n_seeds=3",
    ]
    # Positive controls: every known violation form must be caught.
    assert [text for text in violations if not HARDCODED_SEED.search(text)] == []
    # Negative controls: the sanctioned forms and look-alikes must not be flagged.
    assert [text for text in sanctioned if HARDCODED_SEED.search(text)] == []
    # The scan itself must flag every form in real files, not only in strings.
    (tmp_path / "bad.py").write_text("\n".join(violations), encoding="utf-8")
    assert len(_violations(HARDCODED_SEED, _files_under([tmp_path]))) == len(violations)

    offenders = _violations(HARDCODED_SEED, _scanned_files())
    assert not offenders, f"Hardcoded seed; take it from settings.RANDOM_STATE: {offenders}"


def test_no_sys_path_manipulation(tmp_path: Path) -> None:
    violations = [
        "sys.path.append('..')",
        "sys.path.insert(0, str(root))",
        "from sys import path",
        "from sys import argv, path",
    ]
    sanctioned = ["os.path.join(a, b)", "from os import path", "from sys import argv  # path"]
    # Positive and negative controls, as in the seed test.
    assert [text for text in violations if not SYS_PATH.search(text)] == []
    assert [text for text in sanctioned if SYS_PATH.search(text)] == []
    # Notebooks are scanned as JSON text, one source line per JSON line as nbformat writes.
    notebook = json.dumps({"cells": [{"source": [f"{text}\n" for text in violations]}]}, indent=1)
    (tmp_path / "bad.ipynb").write_text(notebook, encoding="utf-8")
    assert len(_violations(SYS_PATH, _files_under([tmp_path]))) == len(violations)

    offenders = _violations(SYS_PATH, _scanned_files())
    assert not offenders, f"sys.path manipulation; import the installed package: {offenders}"


def test_no_notebook_reads_production(tmp_path: Path) -> None:
    violations = [
        "frame = loader.read_production()",
        "pd.read_parquet(loader.PRODUCTION_PATH)",
        'pd.read_parquet("data/processed/production.parquet")',
    ]
    sanctioned = [
        "frame = loader.read_reference()",
        "settings.PRODUCTION_FIRST_STEP",
        "production_rows = 0",
    ]
    # Positive and negative controls, as in the other two tests.
    assert [text for text in violations if not READS_PRODUCTION.search(text)] == []
    assert [text for text in sanctioned if READS_PRODUCTION.search(text)] == []
    notebook = json.dumps({"cells": [{"source": [f"{text}\n" for text in violations]}]}, indent=1)
    (tmp_path / "bad.ipynb").write_text(notebook, encoding="utf-8")
    assert len(_violations(READS_PRODUCTION, _files_under([tmp_path]))) == len(violations)

    notebooks = _files_under([NOTEBOOKS_DIR])
    assert notebooks, f"No notebook under {NOTEBOOKS_DIR}: the scan would pass vacuously"
    offenders = _violations(READS_PRODUCTION, notebooks)
    assert not offenders, f"A notebook reads the production period: {offenders}"
