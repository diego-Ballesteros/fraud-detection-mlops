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
# The monitoring notebook is the only one that may read production, and only in a code cell
# that says why (ADR 0012). Any other mention, in any cell of any notebook, is a violation.
PRODUCTION_READER = "04_monitoring.ipynb"
PRODUCTION_READ_MARKER = "Lectura de producción autorizada"


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
    marker = f"# {PRODUCTION_READ_MARKER}: fase 4\n"
    read = "frame = loader.read_production()\n"
    bad = _notebook(tmp_path / "bad.ipynb", [_code(f"{text}\n") for text in violations])
    assert _production_reads(bad) == ([f"bad.ipynb, cell {i}" for i in range(3)], 0)
    # In the monitoring notebook a marked code cell is authorized; an unmarked code cell and a
    # markdown cell are not, and the marker authorizes nothing in any other notebook.
    reader = _notebook(
        tmp_path / PRODUCTION_READER,
        [_code(marker, read), _code(read), {"cell_type": "markdown", "source": [marker, read]}],
    )
    assert _production_reads(reader) == (
        [f"{PRODUCTION_READER}, cell 1", f"{PRODUCTION_READER}, cell 2"],
        1,
    )
    other = _notebook(tmp_path / "05_other.ipynb", [_code(marker, read)])
    assert _production_reads(other) == (["05_other.ipynb, cell 0"], 0)

    notebooks = _files_under([NOTEBOOKS_DIR])
    assert notebooks, f"No notebook under {NOTEBOOKS_DIR}: the scan would pass vacuously"
    found = {path.name: _production_reads(path) for path in notebooks}
    offenders = [cell for cells, _ in found.values() for cell in cells]
    assert not offenders, f"A notebook reads the production period: {offenders}"
    # Control: when the monitoring notebook exists, the scan sees its authorized read.
    if PRODUCTION_READER in found:
        assert found[PRODUCTION_READER][1] >= 1, "The scan does not see the authorized read"


def _code(*lines: str) -> dict[str, object]:
    return {"cell_type": "code", "source": list(lines), "outputs": []}


def _notebook(path: Path, cells: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"cells": cells}, indent=1), encoding="utf-8")
    return path


def _production_reads(path: Path) -> tuple[list[str], int]:
    """Cells that mention the production period without authorization, and authorized reads.

    A whole cell is scanned, source and outputs, so a path printed in an output counts too.
    """
    offenders, authorized = [], 0
    for index, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
        if not READS_PRODUCTION.search(json.dumps(cell, ensure_ascii=False)):
            continue
        source = "".join(cell.get("source", []))
        is_marked_code = cell.get("cell_type") == "code" and PRODUCTION_READ_MARKER in source
        if path.name == PRODUCTION_READER and is_marked_code:
            authorized += 1
        else:
            offenders.append(f"{path.name}, cell {index}")
    return offenders, authorized
