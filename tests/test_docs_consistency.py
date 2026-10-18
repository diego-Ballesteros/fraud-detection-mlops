"""Tests that keep documents true where they state facts of the repository.

ROADMAP section 7 lists every module of src/fraud_detection/ and every script of scripts/,
each with its status in one vocabulary for both tables: "existe", or "fase N" when it is
planned. Both listings are checked against the files in both directions, so neither can
fall out of date without a test failing. Every check first runs on a synthetic listing
where the expected problems are known, so a clean result on the real one means something.
"""

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from fraud_detection import settings

ROADMAP = settings.DOCS_DIR / "ROADMAP.md"
PACKAGE_DIR = settings.ROOT_DIR / "src" / "fraud_detection"
SCRIPTS_DIR = settings.ROOT_DIR / "scripts"
ROOT_PACKAGE = "(raíz)"
EXISTS = "existe"
MODULE_HEADER = ("Paquete", "Módulo", "Estado")
SCRIPT_HEADER = ("Script", "Qué hace", "Estado")
PLANNED = re.compile(r"fase \d+")

Row = tuple[str, ...]


def _cells(line: str) -> Row | None:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    return tuple(cell.strip() for cell in stripped.strip("|").split("|"))


def _section_7(text: str) -> str:
    start = text.index("## 7. ")
    return text[start : text.index("\n## 8. ", start)]


def parse_table(section: str, header: Row) -> list[Row]:
    """Return the rows of the first table with the given header, in order."""
    lines = iter(section.splitlines())
    for line in lines:
        if _cells(line) == header:
            next(lines)  # the |---| separator
            rows = []
            for row_line in lines:
                cells = _cells(row_line)
                if cells is None:
                    break
                rows.append(cells)
            return rows
    return []


def module_name(row: Row) -> str:
    return f"{row[0]}/{row[1]}"


def script_name(row: Row) -> str:
    return row[0].strip("`")


def real_modules(package_dir: Path) -> set[str]:
    """Return package/module for every .py file under the package but __init__.py."""
    names = set()
    for path in package_dir.rglob("*.py"):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue
        parent = path.parent.relative_to(package_dir).as_posix()
        names.add(f"{ROOT_PACKAGE if parent == '.' else parent}/{path.stem}")
    return names


def real_scripts(scripts_dir: Path) -> set[str]:
    """Return the file name of every .py script in the scripts folder."""
    return {path.name for path in scripts_dir.glob("*.py")}


def compare(
    rows: list[Row],
    real: set[str],
    header: Row,
    name_of: Callable[[Row], str],
) -> list[str]:
    """Return one line per disagreement between a listing and the files."""
    problems = []
    listed = set()
    for row in rows:
        if len(row) != len(header):
            problems.append(f"malformed row: {row}")
            continue
        name, status = name_of(row), row[-1]
        if name in listed:
            problems.append(f"listed twice: {name}")
        listed.add(name)
        if status == EXISTS:
            if name not in real:
                problems.append(f"listed as existing, but there is no file: {name}")
        elif PLANNED.fullmatch(status):
            if name in real:
                problems.append(f"listed as planned ({status}), but the file exists: {name}")
        else:
            problems.append(f"unknown status {status!r}: {name}")
    for name in sorted(real - listed):
        problems.append(f"not listed: {name}")
    return problems


def compare_modules(rows: list[Row], real: set[str]) -> list[str]:
    return compare(rows, real, MODULE_HEADER, module_name)


def compare_scripts(rows: list[Row], real: set[str]) -> list[str]:
    return compare(rows, real, SCRIPT_HEADER, script_name)


def assert_no_problems(problems: list[str]) -> None:
    assert not problems, "ROADMAP section 7 and the repository disagree:\n" + "\n".join(problems)


SYNTHETIC_SECTION = """## 7. Estructura

```
├── src/fraud_detection/           ← tabla de abajo
├── scripts/                       ← tabla de abajo
```

### Módulos

| Paquete | Módulo | Estado |
|---|---|---|
| (raíz) | config | existe |
| data | ghost | existe |
| models | estimators | fase 2 |

### Scripts

| Script | Qué hace | Estado |
|---|---|---|
| `prepare_data.py` | Prepara | existe |
| `ghost_script.py` | — | existe |
| `run_training.py` | El AutoML | fase 2 |

Texto después de las tablas.
"""


def test_the_parser_reads_both_tables() -> None:
    assert parse_table(SYNTHETIC_SECTION, MODULE_HEADER) == [
        ("(raíz)", "config", "existe"),
        ("data", "ghost", "existe"),
        ("models", "estimators", "fase 2"),
    ]
    assert parse_table(SYNTHETIC_SECTION, SCRIPT_HEADER) == [
        ("`prepare_data.py`", "Prepara", "existe"),
        ("`ghost_script.py`", "—", "existe"),
        ("`run_training.py`", "El AutoML", "fase 2"),
    ]
    # A section with no such table yields no rows, never an exception swallowed.
    assert parse_table("## 7. Estructura\n\nNo table here.\n", MODULE_HEADER) == []


def test_one_module_too_many_and_one_too_few_fail_for_both_reasons() -> None:
    rows = parse_table(SYNTHETIC_SECTION, MODULE_HEADER)
    real = {"(raíz)/config", "data/loader"}
    # data/ghost is listed with no file; data/loader is a file with no row. The check is on
    # the list of problems, not on the assertion message, which pytest rewrites.
    assert compare_modules(rows, real) == [
        "listed as existing, but there is no file: data/ghost",
        "not listed: data/loader",
    ]
    with pytest.raises(AssertionError):
        assert_no_problems(compare_modules(rows, real))
    # Negative control: the same listing without both faults passes.
    fixed = [rows[0], ("data", "loader", "existe"), rows[2]]
    assert_no_problems(compare_modules(fixed, real))


def test_one_script_too_many_and_one_too_few_fail_for_both_reasons() -> None:
    rows = parse_table(SYNTHETIC_SECTION, SCRIPT_HEADER)
    real = {"prepare_data.py", "unlisted.py"}
    # ghost_script.py is listed as existing with no file; unlisted.py exists with no row.
    assert compare_scripts(rows, real) == [
        "listed as existing, but there is no file: ghost_script.py",
        "not listed: unlisted.py",
    ]
    with pytest.raises(AssertionError):
        assert_no_problems(compare_scripts(rows, real))
    # Negative control: the same listing without both faults passes.
    fixed = [rows[0], ("`unlisted.py`", "—", "existe"), rows[2]]
    assert_no_problems(compare_scripts(fixed, real))


def test_the_other_disagreements_are_named() -> None:
    real = {"models/baseline"}
    rows: list[Row] = [
        ("models", "baseline", "fase 2"),
        ("models", "estimators", "someday"),
        ("models", "estimators", "fase 2"),
        ("data", "loader"),
    ]
    assert compare_modules(rows, real) == [
        "listed as planned (fase 2), but the file exists: models/baseline",
        "unknown status 'someday': models/estimators",
        "listed twice: models/estimators",
        "malformed row: ('data', 'loader')",
    ]
    # One vocabulary for both tables: "planificado", the scripts' former status, now fails
    # in either table, and "fase 2" is valid in both.
    assert compare_scripts([("`run_training.py`", "—", "planificado")], set()) == [
        "unknown status 'planificado': run_training.py"
    ]
    assert compare_modules([("models", "estimators", "planificado")], set()) == [
        "unknown status 'planificado': models/estimators"
    ]
    assert compare_scripts([("`run_training.py`", "—", "fase 2")], set()) == []


def test_roadmap_module_table_matches_the_package() -> None:
    rows = parse_table(_section_7(ROADMAP.read_text(encoding="utf-8")), MODULE_HEADER)
    real = real_modules(PACKAGE_DIR)
    # Controls: both sides are populated, so the comparison cannot pass vacuously.
    assert (ROOT_PACKAGE, "config", EXISTS) in rows
    assert f"{ROOT_PACKAGE}/config" in real
    assert_no_problems(compare_modules(rows, real))


def test_roadmap_script_table_matches_the_scripts_folder() -> None:
    rows = parse_table(_section_7(ROADMAP.read_text(encoding="utf-8")), SCRIPT_HEADER)
    real = real_scripts(SCRIPTS_DIR)
    # Controls: both sides are populated, so the comparison cannot pass vacuously.
    assert ("`prepare_data.py`", "CSV → dos parquet, validación y mediciones", EXISTS) in rows
    assert "prepare_data.py" in real
    assert_no_problems(compare_scripts(rows, real))
