"""Tests that keep documents true where they state facts of the repository.

ROADMAP section 7 lists every module of src/fraud_detection/ with its status: "existe" or
"fase N". The listing is checked against the files in both directions, so it cannot fall
out of date without a test failing. Every check first runs on a synthetic listing where
the expected problems are known, so a clean result on the real one means something.
"""

import re
from pathlib import Path

import pytest

from fraud_detection import settings

ROADMAP = settings.DOCS_DIR / "ROADMAP.md"
PACKAGE_DIR = settings.ROOT_DIR / "src" / "fraud_detection"
ROOT_PACKAGE = "(raíz)"
EXISTS = "existe"
PLANNED = re.compile(r"fase \d+")
TABLE_HEADER = ("Paquete", "Módulo", "Estado")

Row = tuple[str, ...]


def _cells(line: str) -> Row | None:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    return tuple(cell.strip() for cell in stripped.strip("|").split("|"))


def _section_7(text: str) -> str:
    start = text.index("## 7. ")
    return text[start : text.index("\n## 8. ", start)]


def parse_module_table(section: str) -> list[Row]:
    """Return the rows of the table headed Paquete | Módulo | Estado, in order."""
    lines = iter(section.splitlines())
    for line in lines:
        if _cells(line) == TABLE_HEADER:
            next(lines)  # the |---|---|---| separator
            rows = []
            for row_line in lines:
                cells = _cells(row_line)
                if cells is None:
                    break
                rows.append(cells)
            return rows
    return []


def real_modules(package_dir: Path) -> set[tuple[str, str]]:
    """Return (package, module) for every .py file under the package but __init__.py."""
    modules = set()
    for path in package_dir.rglob("*.py"):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue
        parent = path.parent.relative_to(package_dir).as_posix()
        modules.add((ROOT_PACKAGE if parent == "." else parent, path.stem))
    return modules


def compare(rows: list[Row], real: set[tuple[str, str]]) -> list[str]:
    """Return one line per disagreement between the listing and the files."""
    problems = []
    listed = set()
    for row in rows:
        if len(row) != len(TABLE_HEADER):
            problems.append(f"malformed row: {row}")
            continue
        package, module, status = row
        name = f"{package}/{module}"
        if (package, module) in listed:
            problems.append(f"listed twice: {name}")
        listed.add((package, module))
        if status == EXISTS:
            if (package, module) not in real:
                problems.append(f"listed as existing, but there is no file: {name}")
        elif PLANNED.fullmatch(status):
            if (package, module) in real:
                problems.append(f"listed as planned ({status}), but the file exists: {name}")
        else:
            problems.append(f"unknown status {status!r}: {name}")
    for package, module in sorted(real - listed):
        problems.append(f"module not listed: {package}/{module}")
    return problems


def assert_listing_matches(rows: list[Row], real: set[tuple[str, str]]) -> None:
    problems = compare(rows, real)
    assert not problems, "ROADMAP section 7 and src/fraud_detection/ disagree:\n" + "\n".join(
        problems
    )


SYNTHETIC_SECTION = """## 7. Estructura

```
├── src/fraud_detection/           ← tabla de abajo
```

### Módulos

| Paquete | Módulo | Estado |
|---|---|---|
| (raíz) | config | existe |
| data | ghost | existe |
| models | estimators | fase 2 |

Texto después de la tabla.
"""


def test_the_parser_reads_the_module_table() -> None:
    assert parse_module_table(SYNTHETIC_SECTION) == [
        ("(raíz)", "config", "existe"),
        ("data", "ghost", "existe"),
        ("models", "estimators", "fase 2"),
    ]
    # A section with no such table yields no rows, never an exception swallowed.
    assert parse_module_table("## 7. Estructura\n\nNo table here.\n") == []


def test_one_module_too_many_and_one_too_few_fail_for_both_reasons() -> None:
    real = {("(raíz)", "config"), ("data", "loader")}
    rows = parse_module_table(SYNTHETIC_SECTION)
    # data/ghost is listed with no file; data/loader is a file with no row. The check is on
    # the list of problems, not on the assertion message, which pytest rewrites.
    assert compare(rows, real) == [
        "listed as existing, but there is no file: data/ghost",
        "module not listed: data/loader",
    ]
    with pytest.raises(AssertionError):
        assert_listing_matches(rows, real)
    # Negative control: the same listing without both faults passes.
    assert_listing_matches([rows[0], ("data", "loader", "existe"), rows[2]], real)


def test_the_other_disagreements_are_named() -> None:
    real = {("models", "baseline")}
    rows: list[Row] = [
        ("models", "baseline", "fase 2"),
        ("models", "estimators", "someday"),
        ("models", "estimators", "fase 2"),
        ("data", "loader"),
    ]
    assert compare(rows, real) == [
        "listed as planned (fase 2), but the file exists: models/baseline",
        "unknown status 'someday': models/estimators",
        "listed twice: models/estimators",
        "malformed row: ('data', 'loader')",
    ]


def test_roadmap_module_table_matches_the_package() -> None:
    rows = parse_module_table(_section_7(ROADMAP.read_text(encoding="utf-8")))
    real = real_modules(PACKAGE_DIR)
    # Controls: both sides are populated, so the comparison cannot pass vacuously.
    assert (ROOT_PACKAGE, "config", EXISTS) in rows
    assert (ROOT_PACKAGE, "config") in real
    assert_listing_matches(rows, real)
