#!/usr/bin/env python3
"""
Docstring Coverage Checker for Kannada Sentiment Analysis Project.

Scans all public modules, classes, and functions under ``src/`` and
reports which ones are missing docstrings.

Usage
-----
    python scripts/check_docstrings.py [--strict] [--module src.pipeline]

Flags
-----
--strict    Exit with code 1 if any missing docstrings are found.
--module    Check only the specified dotted module name.

Output
------
Prints a formatted table; exits 0 when all public symbols are documented,
or 1 (with ``--strict``) when any are missing.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator, NamedTuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

# Modules to check (dotted names, importable after inserting PROJECT_ROOT
# into sys.path)
DEFAULT_MODULES = [
    "src.preprocessing.language_detector",
    "src.preprocessing.transliterator",
    "src.preprocessing.translator",
    "src.models.sentiment_classifier",
    "src.pipeline",
]

# Prefixes that mark a symbol as private / implementation-detail; these are
# excluded from the public-API check.
PRIVATE_PREFIXES = ("_",)

# Names to always skip regardless of prefix (generated dunder methods etc.)
SKIP_NAMES = frozenset(
    {
        "__init_subclass__",
        "__class_getitem__",
        "__subclasshook__",
        "__abstractmethods__",
    }
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Violation(NamedTuple):
    """A single missing-docstring finding."""

    module: str
    kind: str   # "module", "class", "function", or "method"
    name: str   # fully qualified name within the module
    lineno: int  # best-effort source line number (0 if unavailable)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_public(name: str) -> bool:
    """Return True for names that are part of the public API."""
    return not name.startswith(PRIVATE_PREFIXES) and name not in SKIP_NAMES


def _lineno(obj: object) -> int:
    """Best-effort source line number; returns 0 on failure."""
    try:
        return inspect.getsourcelines(obj)[1]  # type: ignore[arg-type]
    except (TypeError, OSError):
        return 0


def _has_docstring(obj: object) -> bool:
    """Return True when *obj* has a non-empty docstring."""
    doc = inspect.getdoc(obj)
    return bool(doc and doc.strip())


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def _scan_class(
    cls: type,
    module_name: str,
    class_name: str,
) -> Iterator[Violation]:
    """Yield Violation instances for undocumented public methods of *cls*."""
    for attr_name, attr in inspect.getmembers(cls, predicate=None):
        if not _is_public(attr_name):
            continue

        # Skip if inherited from a parent class defined outside our src/
        if attr_name in cls.__dict__:
            obj = cls.__dict__[attr_name]
        else:
            continue  # inherited — not our responsibility

        # Unwrap staticmethod / classmethod wrappers
        raw = obj
        if isinstance(obj, (staticmethod, classmethod)):
            raw = obj.__func__

        if not callable(raw) and not isinstance(raw, property):
            continue

        # Properties: check the getter
        if isinstance(raw, property):
            raw = raw.fget  # type: ignore[assignment]
            if raw is None:
                continue

        if not _has_docstring(raw):
            yield Violation(
                module=module_name,
                kind="method",
                name=f"{class_name}.{attr_name}",
                lineno=_lineno(raw),
            )


def scan_module(module_name: str) -> list[Violation]:
    """Import *module_name* and return all missing-docstring findings."""
    violations: list[Violation] = []

    try:
        mod: ModuleType = importlib.import_module(module_name)
    except Exception as exc:
        print(f"  ERROR importing {module_name}: {exc}", file=sys.stderr)
        return violations

    # -- Module-level docstring ------------------------------------------------
    if not _has_docstring(mod):
        violations.append(
            Violation(module=module_name, kind="module", name=module_name, lineno=1)
        )

    # -- Walk public members ---------------------------------------------------
    for name, obj in inspect.getmembers(mod):
        if not _is_public(name):
            continue
        # Only check objects defined in this module (not re-exported imports)
        try:
            if inspect.getmodule(obj) is not mod:
                continue
        except Exception:
            continue

        if inspect.isclass(obj):
            # Class docstring
            if not _has_docstring(obj):
                violations.append(
                    Violation(
                        module=module_name,
                        kind="class",
                        name=name,
                        lineno=_lineno(obj),
                    )
                )
            # Method docstrings
            violations.extend(_scan_class(obj, module_name, name))

        elif inspect.isfunction(obj):
            if not _has_docstring(obj):
                violations.append(
                    Violation(
                        module=module_name,
                        kind="function",
                        name=name,
                        lineno=_lineno(obj),
                    )
                )

    return violations


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_KIND_ICON = {
    "module":   "📦",
    "class":    "🏛 ",
    "function": "🔧",
    "method":   "  🔹",
}


def print_report(all_violations: list[Violation], checked: list[str]) -> None:
    """Print a formatted summary to stdout."""
    width = 72
    print("=" * width)
    print("  DOCSTRING COVERAGE REPORT")
    print("=" * width)
    print(f"  Modules checked : {len(checked)}")
    print(f"  Total violations: {len(all_violations)}")
    print()

    if not all_violations:
        print("  ✅  All public symbols are documented!")
        print("=" * width)
        return

    # Group by module
    by_module: dict[str, list[Violation]] = {}
    for v in all_violations:
        by_module.setdefault(v.module, []).append(v)

    for mod_name, viols in sorted(by_module.items()):
        print(f"  📂 {mod_name}  ({len(viols)} missing)")
        print("  " + "-" * (width - 2))
        for v in sorted(viols, key=lambda x: x.lineno):
            icon = _KIND_ICON.get(v.kind, "  ?")
            line_hint = f"line {v.lineno}" if v.lineno else "line ?"
            print(f"  {icon}  {v.name:<45}  [{line_hint}]")
        print()

    print("=" * width)
    print("  ❌  Fix the violations above and re-run this script.")
    print("=" * width)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that all public symbols in src/ have docstrings."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any missing docstrings are found.",
    )
    parser.add_argument(
        "--module",
        metavar="MODULE",
        help="Check only this dotted module (e.g. src.pipeline).",
    )
    args = parser.parse_args()

    # Make the project importable
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    modules_to_check = [args.module] if args.module else DEFAULT_MODULES

    all_violations: list[Violation] = []
    for mod_name in modules_to_check:
        all_violations.extend(scan_module(mod_name))

    print_report(all_violations, modules_to_check)

    if args.strict and all_violations:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
