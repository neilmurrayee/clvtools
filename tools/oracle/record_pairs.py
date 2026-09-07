#!/usr/bin/env python3
"""Re-record the paired oracle from a live R session.

    R_LIBS=.Rlib uv run python tools/oracle/record_pairs.py

Runs every pair's R snippet once and writes what came back to
``tests/fixtures/pairs/``, so that the ordinary suite can check the Python side
against it without R. See ``tests/pairs.py`` for the design.

This is a script and not a pytest mode on purpose. Recording is the one
operation that makes a failing pair pass, and a suite able to rewrite its own
expectations is one bad flag away from proving nothing.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import pairs  # noqa: E402
import pairs_pnbd  # noqa: E402,F401 - imported for the pairs it registers


def dump(provenance: dict[str, str], field: str, body: dict[str, list[float]]) -> str:
    """JSON with each vector on one line.

    ``json.dumps(indent=1)`` puts every one of 600 doubles on a line of its
    own, which turns a re-recording into a diff nobody reads. One line per case
    keeps the file reviewable: a changed expression shows up as one changed
    line, and a changed *set* of cases as an added or removed one.
    """
    head = ",\n".join(f' "{k}": {json.dumps(v)}' for k, v in provenance.items())
    rows = ",\n".join(f'  {json.dumps(k)}: {json.dumps(v)}' for k, v in body.items())
    return "{\n" + head + f',\n "{field}": {{\n' + rows + "\n }\n}\n"


def main() -> int:
    registry = pairs.registry()
    if not registry:
        print("no pairs registered")
        return 1

    print(f"{len(pairs.cases())} cases from {len(registry)} pairs")
    with tempfile.TemporaryDirectory() as tmp:
        result = pairs.run_r(registry, Path(tmp))

    provenance = {
        "clvtools.version": result["clvtools.version"],
        "r.version": result["r.version"],
        "generated.by": "tools/oracle/record_pairs.py",
    }

    pairs.PAIRS.mkdir(parents=True, exist_ok=True)

    # One file per prelude, holding the vectors it fed both implementations.
    # Recorded separately from the values because a prelude is shared: two
    # families naming the same one must be reading one copy of its inputs.
    for name, inputs in result["preludes"].items():
        path = pairs.PAIRS / f"_prelude_{name}.json"
        path.write_text(dump(provenance, "inputs", inputs))
        print(f"  {path.name}")

    # One file per family, so a diff stays readable.
    by_family: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for pair in registry:
        for input_name in pair.inputs:
            k = pairs.key(pair.id, input_name)
            by_family[pair.family][k] = result["values"][k]

    for family, values in by_family.items():
        path = pairs.PAIRS / f"{family}.json"
        path.write_text(dump(provenance, "values", dict(sorted(values.items()))))
        print(f"  {path.name}  {len(values)} cases")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
