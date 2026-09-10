#!/usr/bin/env python3
"""Check that every scaffold schema is an exact copy of its root contract."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_ROOT = REPOSITORY_ROOT / "contracts"
SCAFFOLD_ROOT = (
    REPOSITORY_ROOT / "template" / "skeleton" / "data" / "specification" / "schemas"
)


def main() -> int:
    errors = []
    expected = sorted(CONTRACT_ROOT.glob("*.json"))
    scaffold_names = {
        path.name.removesuffix(".jinja") for path in SCAFFOLD_ROOT.glob("*.json.jinja")
    }
    contract_names = {path.name for path in expected}
    if scaffold_names != contract_names:
        errors.append("root and scaffold schema filenames differ")
    for contract in expected:
        scaffold = SCAFFOLD_ROOT / f"{contract.name}.jinja"
        if not scaffold.is_file() or scaffold.read_bytes() != contract.read_bytes():
            errors.append(f"scaffold schema differs: {contract.name}")

    runtime_root = (
        REPOSITORY_ROOT
        / "template"
        / "skeleton"
        / "src"
        / "{{ package_name }}"
        / "business_logic"
    )
    runtime_contracts = {
        "application.schema.json": CONTRACT_ROOT / "application.schema.json",
        "release.schema.json": CONTRACT_ROOT / "release-manifest.schema.json",
    }
    for name, contract in runtime_contracts.items():
        runtime = runtime_root / name
        if not runtime.is_file() or runtime.read_bytes() != contract.read_bytes():
            errors.append(f"runtime schema differs: {name}")

    if errors:
        print("Contract synchronization failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Contract schemas synchronized: {len(expected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
