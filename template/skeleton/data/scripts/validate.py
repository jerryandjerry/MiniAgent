"""Validate the compiled application contract."""

from _contract import ContractError, CONTRACT_PATH, load_contract, require_approved


def main() -> int:
    try:
        payload = load_contract()
        require_approved(payload)
    except ContractError as exc:
        print(f"Contract action: {exc}")
        return 2

    print(
        f"Validated {CONTRACT_PATH}: "
        f"{len(payload['artifacts']['workflows'])} workflow locations and "
        f"{len(payload['artifacts']['tools'])} tool locations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
