from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local PTB-XL records100 directory.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="PTB-XL root containing ptbxl_database.csv, scp_statements.csv, and records100/. "
        "Passing the records100 path directly is also accepted.",
    )
    return parser.parse_args()


def normalize_data_dir(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if root.name.lower() == "records100":
        root = root.parent
    return root


def main() -> None:
    args = parse_args()
    root = normalize_data_dir(args.data_dir)
    records100 = root / "records100"
    metadata = root / "ptbxl_database.csv"
    scp = root / "scp_statements.csv"

    print(f"PTB-XL root: {root}")
    print(f"records100: {records100}")
    if not records100.exists():
        raise SystemExit("records100/ was not found. Set PTBXL_DATA_DIR to the PTB-XL root or records100 path.")

    dat_count = sum(1 for _ in records100.rglob("*.dat"))
    hea_count = sum(1 for _ in records100.rglob("*.hea"))
    print(f".dat files: {dat_count}")
    print(f".hea files: {hea_count}")

    if metadata.exists():
        print("ptbxl_database.csv: found")
    else:
        print("ptbxl_database.csv: missing; scripts may try to download metadata if internet is available.")

    if scp.exists():
        print("scp_statements.csv: found")
    else:
        print("scp_statements.csv: missing; scripts may try to download metadata if internet is available.")

    if dat_count < 20000 or hea_count < 20000:
        raise SystemExit("records100 looks incomplete. Expected more than 20,000 .dat and .hea files.")

    print("Local PTB-XL records100 check passed.")


if __name__ == "__main__":
    main()
