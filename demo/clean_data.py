"""
prepare_insulated_test_box.py

Purpose
-------
Load test_box.xlsx, keep only the insulated_concrete sheet,
fill missing numeric values with linear interpolation, and
save a cleaned Excel file for later use.

Usage
-----
python prepare_insulated_test_box.py

By default, this script expects test_box.xlsx in the same folder.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


INPUT_XLSX = "data/test_box.xlsx"
SHEET_NAME = "insulated_concrete"
OUTPUT_XLSX = "data/test_box_insulated_cleaned.xlsx"


def main() -> None:
    input_path = Path(INPUT_XLSX)
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path.resolve()}")

    df = pd.read_excel(input_path, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]

    numeric_cols = [c for c in ["Ti", "Ta", "P", "dt"] if c in df.columns]
    if not numeric_cols:
        raise ValueError("Did not find expected numeric columns among: Ti, Ta, P, dt")

    if "time" in df.columns:
        parsed = pd.to_datetime(df["time"], errors="coerce")
        if parsed.notna().any():
            df["time"] = parsed
            df = df.sort_values("time").reset_index(drop=True)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    output_path = Path(OUTPUT_XLSX)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="insulated_concrete", index=False)

    print(f"Saved cleaned file to: {output_path.resolve()}")
    print(f"Rows kept: {len(df)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
