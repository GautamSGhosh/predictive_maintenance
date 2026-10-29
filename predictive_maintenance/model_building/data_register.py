#!/usr/bin/env python3
"""data_register.py - dataset registration and schema validation (first CI/CD job).

Exit code 0 = dataset valid and registered; 1 = at least one check failed.
Usage: python model_building/data_register.py
"""
import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "engine_data.csv"
# Schema is defined here on purpose: this script must run stand-alone in the CI job.
FEATURES = ["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure",
            "lub oil temp", "Coolant temp"]          # exact CSV headers
TARGET = "Engine Condition"
MIN_ROWS = 1000

results = []


def check(name, passed, detail=""):
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}".rstrip())


def main():
    print("=" * 70, "\nDATA REGISTRATION & SCHEMA VALIDATION\n" + "=" * 70, sep="")
    check("File exists", DATA_PATH.exists(), f"({DATA_PATH})")
    if not DATA_PATH.exists():
        return 1
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    extra = [c for c in df.columns if c not in FEATURES + [TARGET]]
    check("Expected columns present", not missing, f"missing={missing}" if missing else f"({len(FEATURES) + 1} columns)")
    check("No unexpected columns", not extra, f"extra={extra}" if extra else "")
    if missing:
        return 1

    check("Minimum row count", len(df) >= MIN_ROWS, f"({len(df):,} rows)")
    check("Feature columns are numeric", all(pd.api.types.is_numeric_dtype(df[c]) for c in FEATURES))
    check("No missing values", int(df.isna().sum().sum()) == 0, f"({int(df.isna().sum().sum())} missing cells)")
    check("Sensor values are non-negative", bool((df[FEATURES] >= 0).all().all()))
    check("Target is binary {0, 1}", set(df[TARGET].unique()) <= {0, 1})

    print("\n" + "=" * 70, "\nDATASET SUMMARY\n" + "=" * 70, sep="")
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]} | Duplicate rows: {int(df.duplicated().sum())}")
    counts = df[TARGET].value_counts().sort_index()
    for value, count in counts.items():
        label = "Normal" if value == 0 else "Faulty"
        print(f"Class {value} ({label}): {count:,} ({count / len(df):.2%})")
    print("\nData types and ranges:")
    print(df.agg(["min", "max"]).T.assign(dtype=df.dtypes.astype(str)).round(2).to_string())

    ok = all(results)
    print("\n" + ("DATASET VALIDATED AND REGISTERED" if ok else "REGISTRATION FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
