#!/usr/bin/env python3
"""prep.py - load, clean, split and save the train/test artifacts (second CI/CD job).

Usage: python model_building/prep.py
Outputs (artifacts/): Xtrain/Xtest/ytrain/ytest.csv, train.csv, test.csv,
                      split_metadata.json, cleaning_report.json
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "engine_data.csv"
ARTIFACT_DIR = BASE / "artifacts"
TARGET = "engine_condition"          # target name after standardisation
TEST_SIZE, RANDOM_STATE = 0.20, 42
CORR_LIMIT = 0.95


def snake_case(name):
    return re.sub(r"\W+", "_", name.strip().lower()).strip("_")


def find_unnecessary_columns(df, features):
    """Return {column: reason} for constant, ID-like and highly correlated columns."""
    drop = {}
    for col in features:
        if df[col].nunique() <= 1:
            drop[col] = "constant"
        elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() == len(df):
            drop[col] = "identifier-like"
    corr = df[[c for c in features if c not in drop]].corr().abs()
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            if corr.loc[a, b] > CORR_LIMIT and b not in drop:
                drop[b] = f"|r| = {corr.loc[a, b]:.2f} with {a}"
    return drop


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    print(f"Loaded {raw.shape[0]:,} rows x {raw.shape[1]} columns from {DATA_PATH}")

    df = raw.rename(columns=snake_case)
    renamed = {old: new for old, new in zip(raw.columns, df.columns) if old != new}

    n_start = len(df)
    df = df.drop_duplicates().dropna(subset=[TARGET]).reset_index(drop=True)
    n_removed = n_start - len(df)
    features = [c for c in df.columns if c != TARGET]
    negative_cells = int((df[features] < 0).sum().sum())

    dropped = find_unnecessary_columns(df, features)
    features = [c for c in features if c not in dropped]
    df = df[features + [TARGET]]
    assert set(df[TARGET].unique()) == {0, 1}, "target must contain both classes 0 and 1"

    X, y = df[features], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    X_train.to_csv(ARTIFACT_DIR / "Xtrain.csv", index=False)
    X_test.to_csv(ARTIFACT_DIR / "Xtest.csv", index=False)
    y_train.to_csv(ARTIFACT_DIR / "ytrain.csv", index=False)
    y_test.to_csv(ARTIFACT_DIR / "ytest.csv", index=False)
    pd.concat([X_train, y_train], axis=1).to_csv(ARTIFACT_DIR / "train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(ARTIFACT_DIR / "test.csv", index=False)

    report = {"rows_loaded": n_start, "rows_removed_duplicates_or_missing_target": n_removed,
              "negative_sensor_cells": negative_cells, "renamed_columns": renamed,
              "dropped_columns": dropped, "feature_columns": features}
    metadata = {"source_file": str(DATA_PATH.relative_to(BASE)), "total_rows": len(df),
                "train_rows": len(X_train), "test_rows": len(X_test), "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE, "target_column": TARGET, "feature_columns": features,
                "train_faulty_share": round(float(y_train.mean()), 4),
                "test_faulty_share": round(float(y_test.mean()), 4)}
    (ARTIFACT_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    (ARTIFACT_DIR / "split_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Rows removed (duplicates / missing target): {n_removed}")
    print(f"Columns dropped: {dropped or 'none'}")
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Faulty share  train = {y_train.mean():.2%} | test = {y_test.mean():.2%}")
    print("Artifacts written to", ARTIFACT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
