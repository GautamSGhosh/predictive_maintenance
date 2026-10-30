#!/usr/bin/env python3
"""smoke_test.py - fourth CI job: prove that the model committed to the repository can be served."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DEPLOY = BASE / "deployment"


def main():
    """Score two sample engines with the committed model and run the Streamlit app headlessly."""
    bundle = joblib.load(DEPLOY / "best_model.joblib")
    card = json.loads((DEPLOY / "model_card.json").read_text())
    stats = json.loads((DEPLOY / "feature_stats.json").read_text())
    features = bundle["feature_columns"]
    rows = pd.DataFrame([{f: stats[f]["median"] for f in features}, {f: stats[f]["p99"] for f in features}])[features]
    proba = bundle["model"].predict_proba(rows)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all(), "probabilities outside [0, 1]"
    print(f"Committed model: {card['model_name']} | threshold {bundle['threshold']:.2f} | sample probabilities {proba.round(3).tolist()}")

    from streamlit.testing.v1 import AppTest          # run the real app headlessly
    app = AppTest.from_file(str(DEPLOY / "app.py"), default_timeout=120).run()
    assert not app.exception, app.exception
    app.button[0].click().run()
    assert not app.exception, app.exception
    assert app.metric, "the app did not display a prediction"
    print("Streamlit app started and returned a prediction:", app.metric[0].value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
