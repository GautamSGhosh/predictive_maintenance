# Predictive Maintenance - engine condition classifier

End-to-end MLOps project: data validation, preparation, tuned models with MLflow tracking, a Streamlit app and a
GitHub Actions pipeline that retrains the model and commits it to `main`.

* Live app: https://predictivemaintenance-gautamsghosh-capstone.streamlit.app
* Pipeline: `.github/workflows/pipeline.yml` (register-data -> prepare-data -> train-model -> verify-deployment)

| Folder | Content |
|---|---|
| `predictive_maintenance/data/` | registered dataset `engine_data.csv` |
| `predictive_maintenance/model_building/` | `data_register.py`, `prep.py`, `model_utils.py`, `train.py`, `smoke_test.py`, `requirements.txt` |
| `predictive_maintenance/artifacts/` | train/test splits (published between jobs as the `data-splits` workflow artifact) |
| `predictive_maintenance/deployment/` | `app.py`, `requirements.txt`, `feature_stats.json`, `best_model.joblib`, `model_card.json` |

Run locally:

    pip install -r predictive_maintenance/model_building/requirements.txt
    python predictive_maintenance/model_building/data_register.py
    python predictive_maintenance/model_building/prep.py
    python predictive_maintenance/model_building/train.py
    streamlit run predictive_maintenance/deployment/app.py

Responsible use: the model ranks engine snapshots for inspection prioritisation. It is decision support, not an
autonomous safety system. Monitor sensor drift and retrain when the operating population changes.
