"""model_utils.py - shared modelling code used by the notebook and by train.py (CI/CD)."""
import json
import logging
import os
import shutil
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)   # hide the pickle-format advisory

SEED = 42
CV_FOLDS = 5
HIGH_RECALL_TARGET = 0.90          # recall on Faulty engines required by the alternative "high-recall" threshold
EXPERIMENT = "Predictive_Maintenance_Engine"
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "balanced_accuracy": "balanced_accuracy"}
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
BASELINE = "Baseline (always Faulty)"


# ---------------------------------------------------------------- data and pipeline
def load_splits(artifact_dir):
    """Load the train/test artifacts produced by prep.py."""
    artifact_dir = Path(artifact_dir)
    X_train, X_test = (pd.read_csv(artifact_dir / f) for f in ("Xtrain.csv", "Xtest.csv"))
    y_train, y_test = (pd.read_csv(artifact_dir / f).iloc[:, 0] for f in ("ytrain.csv", "ytest.csv"))
    return X_train, X_test, y_train, y_test


def make_cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def build_pipeline(estimator):
    """Median imputation (defensive) followed by the classifier."""
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", estimator)])


def get_model_configs():
    """(name, estimator, parameter grid) for the baseline and the six algorithms."""
    def tree(**kw):
        return DecisionTreeClassifier(random_state=SEED, **kw)

    return [
        (BASELINE, DummyClassifier(strategy="prior"), {}),
        ("Decision Tree", tree(),
         {"classifier__max_depth": [3, 5, 8, 12], "classifier__min_samples_leaf": [10, 30, 60]}),
        ("Bagging", BaggingClassifier(estimator=tree(min_samples_leaf=10), random_state=SEED),
         {"classifier__n_estimators": [100, 200], "classifier__max_samples": [0.3, 0.5, 0.8]}),
        ("Random Forest", RandomForestClassifier(random_state=SEED),
         {"classifier__n_estimators": [200], "classifier__max_depth": [4, 6, 10],
          "classifier__min_samples_leaf": [30, 60, 100]}),
        ("AdaBoost", AdaBoostClassifier(estimator=tree(max_depth=1), random_state=SEED),
         {"classifier__estimator__max_depth": [1, 2, 3], "classifier__n_estimators": [100, 200],
          "classifier__learning_rate": [0.1, 0.3]}),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=SEED),
         {"classifier__n_estimators": [100, 200], "classifier__learning_rate": [0.05, 0.1],
          "classifier__max_depth": [1, 2, 3]}),
        ("XGBoost", XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1),
         {"classifier__n_estimators": [100, 300, 500], "classifier__max_depth": [1, 2, 3, 4],
          "classifier__learning_rate": [0.01, 0.03, 0.1]}),
    ]


# ---------------------------------------------------------------- experiment tracking
def setup_mlflow(project_dir, reset=False):
    """Point MLflow at a local SQLite store inside the project folder."""
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    project_dir = Path(project_dir).resolve()
    db, store = project_dir / "mlflow.db", project_dir / "mlruns"
    if reset:
        db.unlink(missing_ok=True)
        shutil.rmtree(store, ignore_errors=True)
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=store.as_uri())
    mlflow.set_experiment(EXPERIMENT)


def _clean_params(params):
    """Drop the pipeline prefix; values keep their original type (MLflow stringifies them when logging)."""
    return {k.replace("classifier__", ""): v for k, v in params.items()}


def _json_default(obj):
    """JSON fallback for numpy scalars (keeps ints as ints)."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def _log_to_mlflow(result):
    """One parent run per algorithm, one nested run per tuned parameter combination."""
    row, cv_results = result["summary"], result["cv_results"]
    with mlflow.start_run(run_name=row["Algorithm"]):
        mlflow.log_param("algorithm", row["Algorithm"])
        mlflow.log_params({f"best_{k}": v for k, v in result["params"].items()})
        mlflow.log_metrics({"cv_roc_auc": row["CV ROC-AUC"], "cv_roc_auc_sd": row["CV ROC-AUC SD"],
                            "cv_pr_auc": row["CV PR-AUC"], "cv_balanced_accuracy": row["CV Balanced Acc"],
                            "train_roc_auc": row["Train ROC-AUC"], "roc_auc_gap": row["ROC-AUC Gap"],
                            "n_combinations": row["Combinations"]})
        for i, combo in cv_results.iterrows():          # every tuned combination is logged
            with mlflow.start_run(run_name=f"{row['Algorithm']} | combo {i:02d}", nested=True):
                mlflow.log_params(_clean_params(combo["params"]))
                mlflow.log_metrics({"cv_roc_auc": combo["mean_test_roc_auc"],
                                    "cv_roc_auc_sd": combo["std_test_roc_auc"],
                                    "cv_pr_auc": combo["mean_test_pr_auc"],
                                    "cv_balanced_accuracy": combo["mean_test_balanced_accuracy"],
                                    "train_roc_auc": combo["mean_train_roc_auc"]})
        try:
            mlflow.sklearn.log_model(result["pipeline"], name="model", serialization_format="cloudpickle")
        except TypeError:                                # older MLflow versions
            mlflow.sklearn.log_model(result["pipeline"], artifact_path="model", serialization_format="cloudpickle")


def tune_model(name, estimator, grid, X, y, cv=None, track=True):
    """Tune one algorithm with GridSearchCV (or cross-validate the baseline) and log it to MLflow."""
    cv = cv or make_cv()
    pipe = build_pipeline(estimator)
    metrics = ("test_roc_auc", "test_pr_auc", "test_balanced_accuracy", "train_roc_auc")
    if grid:
        search = GridSearchCV(pipe, grid, scoring=SCORING, refit="roc_auc", cv=cv,
                              return_train_score=True, n_jobs=-1).fit(X, y)
        cv_results, best = pd.DataFrame(search.cv_results_), search.best_estimator_
        params, row = _clean_params(search.best_params_), cv_results.loc[search.best_index_]
        means = {m: float(row[f"mean_{m}"]) for m in metrics}
        sd = float(row["std_test_roc_auc"])
    else:
        scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True)
        cv_results, best, params = pd.DataFrame(), pipe.fit(X, y), {}
        means = {m: float(np.mean(scores[m])) for m in metrics}
        sd = float(np.std(scores["test_roc_auc"]))
    summary = {"Algorithm": name, "CV ROC-AUC": means["test_roc_auc"], "CV ROC-AUC SD": sd,
               "CV PR-AUC": means["test_pr_auc"], "CV Balanced Acc": means["test_balanced_accuracy"],
               "Train ROC-AUC": means["train_roc_auc"],
               "Combinations": max(len(cv_results), 1),
               "Best parameters": ", ".join(f"{k}={v}" for k, v in params.items()) or "-"}
    summary["ROC-AUC Gap"] = summary["Train ROC-AUC"] - summary["CV ROC-AUC"]
    result = {"summary": summary, "pipeline": best, "params": params, "cv_results": cv_results}
    if track:
        _log_to_mlflow(result)
    return result


def tune_all(X, y, verbose=True):
    """Tune every configured model; returns a list of result dictionaries."""
    configs, results = get_model_configs(), []
    for i, (name, estimator, grid) in enumerate(configs, 1):
        start = time.time()
        results.append(tune_model(name, estimator, grid, X, y))
        if verbose:
            s = results[-1]["summary"]
            print(f"[{i}/{len(configs)}] {name:26s} CV ROC-AUC = {s['CV ROC-AUC']:.3f} (+/- {s['CV ROC-AUC SD']:.3f}) "
                  f"| {s['Combinations']:2d} combinations | {time.time() - start:5.1f}s")
    return results


def summary_frame(results):
    """Cross-validation summary, one row per algorithm, ranked by CV ROC-AUC."""
    return (pd.DataFrame([r["summary"] for r in results])
            .sort_values("CV ROC-AUC", ascending=False).set_index("Algorithm"))


def one_se_candidates(results):
    """Algorithms within one standard error of the best mean ROC-AUC (the baseline is excluded)."""
    models = [r["summary"] for r in results if r["summary"]["Algorithm"] != BASELINE]
    best = max(models, key=lambda s: s["CV ROC-AUC"])
    se = best["CV ROC-AUC SD"] / np.sqrt(CV_FOLDS)
    names = [s["Algorithm"] for s in models if s["CV ROC-AUC"] >= best["CV ROC-AUC"] - se]
    return names, best["Algorithm"], se


def select_champion(results):
    """One-standard-error rule: among the statistically tied leaders, take the smallest train-vs-validation gap."""
    names, _, _ = one_se_candidates(results)
    pool = [r for r in results if r["summary"]["Algorithm"] in names]
    return min(pool, key=lambda r: (r["summary"]["ROC-AUC Gap"], -r["summary"]["CV ROC-AUC"]))


def edge_parameters(configs, results):
    """For each algorithm, the tuned numeric parameters whose best value sits at the edge of the searched grid."""
    best = {r["summary"]["Algorithm"]: r["params"] for r in results}
    edges = {}
    for name, _, grid in configs:
        flags = []
        for key, values in grid.items():
            param = key.replace("classifier__", "")
            numeric = [v for v in values if isinstance(v, (int, float))]
            if len(values) > 1 and len(numeric) == len(values) and param in best.get(name, {}):
                value = best[name][param]
                if value == min(numeric):
                    flags.append(f"{param}={value} (lowest in grid)")
                elif value == max(numeric):
                    flags.append(f"{param}={value} (highest in grid)")
        edges[name] = flags
    return edges


# ---------------------------------------------------------------- evaluation helpers
def evaluate(y_true, proba, threshold=0.5):
    """All reporting metrics for one set of probabilities at one threshold."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {"Accuracy": accuracy_score(y_true, pred),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall (Faulty)": recall_score(y_true, pred, zero_division=0),
            "Specificity (Normal)": recall_score(y_true, pred, pos_label=0, zero_division=0),
            "F1 (Faulty)": f1_score(y_true, pred, zero_division=0),
            "Macro F1": f1_score(y_true, pred, average="macro", zero_division=0),
            "Balanced Acc": balanced_accuracy_score(y_true, pred),
            "MCC": matthews_corrcoef(y_true, pred),
            "ROC-AUC": roc_auc_score(y_true, proba),
            "PR-AUC": average_precision_score(y_true, proba)}


def oof_probabilities(pipe, X, y, cv=None):
    """Out-of-fold probability of the Faulty class for the training data."""
    return cross_val_predict(clone(pipe), X, y, cv=cv or make_cv(), method="predict_proba")[:, 1]


def threshold_sweep(y, proba):
    rows = []
    for t in THRESHOLDS:
        pred = (proba >= t).astype(int)
        rows.append({"threshold": t, "precision": precision_score(y, pred, zero_division=0),
                     "recall": recall_score(y, pred, zero_division=0),
                     "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
                     "f1": f1_score(y, pred, zero_division=0),
                     "balanced_accuracy": balanced_accuracy_score(y, pred)})
    return pd.DataFrame(rows)


def best_threshold(sweep, criterion="balanced_accuracy"):
    return float(sweep.loc[sweep[criterion].idxmax(), "threshold"])


def threshold_for_recall(sweep, target=HIGH_RECALL_TARGET):
    """Alternative operating point: the highest threshold whose out-of-fold recall on Faulty engines >= target."""
    ok = sweep[sweep["recall"] >= target]
    return float(ok["threshold"].max()) if len(ok) else float(sweep["threshold"].min())


def tune_threshold(pipe, X, y):
    """Out-of-fold probabilities, threshold sweep and the balanced-accuracy threshold (training data only)."""
    oof = oof_probabilities(pipe, X, y)
    sweep = threshold_sweep(y, oof)
    return oof, sweep, best_threshold(sweep)


def cost_per_1000(y_true, pred, fn_cost, fp_cost=1.0):
    """Expected cost per 1,000 engines (a missed fault costs fn_cost, a false alarm fp_cost)."""
    y, pred = np.asarray(y_true), np.asarray(pred)
    fn, fp = int(((y == 1) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
    return (fn * fn_cost + fp * fp_cost) / len(y) * 1000


def cost_optimal_threshold(y, proba, fn_cost):
    costs = [cost_per_1000(y, (proba >= t).astype(int), fn_cost) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmin(costs))])


def lift_table(y, proba, bins=10):
    """Risk deciles: decile 1 holds the engines the model considers most likely to be faulty."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(proba)}).sort_values("p", ascending=False)
    d["Risk decile"] = pd.qcut(np.arange(len(d)), bins, labels=range(1, bins + 1))
    g = d.groupby("Risk decile", observed=True)["y"].agg(Engines="size", Faulty="sum")
    g["Faulty rate %"] = g["Faulty"] / g["Engines"] * 100
    g["Lift vs. average"] = g["Faulty rate %"] / (d["y"].mean() * 100)
    g["Cumulative faults found %"] = g["Faulty"].cumsum() / g["Faulty"].sum() * 100
    return g


def feature_importance(pipe, X, y, repeats=10):
    """Permutation importance as the drop in ROC-AUC (percentage points) when a feature is shuffled."""
    r = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=repeats, random_state=SEED, n_jobs=-1)
    out = pd.DataFrame({"Feature": X.columns, "AUC drop (pp)": r.importances_mean * 100,
                        "Std (pp)": r.importances_std * 100})
    return out.sort_values("AUC drop (pp)", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- persistence
def save_reports(art_dir, results, sweep, importance):
    """Write the three supporting CSV files (same layout from the notebook and from CI)."""
    art_dir = Path(art_dir)
    summary_frame(results).reset_index().to_csv(art_dir / "cv_summary.csv", index=False)
    sweep.to_csv(art_dir / "threshold_sweep.csv", index=False)
    importance.to_csv(art_dir / "permutation_importance.csv", index=False)


def card_extra(champion, test_metrics, sweep, n_train, n_test):
    """Model-card fields that depend on the training run (shared by the notebook and run_pipeline)."""
    return {"cv_roc_auc": champion["summary"]["CV ROC-AUC"], "best_parameters": champion["params"],
            "high_recall_threshold": threshold_for_recall(sweep), "test_metrics": test_metrics,
            "train_rows": n_train, "test_rows": n_test}


def save_bundle(deploy_dir, pipe, threshold, features, name, extra):
    """Save the model bundle and a model card to the deployment folder."""
    deploy_dir = Path(deploy_dir)
    deploy_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipe, "threshold": threshold, "feature_columns": list(features),
                 "model_type": name}, deploy_dir / "best_model.joblib")
    card = {"model_name": name, "task": "Binary classification - predictive maintenance",
            "target": "engine_condition (0 = Normal, 1 = Faulty)", "decision_threshold": threshold,
            "threshold_rule": "maximise out-of-fold balanced accuracy on the training set",
            "feature_columns": list(features), **extra,
            "limitations": ["Decision-support tool, not for autonomous safety-critical decisions",
                            "Moderate discrimination (see test_metrics); use for ranking and triage",
                            "Trained on historical snapshots without time, engine age or maintenance history",
                            "Retrain when sensor distributions drift"]}
    (deploy_dir / "model_card.json").write_text(json.dumps(card, indent=2, default=_json_default))
    return card


def run_pipeline(base_dir, reset_tracking=True):
    """End-to-end training used by CI: tune, select, tune threshold, evaluate, save."""
    base = Path(base_dir)
    art, deploy = base / "artifacts", base / "deployment"
    X_train, X_test, y_train, y_test = load_splits(art)
    setup_mlflow(base, reset=reset_tracking)
    results = tune_all(X_train, y_train)
    champion = select_champion(results)
    name, pipe = champion["summary"]["Algorithm"], champion["pipeline"]
    _, sweep, threshold = tune_threshold(pipe, X_train, y_train)
    test_metrics = evaluate(y_test, pipe.predict_proba(X_test)[:, 1], threshold)
    save_reports(art, results, sweep, feature_importance(pipe, X_test, y_test))
    card = save_bundle(deploy, pipe, threshold, X_train.columns, name,
                       card_extra(champion, test_metrics, sweep, len(X_train), len(X_test)))
    print(f"Champion: {name} | threshold = {threshold:.2f} | test ROC-AUC = {test_metrics['ROC-AUC']:.3f}")
    return card


def verify_saved_model(base_dir):
    """Reload the saved bundle and confirm it reproduces the metrics stored in the model card."""
    base = Path(base_dir)
    bundle = joblib.load(base / "deployment" / "best_model.joblib")
    card = json.loads((base / "deployment" / "model_card.json").read_text())
    _, X_test, _, y_test = load_splits(base / "artifacts")
    proba = bundle["model"].predict_proba(X_test[bundle["feature_columns"]])[:, 1]
    fresh = evaluate(y_test, proba, bundle["threshold"])
    worst = max(abs(fresh[k] - card["test_metrics"][k]) for k in fresh)
    assert worst < 1e-9, f"Saved model does not reproduce the model card (max diff {worst:.2e})"
    print(f"Saved model verified: reproduces {len(fresh)} test metrics (max difference {worst:.1e}).")
    return fresh
