# --- Ensure PYTHONPATH includes src so titanic_lab imports work ---
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np  # noqa: F401
import argparse
import json
import pandas as pd
import time
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# NEW: import build_model and OUT_SUB from your modular script
from titanic_lab.model_titanic import load_train, load_test, build_model, OUT_SUB

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# storage for metrics
OUT_METRICS = Path("outputs/metrics")
OUT_METRICS.mkdir(parents=True, exist_ok=True)


def evaluate_holdout(
    seed: int = 42, *, algo: str = "logreg", model_path: Path | None = None
) -> dict:
    """Quick sanity-check: single stratified holdout with rich diagnostics.

    If model_path is given, loads a prefit Pipeline and evaluates it (no refit).
    Otherwise builds `build_model(algo)` and fits on the holdout train split.
    """
    df = load_train()
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    if model_path is not None:
        pipe = joblib.load(model_path)
        # assume artifact is fully fit (e.g., XGB with early stopping frozen)
    else:
        pipe = build_model(algo=algo)
        pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_val)
    proba_ok = hasattr(pipe, "predict_proba")
    y_proba = pipe.predict_proba(X_val)[:, 1] if proba_ok else None

    acc = float(accuracy_score(y_val, y_pred))
    auc = float(roc_auc_score(y_val, y_proba)) if y_proba is not None else float("nan")
    prec = float(precision_score(y_val, y_pred, pos_label=1))
    rec = float(recall_score(y_val, y_pred, pos_label=1))
    f1_ = float(f1_score(y_val, y_pred, pos_label=1))

    cr_text = classification_report(y_val, y_pred, digits=3)
    cm = confusion_matrix(y_val, y_pred)

    print(
        f"[HOLDOUT] accuracy = {acc:.3f} | ROC-AUC = {auc:.3f} | "
        f"precision(1) = {prec:.3f} | recall(1) = {rec:.3f} | F1(1) = {f1_:.3f}"
    )
    print("\nClassification report:\n", cr_text)
    print("\nConfusion matrix (rows=true, cols=pred):\n", cm)

    return {
        "mode": "holdout",
        "algo": algo,
        "used_model_path": str(model_path) if model_path else None,
        "acc": acc,
        "auc": auc,
        "precision_pos": prec,
        "recall_pos": rec,
        "f1_pos": f1_,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_val, y_pred, output_dict=True),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def evaluate_cv(
    n_splits: int = 5, seed: int = 42, verbose: bool = False, *, algo: str = "logreg"
) -> dict:
    """Stratified K-fold CV with accuracy (Kaggle metric) + ROC-AUC + precision/recall/F1.

    Always builds a fresh estimator per fold via build_model(algo) so refitting is correct.
    """
    df = load_train()
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipe = build_model(algo=algo)

    res = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring={
            "acc": "accuracy",
            "roc": "roc_auc",
            "prec": "precision",
            "rec": "recall",
            "f1": "f1",
        },
        return_train_score=False,
        n_jobs=-1,
    )

    acc_mean, acc_std = float(res["test_acc"].mean()), float(res["test_acc"].std())
    roc_mean, roc_std = float(res["test_roc"].mean()), float(res["test_roc"].std())
    prec_mean, prec_std = float(res["test_prec"].mean()), float(res["test_prec"].std())
    rec_mean, rec_std = float(res["test_rec"].mean()), float(res["test_rec"].std())
    f1_mean, f1_std = float(res["test_f1"].mean()), float(res["test_f1"].std())

    metrics = {
        "schema_version": 1,
        "mode": "cv",
        "algo": algo,
        "n_splits": int(n_splits),
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "roc_mean": roc_mean,
        "roc_std": roc_std,
        "prec_mean": prec_mean,
        "prec_std": prec_std,
        "rec_mean": rec_mean,
        "rec_std": rec_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "fold_acc": [float(x) for x in res["test_acc"]],
        "fold_roc": [float(x) for x in res["test_roc"]],
        "fold_prec": [float(x) for x in res["test_prec"]],
        "fold_rec": [float(x) for x in res["test_rec"]],
        "fold_f1": [float(x) for x in res["test_f1"]],
        "seed": int(seed),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if verbose:
        _print_cv_summary(metrics)

    return metrics


def _print_cv_summary(m: dict) -> None:
    n = m["n_splits"]
    print(f"[{n}-FOLD CV] accuracy     = {m['acc_mean']:.3f} ± {m['acc_std']:.3f}")
    print(f"[{n}-FOLD CV] ROC-AUC      = {m['roc_mean']:.3f} ± {m['roc_std']:.3f}")
    print(f"[{n}-FOLD CV] precision(1) = {m['prec_mean']:.3f} ± {m['prec_std']:.3f}")
    print(f"[{n}-FOLD CV] recall(1)    = {m['rec_mean']:.3f} ± {m['rec_std']:.3f}")
    print(f"[{n}-FOLD CV] F1(1)        = {m['f1_mean']:.3f} ± {m['f1_std']:.3f}")


def persist_metrics(metrics: dict, tag: str) -> Path:
    out = OUT_METRICS / f"{metrics['mode']}_{tag}_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {out}")
    return out


def refit_full_and_submit(tag: str, *, algo: str = "logreg") -> Path:
    """Fit on all training data using the chosen algo and write a timestamped submission."""
    df_train = load_train()
    y = df_train["Survived"].astype(int)
    X = df_train.drop(columns=["Survived"])

    pipe = build_model(algo=algo)
    pipe.fit(X, y)

    df_test = load_test()
    y_pred = pipe.predict(df_test).astype(int)

    OUT_SUB.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    sub_path = OUT_SUB / f"submission_{tag}_{ts}.csv"
    pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": y_pred}).to_csv(
        sub_path, index=False
    )
    print(f"Wrote submission -> {sub_path}")
    return sub_path


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate Titanic model (holdout or CV) and optionally write a submission."
    )
    ap.add_argument(
        "--mode", choices=["cv", "holdout"], default="cv", help="Evaluation mode."
    )
    ap.add_argument(
        "--cv", type=int, default=5, help="Number of CV folds (if mode=cv)."
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    ap.add_argument(
        "--algo",
        choices=["logreg", "xgb"],
        default="logreg",
        help="Which model to build for evaluation and submission.",
    )
    ap.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="(holdout only) Path to a prefit joblib Pipeline to evaluate (e.g., XGB with early stopping).",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Short label saved into filenames. Defaults to '<algo>_baseline' if not provided.",
    )
    ap.add_argument(
        "--submit",
        action="store_true",
        help="After eval, refit on full train (with the same --algo) and write a submission.",
    )
    args = ap.parse_args()

    tag = args.tag or f"{args.algo}_baseline"

    if args.mode == "cv":
        if args.model_path:
            raise SystemExit(
                "--model-path is not compatible with --mode cv (cross-validate refits the model)."
            )
        metrics = evaluate_cv(n_splits=args.cv, seed=args.seed, algo=args.algo)
    else:
        model_path = Path(args.model_path) if args.model_path else None
        metrics = evaluate_holdout(
            seed=args.seed, algo=args.algo, model_path=model_path
        )

    persist_metrics(metrics, tag=tag)

    if args.submit:
        refit_full_and_submit(tag=tag, algo=args.algo)


if __name__ == "__main__":
    main()
