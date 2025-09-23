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


def find_optimal_estimators(
    seed: int = 42, test_ranges: list = None, verbose: bool = True
) -> dict:
    """
    Test different numbers of estimators to find the optimal value for XGBoost.
    Tests multiple values and returns the best performing one.
    """
    if test_ranges is None:
        test_ranges = [50, 100, 200, 300, 500, 800, 1000]

    print(f"Testing estimator counts: {test_ranges}")

    best_score = -1
    best_estimators = 500
    results = []

    for n_est in test_ranges:
        print(f"\nTesting {n_est} estimators...")

        # Run holdout evaluation
        metrics = evaluate_holdout(seed=seed, algo="xgb", n_estimators=n_est)

        score = metrics["auc"]  # Use AUC as the primary metric
        results.append(
            {
                "n_estimators": n_est,
                "auc": score,
                "accuracy": metrics["acc"],
                "f1": metrics["f1_pos"],
            }
        )

        if score > best_score:
            best_score = score
            best_estimators = n_est

        if verbose:
            print(
                f"  -> AUC: {score:.4f}, Accuracy: {metrics['acc']:.4f}, F1: {metrics['f1_pos']:.4f}"
            )

    final_results = {
        "mode": "optimal_search",
        "optimal_estimators": best_estimators,
        "best_auc": best_score,
        "test_results": results,
        "tested_ranges": test_ranges,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if verbose:
        print(
            f"\n🎯 Optimal number of estimators: {best_estimators} (AUC: {best_score:.4f})"
        )

    return final_results


def compare_features(
    seed: int = 42, algo: str = "xgb", n_estimators: int = 50, verbose: bool = True
) -> dict:
    """
    Compare model performance with and without engineered features.
    Tests the boy_master/boy_nonmaster features vs baseline features only.
    """
    print("🔍 Comparing feature engineering impact...\n")

    # Import the model building functions
    from titanic_lab.model_titanic import CAT_COLS
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    # Load data
    df = load_train()
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])

    # Test 1: With engineered features (current setup)
    print("Testing WITH engineered features (boy_master, boy_nonmaster)...")
    metrics_with = evaluate_cv(
        n_splits=5, seed=seed, algo=algo, n_estimators=n_estimators, verbose=False
    )

    # Test 2: Without engineered features (baseline)
    print("Testing WITHOUT engineered features (baseline)...")

    # Temporarily modify the column lists for baseline
    baseline_numeric = ["Age", "Fare", "SibSp", "Parch"]  # Remove boy features
    baseline_cat = CAT_COLS  # Keep categorical the same

    # Create a baseline preprocessor function
    def build_baseline_preprocessor():
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        num_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )

        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            [
                ("num", num_pipe, baseline_numeric),
                ("cat", cat_pipe, baseline_cat),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    # Build baseline model
    def build_baseline_model():
        pre = build_baseline_preprocessor()
        if algo.lower() == "xgb":
            clf = XGBClassifier(
                objective="binary:logistic",
                tree_method="hist",
                n_estimators=n_estimators,
                learning_rate=0.05,
                max_depth=4,
                random_state=seed,
            )
        else:
            clf = LogisticRegression(max_iter=1000, random_state=seed)
        return Pipeline([("pre", pre), ("clf", clf)])

    # Evaluate baseline model with CV
    from sklearn.model_selection import StratifiedKFold, cross_validate

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    baseline_pipe = build_baseline_model()

    res = cross_validate(
        baseline_pipe,
        X,
        y,
        cv=cv,
        scoring={"acc": "accuracy", "roc": "roc_auc", "f1": "f1"},
        return_train_score=False,
        n_jobs=-1,
    )

    metrics_without = {
        "acc_mean": float(res["test_acc"].mean()),
        "acc_std": float(res["test_acc"].std()),
        "roc_mean": float(res["test_roc"].mean()),
        "roc_std": float(res["test_roc"].std()),
        "f1_mean": float(res["test_f1"].mean()),
        "f1_std": float(res["test_f1"].std()),
    }

    # Calculate improvements
    auc_improvement = metrics_with["roc_mean"] - metrics_without["roc_mean"]
    acc_improvement = metrics_with["acc_mean"] - metrics_without["acc_mean"]
    f1_improvement = metrics_with["f1_mean"] - metrics_without["f1_mean"]

    results = {
        "mode": "feature_comparison",
        "algo": algo,
        "n_estimators": n_estimators,
        "with_features": {
            "auc": metrics_with["roc_mean"],
            "auc_std": metrics_with["roc_std"],
            "accuracy": metrics_with["acc_mean"],
            "acc_std": metrics_with["acc_std"],
            "f1": metrics_with["f1_mean"],
            "f1_std": metrics_with["f1_std"],
        },
        "without_features": {
            "auc": metrics_without["roc_mean"],
            "auc_std": metrics_without["roc_std"],
            "accuracy": metrics_without["acc_mean"],
            "acc_std": metrics_without["acc_std"],
            "f1": metrics_without["f1_mean"],
            "f1_std": metrics_without["f1_std"],
        },
        "improvements": {
            "auc": auc_improvement,
            "accuracy": acc_improvement,
            "f1": f1_improvement,
        },
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if verbose:
        print("\n📊 RESULTS SUMMARY:")
        print(
            f"{'Metric':<12} {'With Features':<15} {'Without Features':<17} {'Improvement':<12}"
        )
        print("=" * 60)
        print(
            f"{'ROC-AUC':<12} {metrics_with['roc_mean']:.4f}±{metrics_with['roc_std']:.3f}    {metrics_without['roc_mean']:.4f}±{metrics_without['roc_std']:.3f}      {auc_improvement:+.4f}"
        )
        print(
            f"{'Accuracy':<12} {metrics_with['acc_mean']:.4f}±{metrics_with['acc_std']:.3f}    {metrics_without['acc_mean']:.4f}±{metrics_without['acc_std']:.3f}      {acc_improvement:+.4f}"
        )
        print(
            f"{'F1-Score':<12} {metrics_with['f1_mean']:.4f}±{metrics_with['f1_std']:.3f}    {metrics_without['f1_mean']:.4f}±{metrics_without['f1_std']:.3f}      {f1_improvement:+.4f}"
        )

        print("\n🎯 CONCLUSION:")
        if auc_improvement > 0.01:  # 1% improvement threshold
            print("✅ Engineered features provide SIGNIFICANT improvement!")
        elif auc_improvement > 0.005:  # 0.5% improvement threshold
            print("✅ Engineered features provide modest improvement.")
        elif auc_improvement > 0:
            print("⚠️  Engineered features provide minimal improvement.")
        else:
            print("❌ Engineered features do NOT improve performance.")

    return results


def evaluate_holdout(
    seed: int = 42,
    *,
    algo: str = "logreg",
    model_path: Path | None = None,
    **model_kwargs,
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
        pipe = build_model(algo=algo, **model_kwargs)
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
        "model_kwargs": model_kwargs,
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
    n_splits: int = 5,
    seed: int = 42,
    verbose: bool = False,
    *,
    algo: str = "logreg",
    **model_kwargs,
) -> dict:
    """Stratified K-fold CV with accuracy (Kaggle metric) + ROC-AUC + precision/recall/F1.

    Always builds a fresh estimator per fold via build_model(algo) so refitting is correct.
    """
    df = load_train()
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipe = build_model(algo=algo, **model_kwargs)

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
        "model_kwargs": model_kwargs,
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


def refit_full_and_submit(tag: str, *, algo: str = "logreg", **model_kwargs) -> Path:
    """Fit on all training data using the chosen algo and write a timestamped submission."""
    df_train = load_train()
    y = df_train["Survived"].astype(int)
    X = df_train.drop(columns=["Survived"])

    pipe = build_model(algo=algo, **model_kwargs)
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
        description="Evaluate Titanic model (holdout or CV) and optionally write a submission.",
        epilog="""
Examples:
  # Find optimal number of estimators using early stopping
  python %(prog)s --algo xgb --find-optimal-estimators

  # Use optimal estimators and generate submission
  python %(prog)s --algo xgb --find-optimal-estimators --submit

  # Basic XGBoost with 100 estimators, cross-validation
  python %(prog)s --algo xgb --n-estimators 100 --mode cv

  # XGBoost with 500 estimators, holdout evaluation and submission
  python %(prog)s --algo xgb --n-estimators 500 --mode holdout --submit

  # Logistic regression (n-estimators is ignored)
  python %(prog)s --algo logreg --mode cv
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--n-estimators",
        type=int,
        default=500,
        help="Number of estimators (trees) for XGBoost. Ignored for logistic regression. Default is 500.",
    )
    ap.add_argument(
        "--find-optimal-estimators",
        action="store_true",
        help="Find optimal number of estimators using early stopping (XGBoost only). Overrides other evaluation modes.",
    )
    ap.add_argument(
        "--compare-features",
        action="store_true",
        help="Compare model performance with and without engineered features. Shows feature impact.",
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

    # Handle optimal estimators finding mode
    if getattr(args, "find_optimal_estimators", False):
        if args.algo != "xgb":
            raise SystemExit(
                "--find-optimal-estimators is only supported for XGBoost (--algo xgb)"
            )

        results = find_optimal_estimators(seed=args.seed)

        # Save the results
        persist_metrics(results, tag="optimal_estimators")

        print(
            f"\nRecommendation: Use --n-estimators {results['optimal_estimators']} for best performance"
        )

        if args.submit:
            # Use the optimal number for submission
            optimal_kwargs = {"n_estimators": results["optimal_estimators"]}
            refit_full_and_submit(
                tag=f"optimal_{results['optimal_estimators']}est",
                algo=args.algo,
                **optimal_kwargs,
            )
        return

    # Handle feature comparison mode
    if getattr(args, "compare_features", False):
        n_est = getattr(args, "n_estimators", 50)
        results = compare_features(seed=args.seed, algo=args.algo, n_estimators=n_est)

        # Save the results
        persist_metrics(results, tag="feature_comparison")
        return

    # Prepare model kwargs for regular evaluation
    model_kwargs = {}
    if args.algo == "xgb":
        model_kwargs["n_estimators"] = getattr(args, "n_estimators")

    if args.mode == "cv":
        if args.model_path:
            raise SystemExit(
                "--model-path is not compatible with --mode cv (cross-validate refits the model)."
            )
        metrics = evaluate_cv(
            n_splits=args.cv, seed=args.seed, algo=args.algo, **model_kwargs
        )
    else:
        model_path = Path(args.model_path) if args.model_path else None
        metrics = evaluate_holdout(
            seed=args.seed, algo=args.algo, model_path=model_path, **model_kwargs
        )

    persist_metrics(metrics, tag=tag)

    if args.submit:
        refit_full_and_submit(tag=tag, algo=args.algo, **model_kwargs)


if __name__ == "__main__":
    main()
