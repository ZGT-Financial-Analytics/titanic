# imports
from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
import numpy as np  # noqa: F401
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# paths
from titanic_lab.paths import ROOT, TRAIN_CSV

try:
    from titanic_lab.paths import TEST_CSV  # type: ignore
except Exception:
    TEST_CSV = Path(ROOT) / "data" / "test.csv"
# storage for outputs
OUT_MODELS = Path(ROOT) / "outputs" / "models"
OUT_SUB = Path(ROOT) / "outputs" / "submissions"
OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_SUB.mkdir(parents=True, exist_ok=True)


# establishing numerical and categorical columns to route through respective pipelines in ColumnTransformer
NUMERIC_COLS = ["Age", "Fare", "SibSp", "Parch"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]


# TRAIN data loading function, with type conversions for categorical columns TRAIN
def load_train(path: Path | str = TRAIN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Pclass"] = df["Pclass"].astype("category")
    df["Sex"] = df["Sex"].astype("category")
    df["Embarked"] = df["Embarked"].astype("category")
    return df


# TEST data loading function, with type conversions for categorical columns TEST
def load_test(path: Path | str = TEST_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Pclass"] = df["Pclass"].astype("category")
    df["Sex"] = df["Sex"].astype("category")
    df["Embarked"] = df["Embarked"].astype("category")
    return df


# building preprocessing pipeline, num+cat pipelines -> ColumnTransformer
def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True)),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


# ---------------- Model Selection ----------------
def build_model_logreg() -> Pipeline:
    pre = build_preprocessor()
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight=None,
        n_jobs=None,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def build_model_xgb(
    *,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    random_state: int = 42,
) -> Pipeline:
    pre = build_preprocessor()
    xgb = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("pre", pre), ("clf", xgb)])


def build_model(algo: str = "logreg") -> Pipeline:
    """Factory: 'logreg' (default) or 'xgb'."""
    algo = algo.lower()
    if algo == "xgb":
        return build_model_xgb()
    return build_model_logreg()


# ---- Optional: early-stopping flavor (still returns a Pipeline) ----
def build_model_xgb_earlystop(
    *,
    val_size: float = 0.2,
    random_state: int = 42,
    early_stopping_rounds: int = 50,
    # XGB knobs (sensible defaults)
    n_estimators: int = 2000,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
) -> Pipeline:
    """
    Fits preprocessor + XGB with early stopping on a holdout split,
    then returns a single Pipeline frozen to best_iteration.
    """
    df = load_train()
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])

    # Fit preprocessor first to expose eval_set cleanly
    pre = build_preprocessor()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )
    X_tr_t = pre.fit_transform(X_tr, y_tr)
    X_val_t = pre.transform(X_val)

    booster = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        n_jobs=-1,
        random_state=random_state,
    )
    booster.fit(
        X_tr_t,
        y_tr,
        eval_set=[(X_val_t, y_val)],
        eval_metric="auc",
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    # Freeze to best # of trees
    best_ntree = (
        booster.best_iteration + 1
        if booster.best_iteration is not None
        else booster.n_estimators
    )
    booster.set_params(n_estimators=best_ntree)

    # Refit booster on ALL data transformed by pre
    X_all_t = pre.transform(X)
    booster.fit(X_all_t, y)

    # Return a unified artifact
    return Pipeline([("pre", pre), ("clf", booster)])


# -------- train & save --------
def train_and_save(model_path: Path = OUT_MODELS / "model_titanic.joblib") -> Path:
    df_train = load_train()
    y = df_train["Survived"].astype(int)
    # Belt-and-suspenders: drop target (ColumnTransformer would drop anyway)
    X = df_train.drop(columns=["Survived"])

    model = build_model_logreg()
    model.fit(X, y)  # <- fit (learn imputers, scalers, OHE vocab + classifier weights)

    joblib.dump(model, model_path)
    print(f"Saved model -> {model_path}")
    return model_path


if __name__ == "__main__":
    train_and_save()
