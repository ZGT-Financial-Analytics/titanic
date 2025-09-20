# imports
from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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
        solver="lbfgs",  # dense-friendly; thanks to dense OHE above
        class_weight=None,  # flip to "balanced" if you want
        n_jobs=None,
    )
    model = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", clf),
        ]
    )
    return model


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
