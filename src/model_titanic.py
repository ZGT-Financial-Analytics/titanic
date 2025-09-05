# imports
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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


# data loading function, with type conversions for categorical columns
def load_train(path: Path | str = TRAIN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Pclass"] = df["Pclass"].astype("category")
    df["Sex"] = df["Sex"].astype("category")
    df["Embarked"] = df["Embarked"].astype("category")
    return df


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


# model builder with two stage pipeline; preprocessing and fit/predict processed data
def build_model() -> Pipeline:
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

    model = build_model()
    model.fit(X, y)  # <- fit (learn imputers, scalers, OHE vocab + classifier weights)

    joblib.dump(model, model_path)
    print(f"Saved model -> {model_path}")
    return model_path


# -------- predict test & save submission --------
def predict_submission(
    model_path: Path, sub_path: Path = OUT_SUB / "submission.csv"
) -> Path:
    model: Pipeline = joblib.load(model_path)
    df_test = load_test()

    y_pred = model.predict(df_test)  # <- predict (internally runs preprocess + clf)
    sub = pd.DataFrame(
        {
            "PassengerId": df_test["PassengerId"],
            "Survived": y_pred.astype(int),
        }
    )
    sub.to_csv(sub_path, index=False)
    print(f"Wrote submission -> {sub_path}")
    return sub_path


if __name__ == "__main__":
    model_path = train_and_save()
    predict_submission(model_path)
