# %% imports
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from titanic_lab.paths import ROOT, TRAIN_CSV

OUT_MODELS = Path(ROOT) / "outputs" / "models"
OUT_MODELS.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["Age", "Fare", "SibSp", "Parch"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]


def load_train(path: Path | str = TRAIN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Pclass"] = df["Pclass"].astype("category")
    df["Sex"] = df["Sex"].astype("category")
    df["Embarked"] = df["Embarked"].astype("category")
    return df


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
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
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


def split_X_y(df: pd.DataFrame):
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived", "PassengerId"])  # keep PassengerId out of features
    return X, y


def fit_preprocessor_on_train(df_train: pd.DataFrame) -> Pipeline:
    pre = build_preprocessor()
    # Wrap in Pipeline so later you can just append a classifier
    pipe = Pipeline(steps=[("pre", pre)])
    X, _ = split_X_y(df_train)
    pipe.fit(X)  # fit imputers/encoders/scalers on TRAIN ONLY
    return pipe


def transform_any(df: pd.DataFrame, pipe: Pipeline) -> np.ndarray:
    X = df.drop(columns=["Survived", "PassengerId"], errors="ignore")
    X_trans = pipe.transform(X)
    return X_trans


def save_preprocessor(
    pipe: Pipeline, path: Path = OUT_MODELS / "preprocess_basic.joblib"
) -> Path:
    joblib.dump(pipe, path)
    return path


if __name__ == "__main__":
    # Minimal CLI behavior: fit on train and serialize the preprocessor
    df_train = load_train(TRAIN_CSV)
    pipe = fit_preprocessor_on_train(df_train)
    p = save_preprocessor(pipe)
    print(f"Saved basic preprocessor -> {p}")

    # Quick sanity: transformed shape
    Xt = transform_any(df_train, pipe)
    print("Transformed train shape:", Xt.shape)

# %%
