from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401
from sklearn.preprocessing import OneHotEncoder  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401


categorical


# zscore scaling
def build_pre_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        num_pipeline_named_steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("standardize", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    categorical_pipeline = Pipeline(
        cat_pipeline_named_steps=[("impute", OneHotEncoder())]
    )
