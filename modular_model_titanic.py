"""Reusable model factory for the Titanic project.

Provides helpers to build preprocessing and modeling pipelines in a modular
way. Intended to be imported by evaluation and training scripts.
"""

from typing import Any, Dict, Literal, Optional

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLS = ["Age", "Fare", "SibSp", "Parch"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]


def build_preprocessor(scale_numeric: bool, ohe_dense: bool) -> ColumnTransformer:
    """Build a ColumnTransformer that preprocesses numeric and categorical columns.

    Args:
        scale_numeric: Whether to include a StandardScaler for numeric columns.
        ohe_dense: Whether to output a dense array from OneHotEncoder (useful for
            linear models that expect dense inputs).

    Returns:
        A configured ColumnTransformer.
    """
    # numeric pipeline: impute, optionally scale
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler(with_mean=True)))
    num_pipe = Pipeline(steps=num_steps)

    # categorical pipeline: impute, one-hot encode
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not ohe_dense),
            ),
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


def build_estimator(
    kind: Literal["logreg", "gb", "rf"] = "logreg", **kwargs: Any
) -> BaseEstimator:
    """Create an estimator by kind.

    Args:
        kind: One of 'logreg' (LogisticRegression), 'gb' (GradientBoostingClassifier),
            or 'rf' (RandomForestClassifier).
        **kwargs: Passed through to the estimator constructor.

    Returns:
        An sklearn BaseEstimator instance.
    """
    if kind == "logreg":
        return LogisticRegression(max_iter=1000, solver="lbfgs", **kwargs)
    if kind == "gb":
        return GradientBoostingClassifier(**kwargs)
    if kind == "rf":
        return RandomForestClassifier(**kwargs)
    raise ValueError(f"Unknown estimator kind: {kind}")


def build_model(
    kind: Literal["logreg", "gb", "rf"] = "logreg",
    est_kwargs: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Build a ML pipeline with preprocessing and an estimator.

    Behavior:
      - Linear models (logreg) get numeric scaling and dense OHE output.
      - Tree models (gb, rf) do not scale numeric features and use sparse OHE output.

    Args:
        kind: Estimator kind.
        est_kwargs: Optional kwargs forwarded to the estimator constructor.

    Returns:
        A sklearn Pipeline object with steps [('pre', ColumnTransformer), ('clf', estimator)].
    """
    est_kwargs = est_kwargs or {}
    is_linear = kind == "logreg"

    # scale only for linear models; dense OHE for linear models as well
    pre = build_preprocessor(scale_numeric=is_linear, ohe_dense=is_linear)
    clf = build_estimator(kind, **est_kwargs)

    return Pipeline(steps=[("pre", pre), ("clf", clf)])
