"""Stacked agent: alternative ensemble model for Titanic survival."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from titanic_lab.paths import ROOT, TEST_CSV, TRAIN_CSV


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering independent of any notebook logic."""

    engineered = df.copy()

    # Titles from passenger names
    engineered["Title"] = (
        engineered["Name"]
        .str.extract(r",\s*([^\.]+)\.")
        .iloc[:, 0]
        .fillna("Unknown")
        .str.strip()
    )

    # Family based metrics
    engineered["FamilySize"] = (
        engineered["SibSp"].fillna(0) + engineered["Parch"].fillna(0) + 1
    )
    engineered["IsAlone"] = (engineered["FamilySize"] == 1).astype(int)
    engineered["FarePerPerson"] = engineered["Fare"].fillna(0) / engineered[
        "FamilySize"
    ].replace(0, 1)

    # Cabin deck and ticket prefixes capture travel context
    engineered["CabinDeck"] = engineered["Cabin"].fillna("U").str[0]
    engineered["TicketPrefix"] = (
        engineered["Ticket"].fillna("NA").str.replace(r"[^A-Za-z]", "", regex=True)
    )
    engineered.loc[engineered["TicketPrefix"] == "", "TicketPrefix"] = "NONE"

    # Ticket frequency to capture group travel
    engineered["TicketFreq"] = (
        engineered.groupby("Ticket")["Ticket"].transform("count").fillna(1)
    )

    # Interactions capturing socio-economic factors
    engineered["FareTimesClass"] = engineered["Fare"].fillna(0) * engineered[
        "Pclass"
    ].fillna(0)
    engineered["AgeClass"] = engineered["Age"].fillna(
        engineered["Age"].median()
    ) * engineered["Pclass"].fillna(0)

    return engineered


def _build_pipeline() -> Pipeline:
    """Construct preprocessing + stacking estimator pipeline."""

    feature_builder = FunctionTransformer(_engineer_features, validate=False)

    numeric_features = [
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",
        "FarePerPerson",
        "TicketFreq",
        "FareTimesClass",
        "AgeClass",
    ]

    categorical_features = [
        "Pclass",
        "Sex",
        "Embarked",
        "CabinDeck",
        "TicketPrefix",
        "Title",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    base_estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(
            C=0.5,
            max_iter=1000,
            penalty="l2",
            solver="lbfgs",
        ),
        stack_method="predict_proba",
        passthrough=False,
        cv=5,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("features", feature_builder),
            ("preprocess", preprocessor),
            ("stack", stacker),
        ]
    )

    return pipeline


def main() -> None:
    """Train CV, fit full data, and export submission."""

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    y = train["Survived"].astype(int).values
    X = train.drop(columns=["Survived"])

    pipeline = _build_pipeline()

    print("⚙️  Evaluating stacked agent with 5-fold CV...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"   CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"   Individual folds: {[f'{score:.4f}' for score in cv_scores]}")

    print("🚀 Fitting stacked agent on full training data...")
    pipeline.fit(X, y)

    print("📦 Generating predictions for Kaggle test set...")
    test_predictions = pipeline.predict(test)

    survival_rate = np.mean(test_predictions)
    print(
        f"   Predicted survivors: {test_predictions.sum()} / {len(test_predictions)} "
        f"({survival_rate:.1%})"
    )

    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"],
            "Survived": test_predictions.astype(int),
        }
    )

    output_path = ROOT / "outputs" / "submissions" / "submission_stacked_agent.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"💾 Saved submission to: {output_path}")
    print("✅ Stacked agent ready for Kaggle upload.")


if __name__ == "__main__":
    main()
