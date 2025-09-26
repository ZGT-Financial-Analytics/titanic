#!/usr/bin/env python3
"""
Analyze how XGBoost learns Age-Sex interactions without explicit cross terms.
"""

import pandas as pd
from src.titanic_lab.model_titanic import load_train, build_model


def analyze_xgb_interactions():
    """Show how XGBoost learns Age-Sex interactions implicitly."""

    print("🌳 HOW XGBOOST LEARNS AGE-SEX INTERACTIONS")
    print("=" * 55)

    # Load data and train model
    df = load_train()
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])

    # Train XGBoost model
    model = build_model(algo="xgb", n_estimators=15)
    model.fit(X, y)

    # Get the XGBoost classifier
    xgb_clf = model.named_steps["clf"]

    print(f"\n1. MODEL STRUCTURE:")
    print(f"Number of trees: {xgb_clf.n_estimators}")
    print(f"Max depth: {xgb_clf.max_depth}")

    # Create test cases for different age-sex combinations
    print(f"\n2. PREDICTIONS FOR AGE-SEX COMBINATIONS:")
    print("-" * 50)

    # Get feature names after preprocessing
    X_transformed = model.named_steps["pre"].transform(X)
    feature_names = model.named_steps["pre"].get_feature_names_out()

    print(f"Features used by model: {list(feature_names)}")

    # Create synthetic examples
    test_cases = []

    # Child female (age 10)
    child_female = pd.DataFrame(
        {
            "Age": [10],
            "Fare": [20],
            "SibSp": [1],
            "Parch": [1],
            "Sex": ["female"],
            "Embarked": ["S"],
            "Pclass": [3],
            "Age_Sex_interaction": [0],  # 10 * 0 (female=0)
        }
    )

    # Child male (age 10)
    child_male = pd.DataFrame(
        {
            "Age": [10],
            "Fare": [20],
            "SibSp": [1],
            "Parch": [1],
            "Sex": ["male"],
            "Embarked": ["S"],
            "Pclass": [3],
            "Age_Sex_interaction": [10],  # 10 * 1 (male=1)
        }
    )

    # Adult female (age 30)
    adult_female = pd.DataFrame(
        {
            "Age": [30],
            "Fare": [20],
            "SibSp": [0],
            "Parch": [0],
            "Sex": ["female"],
            "Embarked": ["S"],
            "Pclass": [3],
            "Age_Sex_interaction": [0],  # 30 * 0
        }
    )

    # Adult male (age 30)
    adult_male = pd.DataFrame(
        {
            "Age": [30],
            "Fare": [20],
            "SibSp": [0],
            "Parch": [0],
            "Sex": ["male"],
            "Embarked": ["S"],
            "Pclass": [3],
            "Age_Sex_interaction": [30],  # 30 * 1
        }
    )

    cases = [
        ("Child Female (age 10)", child_female),
        ("Child Male (age 10)", child_male),
        ("Adult Female (age 30)", adult_female),
        ("Adult Male (age 30)", adult_male),
    ]

    for name, case_df in cases:
        pred_proba = model.predict_proba(case_df)[0, 1]  # Probability of survival
        print(f"{name:<25}: {pred_proba:.3f} survival probability")

    # Check actual data survival rates for comparison
    print(f"\n3. ACTUAL DATA SURVIVAL RATES:")
    print("-" * 50)

    # Create age groups
    df_analysis = df.copy()
    df_analysis["age_group"] = pd.cut(
        df_analysis["Age"].fillna(df_analysis["Age"].median()),
        bins=[0, 16, 100],
        labels=["Child", "Adult"],
    )

    survival_rates = df_analysis.groupby(["age_group", "Sex"])["Survived"].agg(
        ["count", "mean"]
    )

    for (age_group, sex), data in survival_rates.iterrows():
        if data["count"] > 5:  # Only show groups with reasonable sample size
            print(
                f"{age_group} {sex:<6} (n={data['count']:2d}): {data['mean']:.3f} survival rate"
            )

    # Feature importance analysis
    print(f"\n4. FEATURE IMPORTANCE IN MODEL:")
    print("-" * 50)

    importance_dict = dict(zip(feature_names, xgb_clf.feature_importances_))

    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    for feature, importance in sorted_features:
        if importance > 0.01:  # Only show important features
            print(f"{feature:<25}: {importance:.4f}")

    print(f"\n5. WHY XGBOOST DOESN'T NEED EXPLICIT INTERACTIONS:")
    print("-" * 55)
    print("• Trees naturally create conditional splits")
    print("• Each tree can split on Age, then Sex (or vice versa)")
    print("• Multiple trees learn different Age-Sex combinations")
    print("• Ensemble averages capture the interaction effect")
    print("• This is why tree models often outperform linear models")

    return model


if __name__ == "__main__":
    model = analyze_xgb_interactions()
