#!/usr/bin/env python3
"""
Simple demonstration of how XGBoost learns Age-Sex interactions.
"""

import pandas as pd
from src.titanic_lab.model_titanic import load_train, build_model


def simple_interaction_demo():
    """Show how XGBoost learns Age-Sex interactions without explicit features."""

    print("🌳 HOW XGBOOST LEARNS AGE-SEX INTERACTIONS")
    print("=" * 55)

    # Load data
    df = load_train()
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])

    # Train simple XGBoost model
    model = build_model(algo="xgb", n_estimators=15)
    model.fit(X, y)

    print("\n🎯 KEY INSIGHT: Tree-based models create interactions through SPLITS")
    print("\nA typical XGBoost tree might look like:")
    print("Root")
    print("├─ Sex_female == 1 (IS FEMALE)")
    print("│   ├─ Age <= 16 → HIGH survival (child female)")
    print("│   └─ Age > 16 → MEDIUM survival (adult female)")
    print("└─ Sex_female == 0 (IS MALE)")
    print("    ├─ Age <= 16 → MEDIUM survival (child male)")
    print("    └─ Age > 16 → LOW survival (adult male)")

    # Test predictions on specific cases
    print("\n📊 MODEL PREDICTIONS:")
    print("-" * 40)

    # Create test cases (using median values for other features)
    test_cases = []

    # Child female
    child_f = X.iloc[0:1].copy()
    child_f["Age"] = 10
    child_f["Sex"] = "female"
    child_f["Pclass"] = 3
    pred_cf = model.predict_proba(child_f)[0, 1]

    # Child male
    child_m = X.iloc[0:1].copy()
    child_m["Age"] = 10
    child_m["Sex"] = "male"
    child_m["Pclass"] = 3
    pred_cm = model.predict_proba(child_m)[0, 1]

    # Adult female
    adult_f = X.iloc[0:1].copy()
    adult_f["Age"] = 30
    adult_f["Sex"] = "female"
    adult_f["Pclass"] = 3
    pred_af = model.predict_proba(adult_f)[0, 1]

    # Adult male
    adult_m = X.iloc[0:1].copy()
    adult_m["Age"] = 30
    adult_m["Sex"] = "male"
    adult_m["Pclass"] = 3
    pred_am = model.predict_proba(adult_m)[0, 1]

    print(f"Child Female (10 yrs): {pred_cf:.1%} survival")
    print(f"Child Male   (10 yrs): {pred_cm:.1%} survival")
    print(f"Adult Female (30 yrs): {pred_af:.1%} survival")
    print(f"Adult Male   (30 yrs): {pred_am:.1%} survival")

    # Show actual data patterns
    print("\n📈 ACTUAL DATA PATTERNS:")
    print("-" * 40)

    # Calculate actual survival rates
    children = df[df["Age"] <= 16]
    adults = df[df["Age"] > 16]

    child_f_rate = children[children["Sex"] == "female"]["Survived"].mean()
    child_m_rate = children[children["Sex"] == "male"]["Survived"].mean()
    adult_f_rate = adults[adults["Sex"] == "female"]["Survived"].mean()
    adult_m_rate = adults[adults["Sex"] == "male"]["Survived"].mean()

    print(
        f"Child Female actual: {child_f_rate:.1%} (n={len(children[children['Sex'] == 'female'])})"
    )
    print(
        f"Child Male   actual: {child_m_rate:.1%} (n={len(children[children['Sex'] == 'male'])})"
    )
    print(
        f"Adult Female actual: {adult_f_rate:.1%} (n={len(adults[adults['Sex'] == 'female'])})"
    )
    print(
        f"Adult Male   actual: {adult_m_rate:.1%} (n={len(adults[adults['Sex'] == 'male'])})"
    )

    print("\n💡 CONCLUSION:")
    print("XGBoost learned the Age-Sex interaction patterns WITHOUT explicit")
    print("cross-terms because decision trees naturally create conditional splits!")
    print("\nThis is why tree models are so powerful for capturing interactions.")


if __name__ == "__main__":
    simple_interaction_demo()
