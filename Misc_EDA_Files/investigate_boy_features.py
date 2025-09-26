#!/usr/bin/env python3
"""
Investigate why boy_master/boy_nonmaster features don't help despite strong t-score.
"""

import pandas as pd
from scipy import stats

# Load the data
from src.titanic_lab.model_titanic import load_train


def analyze_boy_features():
    """Analyze the boy features to understand why they don't help the model."""

    df = load_train()

    # Add the boy features back for analysis
    if "Title" not in df.columns:
        df["Title"] = (
            df["Name"].str.extract(r",\s*([^\.]+)\.").iloc[:, 0].str.strip().str.lower()
        )

    is_boy = df["Sex"].str.lower().eq("male") & df["Age"].lt(18)
    df["boy_master"] = (is_boy & df["Title"].eq("master")).astype(int)
    df["boy_nonmaster"] = (is_boy & ~df["Title"].eq("master")).astype(int)

    print("🔍 BOY FEATURES ANALYSIS")
    print("=" * 50)

    # 1. Sample sizes
    print("\n1. SAMPLE SIZES:")
    print(f"Total samples: {len(df)}")
    print(f"Boys under 18: {is_boy.sum()}")
    print(f"Boy masters: {df['boy_master'].sum()}")
    print(f"Boy non-masters: {df['boy_nonmaster'].sum()}")
    print(f"Boy percentage of dataset: {is_boy.sum() / len(df) * 100:.1f}%")

    # 2. Survival rates
    print("\n2. SURVIVAL RATES:")
    boy_master_survival = df[df["boy_master"] == 1]["Survived"].mean()
    boy_nonmaster_survival = df[df["boy_nonmaster"] == 1]["Survived"].mean()
    overall_survival = df["Survived"].mean()

    print(f"Overall survival rate: {overall_survival:.3f}")
    print(f"Boy master survival rate: {boy_master_survival:.3f}")
    print(f"Boy non-master survival rate: {boy_nonmaster_survival:.3f}")
    print(f"Difference: {boy_master_survival - boy_nonmaster_survival:.3f}")

    # 3. T-test
    print("\n3. STATISTICAL TEST:")
    boy_master_survived = df[df["boy_master"] == 1]["Survived"]
    boy_nonmaster_survived = df[df["boy_nonmaster"] == 1]["Survived"]

    if len(boy_master_survived) > 0 and len(boy_nonmaster_survived) > 0:
        t_stat, p_val = stats.ttest_ind(boy_master_survived, boy_nonmaster_survived)
        print(f"T-statistic: {t_stat:.3f}")
        print(f"P-value: {p_val:.6f}")
    else:
        print("Cannot perform t-test: insufficient samples")

    # 4. Check for redundancy with existing features
    print("\n4. FEATURE REDUNDANCY CHECK:")

    # Check correlation with Age and Sex
    df_numeric = df[["Age", "boy_master", "boy_nonmaster"]].fillna(df["Age"].median())
    df_numeric["is_male"] = (df["Sex"] == "male").astype(int)

    corr_matrix = df_numeric.corr()
    print("Correlation matrix:")
    print(corr_matrix)

    # 5. Check if XGBoost can learn Age*Sex interaction
    print("\n5. AGE-SEX INTERACTION ANALYSIS:")

    # Create age groups
    df["age_group"] = pd.cut(
        df["Age"].fillna(df["Age"].median()),
        bins=[0, 16, 30, 50, 100],
        labels=["Child", "Young", "Middle", "Old"],
    )

    # Survival by age group and sex
    survival_by_age_sex = df.groupby(["age_group", "Sex"])["Survived"].agg(
        ["count", "mean"]
    )
    print("\nSurvival by Age Group and Sex:")
    print(survival_by_age_sex)

    # 6. Feature importance simulation
    print("\n6. FEATURE IMPORTANCE SIMULATION:")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    # Prepare data
    X = df[
        [
            "Age",
            "Sex",
            "Pclass",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "boy_master",
            "boy_nonmaster",
        ]
    ].copy()
    y = df["Survived"]

    # Fill missing values
    X["Age"].fillna(X["Age"].median(), inplace=True)
    X["Fare"].fillna(X["Fare"].median(), inplace=True)
    X["Embarked"].fillna(X["Embarked"].mode()[0], inplace=True)

    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    X["Sex"] = le_sex.fit_transform(X["Sex"])
    X["Embarked"] = le_embarked.fit_transform(X["Embarked"])

    # Fit random forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Feature importance
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nFeature Importance (Random Forest):")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']:<15}: {row['importance']:.4f}")

    return df


if __name__ == "__main__":
    df = analyze_boy_features()
