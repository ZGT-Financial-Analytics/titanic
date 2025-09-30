# complete_wcg_xgboost_model.py - Complete Python translation of the WCG+XGBoost model
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor

from titanic_lab.paths import TRAIN_CSV, TEST_CSV, ROOT


class CompleteWCGXGBoostModel(BaseEstimator, ClassifierMixin):
    """
    Complete translation of the WCG+XGBoost model that achieves 84.7% accuracy.

    This model combines:
    1. WCG (Women and Children Groups) rules based on family survival patterns
    2. XGBoost for adult males with engineered features
    3. XGBoost for solo females with different features and threshold
    """

    def __init__(self):
        # XGBoost parameters exactly as in R
        self.xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "error",
            "max_depth": 5,
            "eta": 0.1,
            "gamma": 0.1,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "verbosity": 0,
        }
        self.n_rounds = 500

        # Models for different groups
        self.male_model = None
        self.female_model = None

        # Group survival information learned from training
        self.group_survival = {}

    def _impute_missing_values(self, df):
        """Impute missing Age and Fare using decision trees (like R's rpart)"""
        df_imp = df.copy()

        # Simple encoding for titles (needed for imputation)
        title_map = {"man": 0, "woman": 1, "boy": 2}
        df_imp["Title_encoded"] = df_imp["Title"].map(title_map).fillna(0)

        # Impute Age
        age_missing = df_imp["Age"].isna()
        if age_missing.any():
            age_features = ["Title_encoded", "Pclass", "SibSp", "Parch"]
            age_train = df_imp[~age_missing]
            age_test = df_imp[age_missing]

            dt_age = DecisionTreeRegressor(random_state=42)
            dt_age.fit(age_train[age_features], age_train["Age"])
            df_imp.loc[age_missing, "Age"] = dt_age.predict(age_test[age_features])

        # Impute Fare
        fare_missing = df_imp["Fare"].isna()
        if fare_missing.any():
            fare_features = ["Title_encoded", "Pclass", "Age"]
            # Add Sex encoding
            df_imp["Sex_encoded"] = (df_imp["Sex"] == "male").astype(int)
            fare_features.append("Sex_encoded")

            fare_train = df_imp[~fare_missing]
            fare_test = df_imp[fare_missing]

            dt_fare = DecisionTreeRegressor(random_state=42)
            dt_fare.fit(fare_train[fare_features], fare_train["Fare"])
            df_imp.loc[fare_missing, "Fare"] = dt_fare.predict(fare_test[fare_features])

        return df_imp

    def _engineer_features(self, df):
        """Create all engineered features exactly as in R"""
        df_eng = df.copy()

        # Title engineering exactly as in R
        df_eng["Title"] = "man"
        df_eng.loc[df_eng["Name"].str.contains("Master", na=False), "Title"] = "boy"
        df_eng.loc[df_eng["Sex"] == "female", "Title"] = "woman"

        # Impute missing values
        df_eng = self._impute_missing_values(df_eng)

        # Ticket frequency and adjusted fare
        ticket_counts = df_eng["Ticket"].value_counts()
        df_eng["TicketFreq"] = df_eng["Ticket"].map(ticket_counts)
        df_eng["FareAdj"] = df_eng["Fare"] / df_eng["TicketFreq"]

        # Family size
        df_eng["FamilySize"] = df_eng["SibSp"] + df_eng["Parch"] + 1

        # Group ID engineering for WCG
        df_eng["Surname"] = df_eng["Name"].str.extract(r"^([^,]+)").iloc[:, 0]
        # Create ticket ID for grouping
        df_eng["TicketId"] = (
            df_eng["Pclass"].astype(str)
            + "-"
            + df_eng["Ticket"].str.replace(r".$", "X", regex=True)
            + "-"
            + df_eng["Fare"].astype(str)
            + "-"
            + df_eng["Embarked"].fillna("S")
        )

        # Create GroupId - initially surname-based for women and children
        df_eng["GroupId"] = (
            df_eng["Surname"]
            + "-"
            + df_eng["Pclass"].astype(str)
            + "-"
            + df_eng["Ticket"].str.replace(r".$", "X", regex=True)
            + "-"
            + df_eng["Fare"].astype(str)
            + "-"
            + df_eng["Embarked"].fillna("S")
        )

        # Set men to 'noGroup'
        df_eng.loc[df_eng["Title"] == "man", "GroupId"] = "noGroup"

        # Group frequency and remove single-person groups
        group_counts = df_eng["GroupId"].value_counts()
        df_eng["GroupFreq"] = df_eng["GroupId"].map(group_counts)
        df_eng.loc[df_eng["GroupFreq"] <= 1, "GroupId"] = "noGroup"

        # Add nannies/relatives to groups by ticket matching
        no_group_mask = (df_eng["Title"] != "man") & (df_eng["GroupId"] == "noGroup")
        for idx in df_eng[no_group_mask].index:
            ticket_id = df_eng.loc[idx, "TicketId"]
            matching_groups = df_eng[df_eng["TicketId"] == ticket_id]["GroupId"]
            valid_groups = matching_groups[matching_groups != "noGroup"]
            if len(valid_groups) > 0:
                df_eng.loc[idx, "GroupId"] = valid_groups.iloc[0]

        return df_eng

    def _calculate_group_survival(self, df_train):
        """Calculate group survival rates from training data"""
        group_survival = {}

        # Calculate survival rate for each group
        for group_id in df_train["GroupId"].unique():
            if group_id == "noGroup":
                continue
            group_data = df_train[df_train["GroupId"] == group_id]
            if len(group_data) > 0:
                survival_rate = group_data["Survived"].mean()
                group_survival[group_id] = survival_rate

        return group_survival

    def fit(self, X, y):
        """Fit the complete WCG+XGBoost model"""
        # Engineer features
        df_eng = self._engineer_features(X.copy())
        df_eng["Survived"] = y

        # Calculate group survival rates
        self.group_survival = self._calculate_group_survival(df_eng)

        # Train XGBoost for adult males
        male_data = df_eng[df_eng["Title"] == "man"].copy()
        if len(male_data) > 0:
            # Create features exactly as in R: x1=Fare/(TicketFreq*10), x2=FamilySize+Age/70
            male_features = pd.DataFrame(
                {
                    "x1": male_data["Fare"] / (male_data["TicketFreq"] * 10),
                    "x2": male_data["FamilySize"] + male_data["Age"] / 70,
                }
            )
            male_labels = male_data["Survived"]

            dtrain_male = xgb.DMatrix(male_features, label=male_labels)
            self.male_model = xgb.train(
                params=self.xgb_params,
                dtrain=dtrain_male,
                num_boost_round=self.n_rounds,
                verbose_eval=False,
            )

        # Train XGBoost for solo females
        solo_female_data = df_eng[
            (df_eng["Title"] == "woman") & (df_eng["FamilySize"] == 1)
        ].copy()
        if len(solo_female_data) > 0:
            # Create features exactly as in R: x1=FareAdj/10, x2=Age/15
            female_features = pd.DataFrame(
                {
                    "x1": solo_female_data["FareAdj"] / 10,
                    "x2": solo_female_data["Age"] / 15,
                }
            )
            female_labels = solo_female_data["Survived"]

            dtrain_female = xgb.DMatrix(female_features, label=female_labels)
            self.female_model = xgb.train(
                params=self.xgb_params,
                dtrain=dtrain_female,
                num_boost_round=self.n_rounds,
                verbose_eval=False,
            )

        return self

    def predict(self, X):
        """Make predictions using the complete WCG+XGBoost model"""
        df_eng = self._engineer_features(X.copy())
        predictions = np.zeros(len(df_eng))

        # Step 1: Apply WCG rules (Women and Children first)
        # All females survive by default
        predictions[df_eng["Sex"] == "female"] = 1

        # Apply group survival rules
        for i, (idx, row) in enumerate(df_eng.iterrows()):
            group_id = row["GroupId"]
            if group_id in self.group_survival:
                group_survival_rate = self.group_survival[group_id]
                if row["Title"] == "woman" and group_survival_rate == 0:
                    predictions[i] = 0  # Woman in group where all died
                elif row["Title"] == "boy" and group_survival_rate == 1:
                    predictions[i] = 1  # Boy in group where all survived

        # Step 2: Apply XGBoost for adult males (threshold >= 0.9)
        if self.male_model is not None:
            male_mask = df_eng["Title"] == "man"
            if male_mask.any():
                male_data = df_eng[male_mask]
                male_features = pd.DataFrame(
                    {
                        "x1": male_data["Fare"] / (male_data["TicketFreq"] * 10),
                        "x2": male_data["FamilySize"] + male_data["Age"] / 70,
                    }
                )
                dtest_male = xgb.DMatrix(male_features)
                male_probs = self.male_model.predict(dtest_male)
                # Use threshold of 0.9 as in R model
                male_predictions = (male_probs >= 0.9).astype(int)
                predictions[male_mask] = male_predictions

        # Step 3: Apply XGBoost for solo females (threshold <= 0.08 for death)
        if self.female_model is not None:
            solo_female_mask = (df_eng["Title"] == "woman") & (
                df_eng["FamilySize"] == 1
            )
            # Only apply to females not already handled by WCG
            wcg_handled = df_eng["GroupId"].isin(self.group_survival.keys())
            solo_female_mask = solo_female_mask & ~wcg_handled

            if solo_female_mask.any():
                female_data = df_eng[solo_female_mask]
                female_features = pd.DataFrame(
                    {"x1": female_data["FareAdj"] / 10, "x2": female_data["Age"] / 15}
                )
                dtest_female = xgb.DMatrix(female_features)
                female_probs = self.female_model.predict(dtest_female)
                # Use threshold of 0.08 as in R model (predict death if p <= 0.08)
                female_predictions = (female_probs > 0.08).astype(int)
                predictions[solo_female_mask] = female_predictions

        return predictions.astype(int)


def main():
    """Test the complete WCG+XGBoost model"""
    print("🚀 COMPLETE WCG+XGBOOST MODEL TEST")
    print("=" * 50)

    # Load data
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    y = train["Survived"].astype(int)
    X = train.drop(columns=["Survived"])

    # Initialize model
    model = CompleteWCGXGBoostModel()

    # Cross-validation
    print("📈 CROSS-VALIDATION RESULTS:")
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"   CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"   Individual folds: {[f'{score:.3f}' for score in cv_scores]}")

        if cv_scores.mean() >= 0.84:
            print(
                f"✅ SUCCESS: CV accuracy ({cv_scores.mean():.1%}) meets/exceeds 84%!"
            )
        else:
            print(
                f"⚠️  WARNING: CV accuracy ({cv_scores.mean():.1%}) is below promised 84%"
            )

    except Exception as e:
        print(f"   CV failed: {e}")

    # Train on full dataset and predict
    print("\n🏗️  TRAINING FINAL MODEL:")
    model.fit(X, y)

    # Generate predictions
    print("💾 GENERATING PREDICTIONS:")
    test_pred = model.predict(test)

    # Analysis
    df_eng = model._engineer_features(test)

    # Count different prediction groups
    male_predictions = test_pred[df_eng["Title"] == "man"]
    female_predictions = test_pred[df_eng["Sex"] == "female"]
    solo_female_predictions = test_pred[
        (df_eng["Title"] == "woman") & (df_eng["FamilySize"] == 1)
    ]

    print(f"   Total males surviving: {male_predictions.sum()}/{len(male_predictions)}")
    print(
        f"   Total females surviving: {female_predictions.sum()}/{len(female_predictions)}"
    )
    print(
        f"   Solo females surviving: {solo_female_predictions.sum()}/{len(solo_female_predictions)}"
    )
    print(f"   Overall survival rate: {test_pred.mean():.1%}")

    # Save submission
    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})

    output_path = (
        ROOT / "outputs" / "submissions" / "submission_complete_wcg_xgboost.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(f"   Saved: {output_path}")


if __name__ == "__main__":
    main()
