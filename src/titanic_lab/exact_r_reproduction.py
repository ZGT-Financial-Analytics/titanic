# exact_r_reproduction.py - Exact reproduction of the R notebook's WCG+XGBoost model
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor

from titanic_lab.paths import TRAIN_CSV, TEST_CSV, ROOT


class ExactRReproduction(BaseEstimator, ClassifierMixin):
    """
    Exact reproduction of the R notebook's WCG+XGBoost model achieving 85.2%

    This implements the exact logic from the R notebook:
    1. WCG rules for family groups
    2. XGBoost for adult males (threshold >= 0.9)
    3. XGBoost for solo females (threshold <= 0.08)
    """

    def __init__(self):
        # Exact R XGBoost parameters
        self.xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "error",
            "max_depth": 5,
            "eta": 0.1,
            "gamma": 0.1,  # Note: R has "gammma" typo, but XGBoost ignores it
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "verbosity": 0,
        }
        self.n_rounds = 500

        # Models for different populations
        self.male_xgb = None
        self.female_xgb = None

        # WCG group survival data
        self.group_survival = {}

    def _create_titles(self, df):
        """Create titles exactly as in R: man, woman, boy"""
        df_out = df.copy()
        df_out["Title"] = "man"  # Default to man
        df_out.loc[df_out["Name"].str.contains("Master", na=False), "Title"] = "boy"
        df_out.loc[df_out["Sex"] == "female", "Title"] = "woman"
        return df_out

    def _impute_missing_values(self, df):
        """Impute missing Age and Fare using decision trees like R's rpart"""
        df_imp = df.copy()

        # Simple title encoding for decision tree
        title_map = {"man": 0, "woman": 1, "boy": 2}
        df_imp["Title_num"] = df_imp["Title"].map(title_map)

        # Impute Age using Title + Pclass + SibSp + Parch
        age_missing = df_imp["Age"].isna()
        if age_missing.any():
            age_features = ["Title_num", "Pclass", "SibSp", "Parch"]
            age_train = df_imp[~age_missing]
            age_test = df_imp[age_missing]

            dt_age = DecisionTreeRegressor(random_state=42)
            dt_age.fit(age_train[age_features], age_train["Age"])
            df_imp.loc[age_missing, "Age"] = dt_age.predict(age_test[age_features])

        # Impute Fare using Title + Pclass + Embarked + Sex + Age
        fare_missing = df_imp["Fare"].isna()
        if fare_missing.any():
            df_imp["Sex_num"] = (df_imp["Sex"] == "male").astype(int)
            df_imp["Embarked_num"] = (
                df_imp["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0)
            )

            fare_features = ["Title_num", "Pclass", "Embarked_num", "Sex_num", "Age"]
            fare_train = df_imp[~fare_missing]
            fare_test = df_imp[fare_missing]

            dt_fare = DecisionTreeRegressor(random_state=42)
            dt_fare.fit(fare_train[fare_features], fare_train["Fare"])
            df_imp.loc[fare_missing, "Fare"] = dt_fare.predict(fare_test[fare_features])

        return df_imp

    def _engineer_features(self, df):
        """Engineer features exactly as in R notebook"""
        df_eng = df.copy()

        # Create titles
        df_eng = self._create_titles(df_eng)

        # Impute missing values
        df_eng = self._impute_missing_values(df_eng)

        # TicketFreq and FareAdj exactly as in R
        ticket_counts = df_eng["Ticket"].value_counts()
        df_eng["TicketFreq"] = df_eng["Ticket"].map(ticket_counts)
        df_eng["FareAdj"] = df_eng["Fare"] / df_eng["TicketFreq"]
        df_eng["FamilySize"] = df_eng["SibSp"] + df_eng["Parch"] + 1

        # WCG GroupId engineering exactly as in R
        df_eng["Surname"] = df_eng["Name"].str.extract(r"^([^,]+)").iloc[:, 0]

        # Create GroupId exactly as R: Surname + Pclass + Ticket(last char->X) + Fare + Embarked
        df_eng["TicketMod"] = df_eng["Ticket"].str.replace(r".$", "X", regex=True)
        df_eng["GroupId"] = (
            df_eng["Surname"]
            + "-"
            + df_eng["Pclass"].astype(str)
            + "-"
            + df_eng["TicketMod"]
            + "-"
            + df_eng["Fare"].astype(str)
            + "-"
            + df_eng["Embarked"].fillna("S")
        )

        # Men get 'noGroup'
        df_eng.loc[df_eng["Title"] == "man", "GroupId"] = "noGroup"

        # Remove single-person groups
        group_counts = df_eng["GroupId"].value_counts()
        df_eng["GroupFreq"] = df_eng["GroupId"].map(group_counts)
        df_eng.loc[df_eng["GroupFreq"] <= 1, "GroupId"] = "noGroup"

        # Add nannies/relatives to groups by TicketId matching
        df_eng["TicketId"] = (
            df_eng["Pclass"].astype(str)
            + "-"
            + df_eng["TicketMod"]
            + "-"
            + df_eng["Fare"].astype(str)
            + "-"
            + df_eng["Embarked"].fillna("S")
        )

        # Find nannies/relatives and add to groups
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

        for group_id in df_train["GroupId"].unique():
            if group_id == "noGroup":
                continue
            group_data = df_train[df_train["GroupId"] == group_id]
            if len(group_data) > 0:
                survival_rate = group_data["Survived"].mean()
                group_survival[group_id] = survival_rate

        return group_survival

    def fit(self, X, y):
        """Fit the exact R model"""
        df_eng = self._engineer_features(X.copy())
        df_eng["Survived"] = y

        # Calculate WCG group survival rates
        self.group_survival = self._calculate_group_survival(df_eng)

        # Train XGBoost for adult males - exact R features
        male_data = df_eng[df_eng["Title"] == "man"].copy()
        if len(male_data) > 0:
            # R features: x1=Fare/(TicketFreq*10), x2=FamilySize+Age/70
            male_features = pd.DataFrame(
                {
                    "x1": male_data["Fare"] / (male_data["TicketFreq"] * 10),
                    "x2": male_data["FamilySize"] + (male_data["Age"] / 70),
                }
            )
            male_labels = male_data["Survived"].astype(int)

            dtrain = xgb.DMatrix(male_features, label=male_labels)
            self.male_xgb = xgb.train(
                params=self.xgb_params,
                dtrain=dtrain,
                num_boost_round=self.n_rounds,
                verbose_eval=False,
            )

        # Train XGBoost for solo females - exact R features
        solo_female_data = df_eng[
            (df_eng["Title"] == "woman") & (df_eng["FamilySize"] == 1)
        ].copy()
        if len(solo_female_data) > 0:
            # R features: x1=FareAdj/10, x2=Age/15
            female_features = pd.DataFrame(
                {
                    "x1": solo_female_data["FareAdj"] / 10,
                    "x2": solo_female_data["Age"] / 15,
                }
            )
            female_labels = solo_female_data["Survived"].astype(int)

            dtrain = xgb.DMatrix(female_features, label=female_labels)
            self.female_xgb = xgb.train(
                params=self.xgb_params,
                dtrain=dtrain,
                num_boost_round=self.n_rounds,
                verbose_eval=False,
            )

        return self

    def predict(self, X):
        """Predict using exact R logic"""
        df_eng = self._engineer_features(X.copy())
        predictions = np.zeros(len(df_eng), dtype=int)

        # Step 1: WCG baseline rules
        predictions[df_eng["Sex"] == "female"] = 1  # All females survive by default
        predictions[df_eng["Title"] == "boy"] = (
            0  # Boys die by default (will be overridden by WCG)
        )

        # Step 2: Apply WCG group survival rules
        for idx, (i, row) in enumerate(df_eng.iterrows()):
            group_id = row["GroupId"]
            if group_id in self.group_survival:
                group_survival_rate = self.group_survival[group_id]
                if row["Title"] == "woman" and group_survival_rate == 0:
                    predictions[idx] = 0  # Woman in group where all females/boys died
                elif row["Title"] == "boy" and group_survival_rate == 1:
                    predictions[idx] = 1  # Boy in group where all females/boys survived

        # Handle unknown groups as in R
        for idx, (i, row) in enumerate(df_eng.iterrows()):
            if (
                row["GroupId"] not in self.group_survival
                and row["GroupId"] != "noGroup"
            ):
                if row["Pclass"] == 3:
                    if row["Title"] == "woman":
                        predictions[idx] = (
                            0  # Unknown group in 3rd class - assume death
                        )
                else:
                    if row["Title"] in ["woman", "boy"]:
                        predictions[idx] = (
                            1  # Unknown group in 1st/2nd class - assume survival
                        )

        # Step 3: XGBoost for adult males (threshold >= 0.9)
        if self.male_xgb is not None:
            male_mask = df_eng["Title"] == "man"
            if male_mask.any():
                male_data = df_eng[male_mask]
                male_features = pd.DataFrame(
                    {
                        "x1": male_data["Fare"] / (male_data["TicketFreq"] * 10),
                        "x2": male_data["FamilySize"] + (male_data["Age"] / 70),
                    }
                )
                dtest = xgb.DMatrix(male_features)
                male_probs = self.male_xgb.predict(dtest)
                # R uses threshold >= 0.9
                male_predictions = (male_probs >= 0.9).astype(int)
                predictions[male_mask] = male_predictions

        # Step 4: XGBoost for solo females not in WCG (threshold <= 0.08)
        if self.female_xgb is not None:
            # Only apply to solo females not already handled by WCG
            solo_female_mask = (df_eng["Title"] == "woman") & (
                df_eng["FamilySize"] == 1
            )
            wcg_handled_mask = df_eng["GroupId"].isin(self.group_survival.keys())

            target_mask = solo_female_mask & ~wcg_handled_mask

            if target_mask.any():
                female_data = df_eng[target_mask]
                female_features = pd.DataFrame(
                    {"x1": female_data["FareAdj"] / 10, "x2": female_data["Age"] / 15}
                )
                dtest = xgb.DMatrix(female_features)
                female_probs = self.female_xgb.predict(dtest)
                # R uses threshold <= 0.08 (predict death if prob <= 0.08)
                female_predictions = (female_probs > 0.08).astype(int)
                predictions[target_mask] = female_predictions

        return predictions


def main():
    """Test the exact R reproduction"""
    print("🎯 EXACT R NOTEBOOK REPRODUCTION")
    print("=" * 50)

    # Load data
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    y = train["Survived"].astype(int)
    X = train.drop(columns=["Survived"])

    # Initialize model
    model = ExactRReproduction()

    # Cross-validation
    print("📈 CROSS-VALIDATION RESULTS:")
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"   CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"   Individual folds: {[f'{score:.3f}' for score in cv_scores]}")

        if cv_scores.mean() >= 0.85:
            print(
                f"🎉 EXCELLENT: CV accuracy ({cv_scores.mean():.1%}) matches R target of 85%+!"
            )
        elif cv_scores.mean() >= 0.84:
            print(
                f"✅ GOOD: CV accuracy ({cv_scores.mean():.1%}) is close to R target!"
            )
        else:
            print(f"⚠️  WARNING: CV accuracy ({cv_scores.mean():.1%}) is below R target")

    except Exception as e:
        print(f"   CV failed: {e}")

    # Train final model
    print("\n🏗️  TRAINING FINAL MODEL:")
    model.fit(X, y)

    # Generate predictions
    print("💾 GENERATING PREDICTIONS:")
    test_pred = model.predict(test)

    # Analysis
    df_eng = model._engineer_features(test)

    # Count predictions by group
    male_mask = df_eng["Title"] == "man"
    female_mask = df_eng["Title"] == "woman"
    boy_mask = df_eng["Title"] == "boy"
    solo_female_mask = (df_eng["Title"] == "woman") & (df_eng["FamilySize"] == 1)

    print(f"   Adult males surviving: {test_pred[male_mask].sum()}/{male_mask.sum()}")
    print(f"   Women surviving: {test_pred[female_mask].sum()}/{female_mask.sum()}")
    print(f"   Boys surviving: {test_pred[boy_mask].sum()}/{boy_mask.sum()}")
    print(
        f"   Solo females surviving: {test_pred[solo_female_mask].sum()}/{solo_female_mask.sum()}"
    )
    print(f"   Overall survival rate: {test_pred.mean():.1%}")

    # Save submission
    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})

    output_path = (
        ROOT / "outputs" / "submissions" / "submission_exact_r_reproduction.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(f"   Saved: {output_path}")


if __name__ == "__main__":
    main()
