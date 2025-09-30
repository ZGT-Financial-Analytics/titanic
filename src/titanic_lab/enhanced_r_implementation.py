# enhanced_r_implementation.py - Enhanced with proper imputation and ensemble
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor

from titanic_lab.paths import TRAIN_CSV, TEST_CSV, ROOT


class EnhancedRImplementation(BaseEstimator, ClassifierMixin):
    """
    Enhanced R implementation with proper imputation and ensemble logic
    """

    def __init__(self):
        # Exact R XGBoost parameters
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

        # Component models
        self.male_xgb = None
        self.female_xgb = None

        # WCG data structures
        self.group_survival = {}

        # Age imputation model
        self.age_imputer = None
        self.fare_imputer = None

    def _create_titles(self, df):
        """Create titles exactly as in R"""
        df_out = df.copy()
        df_out["Title"] = "man"
        df_out.loc[df_out["Name"].str.contains("Master", na=False), "Title"] = "boy"
        df_out.loc[df_out["Sex"] == "female", "Title"] = "woman"
        return df_out

    def _impute_missing_values(self, df, fit_imputers=False):
        """Impute missing values using decision trees like R's rpart"""
        df_imp = df.copy()

        # Create titles for imputation
        df_imp = self._create_titles(df_imp)

        # Title encoding for imputation
        title_map = {"man": 0, "woman": 1, "boy": 2}
        df_imp["Title_num"] = df_imp["Title"].map(title_map)

        # Embarked imputation (simple mode)
        df_imp["Embarked"] = df_imp["Embarked"].fillna("S")
        df_imp["Embarked_num"] = df_imp["Embarked"].map({"S": 0, "C": 1, "Q": 2})
        df_imp["Sex_num"] = (df_imp["Sex"] == "male").astype(int)

        # Age imputation using decision tree
        age_missing = df_imp["Age"].isna()
        if age_missing.any():
            age_features = [
                "Title_num",
                "Pclass",
                "SibSp",
                "Parch",
                "Sex_num",
                "Embarked_num",
            ]

            if fit_imputers:
                # Fit imputer on training data
                age_train = df_imp[~age_missing]
                self.age_imputer = DecisionTreeRegressor(random_state=42, max_depth=10)
                self.age_imputer.fit(age_train[age_features], age_train["Age"])

            if self.age_imputer is not None:
                age_test = df_imp[age_missing]
                df_imp.loc[age_missing, "Age"] = self.age_imputer.predict(
                    age_test[age_features]
                )

        # Fare imputation using decision tree
        fare_missing = df_imp["Fare"].isna()
        if fare_missing.any():
            fare_features = ["Title_num", "Pclass", "Embarked_num", "Sex_num"]
            # Add Age if not missing
            if not df_imp["Age"].isna().any():
                fare_features.append("Age")

            if fit_imputers:
                # Fit imputer on training data
                fare_train = df_imp[~fare_missing]
                self.fare_imputer = DecisionTreeRegressor(random_state=42, max_depth=10)
                self.fare_imputer.fit(fare_train[fare_features], fare_train["Fare"])

            if self.fare_imputer is not None:
                fare_test = df_imp[fare_missing]
                df_imp.loc[fare_missing, "Fare"] = self.fare_imputer.predict(
                    fare_test[fare_features]
                )

        return df_imp

    def _engineer_features(self, df, fit_imputers=False):
        """Engineer all features exactly as in R"""
        df_eng = df.copy()

        # Impute missing values first
        df_eng = self._impute_missing_values(df_eng, fit_imputers=fit_imputers)

        # Core feature engineering from R
        ticket_counts = df_eng["Ticket"].value_counts()
        df_eng["TicketFreq"] = df_eng["Ticket"].map(ticket_counts)
        df_eng["FareAdj"] = df_eng["Fare"] / df_eng["TicketFreq"]
        df_eng["FamilySize"] = df_eng["SibSp"] + df_eng["Parch"] + 1

        # WCG GroupId engineering - exact R logic
        df_eng["Surname"] = df_eng["Name"].str.extract(r"^([^,]+)").iloc[:, 0]
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
            + df_eng["Embarked"]
        )

        # Men get 'noGroup' (they don't participate in WCG)
        df_eng.loc[df_eng["Title"] == "man", "GroupId"] = "noGroup"

        # Remove single-person groups (they become 'noGroup')
        group_counts = df_eng["GroupId"].value_counts()
        df_eng["GroupFreq"] = df_eng["GroupId"].map(group_counts)
        df_eng.loc[df_eng["GroupFreq"] <= 1, "GroupId"] = "noGroup"

        # Add nannies to groups (from R notebook)
        # Solo women/children who share tickets with groups should join those groups
        no_group_mask = (df_eng["Title"] != "man") & (df_eng["GroupId"] == "noGroup")
        for idx in df_eng[no_group_mask].index:
            ticket = df_eng.loc[idx, "Ticket"]
            pclass = df_eng.loc[idx, "Pclass"]
            fare = df_eng.loc[idx, "Fare"]
            embarked = df_eng.loc[idx, "Embarked"]

            # Find groups with same ticket/class/fare/embarked
            matching_groups = df_eng[
                (df_eng["Ticket"] == ticket)
                & (df_eng["Pclass"] == pclass)
                & (df_eng["Fare"] == fare)
                & (df_eng["Embarked"] == embarked)
                & (df_eng["GroupId"] != "noGroup")
            ]["GroupId"]

            if len(matching_groups) > 0:
                df_eng.loc[idx, "GroupId"] = matching_groups.iloc[0]

        return df_eng

    def _calculate_wcg_survival(self, df_train):
        """Calculate WCG group survival rates - including mixed groups with majority rule"""
        group_survival = {}

        for group_id in df_train["GroupId"].unique():
            if group_id == "noGroup":
                continue

            group_data = df_train[df_train["GroupId"] == group_id]
            if len(group_data) >= 2:  # Only multi-person groups
                survival_rate = group_data["Survived"].mean()

                # Store definitive outcomes
                if survival_rate == 0.0:
                    group_survival[group_id] = 0
                elif survival_rate == 1.0:
                    group_survival[group_id] = 1
                # For mixed groups, use majority rule if strong majority (>=75% or <=25%)
                elif survival_rate >= 0.75:
                    group_survival[group_id] = 1
                elif survival_rate <= 0.25:
                    group_survival[group_id] = 0
                # Otherwise leave to gender model

        return group_survival

    def fit(self, X, y):
        """Fit the enhanced model"""
        df_eng = self._engineer_features(X.copy(), fit_imputers=True)
        df_eng["Survived"] = y

        # Calculate WCG group survival with enhanced rules
        self.group_survival = self._calculate_wcg_survival(df_eng)

        # Train XGBoost for adult males
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

        # Train XGBoost for solo females not handled by WCG
        solo_female_data = df_eng[
            (df_eng["Title"] == "woman")
            & (df_eng["FamilySize"] == 1)
            & (df_eng["GroupId"] == "noGroup")
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
        """Predict using enhanced R notebook logic"""
        df_eng = self._engineer_features(X.copy(), fit_imputers=False)
        predictions = np.zeros(len(df_eng), dtype=int)

        # Step 1: Gender model baseline - all women survive, all men die
        predictions[df_eng["Sex"] == "female"] = 1
        predictions[df_eng["Sex"] == "male"] = 0

        # Step 2: Apply WCG rules to override gender model
        for idx, (_, row) in enumerate(df_eng.iterrows()):
            group_id = row["GroupId"]

            if group_id in self.group_survival:
                # Override with group outcome
                predictions[idx] = self.group_survival[group_id]

        # Step 3: Apply XGBoost for adult males (threshold >= 0.9)
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

                # R uses threshold >= 0.9 for male survival
                male_predictions = (male_probs >= 0.9).astype(int)
                predictions[male_mask] = male_predictions

        # Step 4: Apply XGBoost for solo females not in WCG (threshold <= 0.08)
        if self.female_xgb is not None:
            solo_female_mask = (
                (df_eng["Title"] == "woman")
                & (df_eng["FamilySize"] == 1)
                & (df_eng["GroupId"] == "noGroup")
            )

            if solo_female_mask.any():
                female_data = df_eng[solo_female_mask]
                female_features = pd.DataFrame(
                    {"x1": female_data["FareAdj"] / 10, "x2": female_data["Age"] / 15}
                )
                dtest = xgb.DMatrix(female_features)
                female_probs = self.female_xgb.predict(dtest)

                # R uses threshold <= 0.08 for female death prediction
                female_predictions = (female_probs > 0.08).astype(int)
                predictions[solo_female_mask] = female_predictions

        # Step 5: Enhanced ensemble logic based on R notebook patterns
        # Apply additional rules for edge cases observed in R
        for idx, (_, row) in enumerate(df_eng.iterrows()):
            # Skip if already handled by WCG
            if row["GroupId"] in self.group_survival:
                continue

            # Enhanced decision rules based on R notebook patterns
            # Young boys in 1st/2nd class are more likely to survive
            if row["Title"] == "boy" and row["Pclass"] <= 2 and row["Age"] < 10:
                predictions[idx] = 1

            # Elderly women in 3rd class with low fare are at risk
            elif (
                row["Title"] == "woman"
                and row["Pclass"] == 3
                and row["Age"] > 55
                and row["FareAdj"] < 10
            ):
                predictions[idx] = 0

            # Women traveling alone in 1st class almost always survive
            elif (
                row["Title"] == "woman"
                and row["Pclass"] == 1
                and row["FamilySize"] == 1
            ):
                predictions[idx] = 1

            # Adult males with high fare and small family have better chance
            elif (
                row["Title"] == "man" and row["FareAdj"] > 50 and row["FamilySize"] <= 2
            ):
                # Don't override XGBoost if it predicted survival
                if predictions[idx] == 0 and row["Pclass"] == 1:
                    predictions[idx] = 1

        return predictions.astype(int)


def main():
    """Test the enhanced R implementation"""
    print("🎯 ENHANCED R IMPLEMENTATION - TARGET: 85.2%")
    print("=" * 50)

    # Load data
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    y = train["Survived"].astype(int)
    X = train.drop(columns=["Survived"])

    # Initialize model
    model = EnhancedRImplementation()

    # Cross-validation
    print("📈 CROSS-VALIDATION RESULTS:")
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"   CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"   Individual folds: {[f'{score:.3f}' for score in cv_scores]}")

        if cv_scores.mean() >= 0.85:
            print(
                f"🎉 EXCELLENT: CV accuracy ({cv_scores.mean():.1%}) matches/exceeds R target!"
            )
        elif cv_scores.mean() >= 0.83:
            print(
                f"✅ VERY GOOD: CV accuracy ({cv_scores.mean():.1%}) is very close to R target!"
            )
        elif cv_scores.mean() >= 0.81:
            print(
                f"🔥 GOOD: CV accuracy ({cv_scores.mean():.1%}) is approaching R target!"
            )
        else:
            print(f"⚠️  CV accuracy ({cv_scores.mean():.1%}) needs improvement")
    except Exception as e:
        print(f"   CV failed: {e}")

    # Train final model and analyze
    print("\n🏗️  TRAINING FINAL MODEL:")
    model.fit(X, y)

    # Analyze training data groupings
    df_train_eng = model._engineer_features(X, fit_imputers=False)
    df_train_eng["Survived"] = y

    print(f"   WCG definitive groups found: {len(model.group_survival)}")
    definitive_groups = sum(1 for v in model.group_survival.values() if v in [0, 1])
    majority_groups = len(model.group_survival) - definitive_groups
    print(f"   Definitive groups (100% or 0%): {definitive_groups}")
    print(f"   Majority rule groups (75%+ or 25%-): {majority_groups}")

    # Generate test predictions
    print("\n💾 GENERATING TEST PREDICTIONS:")
    test_pred = model.predict(test)

    # Analyze test predictions
    df_test_eng = model._engineer_features(test)

    male_mask = df_test_eng["Title"] == "man"
    female_mask = df_test_eng["Title"] == "woman"
    boy_mask = df_test_eng["Title"] == "boy"
    wcg_handled_mask = df_test_eng["GroupId"].isin(model.group_survival.keys())
    solo_female_mask = (
        (df_test_eng["Title"] == "woman")
        & (df_test_eng["FamilySize"] == 1)
        & (df_test_eng["GroupId"] == "noGroup")
    )

    print(f"   Adult males surviving: {test_pred[male_mask].sum()}/{male_mask.sum()}")
    print(f"   Women surviving: {test_pred[female_mask].sum()}/{female_mask.sum()}")
    print(f"   Boys surviving: {test_pred[boy_mask].sum()}/{boy_mask.sum()}")
    print(f"   WCG handled passengers: {wcg_handled_mask.sum()}")
    print(f"   Solo females (XGBoost): {solo_female_mask.sum()}")
    print(
        f"   Solo females surviving: {test_pred[solo_female_mask].sum()}/{solo_female_mask.sum()}"
    )
    print(f"   Overall survival rate: {test_pred.mean():.1%}")

    # Save submission
    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})

    output_path = (
        ROOT / "outputs" / "submissions" / "submission_enhanced_r_implementation.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(f"   Saved: {output_path}")
    print("\n🚀 Ready for submission!")


if __name__ == "__main__":
    main()
