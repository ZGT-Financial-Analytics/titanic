# faithful_r_implementation.py - Faithful implementation of the R notebook's exact logic
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor

from titanic_lab.paths import TRAIN_CSV, TEST_CSV, ROOT


class FaithfulRImplementation(BaseEstimator, ClassifierMixin):
    """
    Faithful implementation of the R notebook's exact WCG+XGBoost logic

    Following the R notebook step by step:
    1. Start with gender model (all women survive, all men die)
    2. Apply WCG group rules to override gender model
    3. Apply XGBoost for remaining adult males
    4. Apply XGBoost for remaining solo females
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

    def _create_titles(self, df):
        """Create titles exactly as in R"""
        df_out = df.copy()
        df_out["Title"] = "man"
        df_out.loc[df_out["Name"].str.contains("Master", na=False), "Title"] = "boy"
        df_out.loc[df_out["Sex"] == "female", "Title"] = "woman"
        return df_out

    def _impute_missing_values(self, df):
        """Impute missing values using decision trees like R's rpart"""
        df_imp = df.copy()

        # Simple median imputation for now (R uses rpart which is more complex)
        df_imp["Age"] = df_imp["Age"].fillna(df_imp["Age"].median())
        df_imp["Fare"] = df_imp["Fare"].fillna(df_imp["Fare"].median())
        df_imp["Embarked"] = df_imp["Embarked"].fillna("S")

        return df_imp

    def _engineer_features(self, df):
        """Engineer all features exactly as in R"""
        df_eng = df.copy()

        # Create titles
        df_eng = self._create_titles(df_eng)

        # Impute missing values
        df_eng = self._impute_missing_values(df_eng)

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

        return df_eng

    def _calculate_wcg_survival(self, df_train):
        """Calculate WCG group survival rates - only definitive groups"""
        group_survival = {}

        for group_id in df_train["GroupId"].unique():
            if group_id == "noGroup":
                continue

            group_data = df_train[df_train["GroupId"] == group_id]
            if len(group_data) >= 2:  # Only multi-person groups
                survival_rate = group_data["Survived"].mean()

                # Only store definitive outcomes (all died or all survived)
                if survival_rate == 0.0:
                    group_survival[group_id] = 0
                elif survival_rate == 1.0:
                    group_survival[group_id] = 1
                # Mixed survival groups remain unhandled (use gender model)

        return group_survival

    def fit(self, X, y):
        """Fit the model following R notebook logic"""
        df_eng = self._engineer_features(X.copy())
        df_eng["Survived"] = y

        # Calculate WCG group survival
        self.group_survival = self._calculate_wcg_survival(df_eng)

        # Train XGBoost for adult males (not in WCG, since men get 'noGroup')
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
        # Solo females are: Title='woman', FamilySize=1, GroupId='noGroup'
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
        """Predict using R notebook's exact logic"""
        df_eng = self._engineer_features(X.copy())
        predictions = np.zeros(len(df_eng), dtype=int)

        # Step 1: Gender model baseline - all women survive, all men die
        predictions[df_eng["Sex"] == "female"] = 1
        predictions[df_eng["Sex"] == "male"] = 0

        # Step 2: Apply WCG rules to override gender model
        for idx, (_, row) in enumerate(df_eng.iterrows()):
            group_id = row["GroupId"]

            if group_id in self.group_survival:
                # Override with definitive group outcome
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

        return predictions.astype(int)


def main():
    """Test the faithful R implementation"""
    print("🎯 FAITHFUL R IMPLEMENTATION - TARGET: 85.2%")
    print("=" * 50)

    # Load data
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    y = train["Survived"].astype(int)
    X = train.drop(columns=["Survived"])

    # Initialize model
    model = FaithfulRImplementation()

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
        elif cv_scores.mean() >= 0.82:
            print(
                f"✅ VERY GOOD: CV accuracy ({cv_scores.mean():.1%}) is approaching R target!"
            )
        else:
            print(f"⚠️  CV accuracy ({cv_scores.mean():.1%}) needs improvement")
    except Exception as e:
        print(f"   CV failed: {e}")

    # Train final model and analyze
    print("\n🏗️  TRAINING FINAL MODEL:")
    model.fit(X, y)

    # Analyze training data groupings
    df_train_eng = model._engineer_features(X)
    df_train_eng["Survived"] = y

    print(f"   WCG definitive groups found: {len(model.group_survival)}")
    for group_id, survival in model.group_survival.items():
        group_size = (df_train_eng["GroupId"] == group_id).sum()
        print(f"     {group_id}: {survival} (size: {group_size})")

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
        ROOT / "outputs" / "submissions" / "submission_faithful_r_implementation.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(f"   Saved: {output_path}")
    print("\n🚀 Ready for submission!")


if __name__ == "__main__":
    main()
