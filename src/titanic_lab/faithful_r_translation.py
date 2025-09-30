# faithful_r_translation.py - Exact Python translation of the R XGBoost model
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin

from titanic_lab.paths import TRAIN_CSV, TEST_CSV, ROOT


class FaithfulRTranslation(BaseEstimator, ClassifierMixin):
    """
    Faithful translation of the R XGBoost model that targets solo females specifically.
    """

    def __init__(self):
        # Exact R parameters
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
        self.model = None

    def _engineer_features(self, df):
        """Create the exact features used in R code"""
        df_eng = df.copy()

        # Extract title (equivalent to R's title extraction)
        df_eng["Title"] = df_eng["Name"].str.extract(r",\s*([^\.]+)\.")
        df_eng["Title"] = df_eng["Title"].str.strip().str.lower()

        # Normalize titles - map to 'woman' for adult females
        title_map = {
            "miss": "woman",  # Assuming miss -> woman for this model
            "mrs": "woman",
            "ms": "woman",
            "mme": "woman",
            "mlle": "woman",
            "lady": "woman",
            "the countess": "woman",
        }
        df_eng["Title"] = df_eng["Title"].map(title_map).fillna(df_eng["Title"])

        # Family size (same as original)
        df_eng["FamilySize"] = df_eng["SibSp"].fillna(0) + df_eng["Parch"].fillna(0) + 1

        # For ticket frequency calculation, we need to compute FareAdj
        # Group by ticket and count occurrences
        ticket_counts = df_eng["Ticket"].value_counts()
        df_eng["TicketFreq"] = df_eng["Ticket"].map(ticket_counts)
        df_eng["FareAdj"] = df_eng["Fare"] / df_eng["TicketFreq"]

        # Transform features exactly as in R
        df_eng["x1"] = df_eng["FareAdj"] / 10  # FareAdj/10
        df_eng["x2"] = df_eng["Age"] / 15  # Age/15

        return df_eng

    def fit(self, X, y):
        """Fit the model using R's exact approach"""
        df_eng = self._engineer_features(X)

        # Filter to solo females only (equivalent to R filtering)
        # PassengerId <= 891 is handled by the training data itself
        solo_females = (df_eng["Title"] == "woman") & (df_eng["FamilySize"] == 1)

        if solo_females.sum() == 0:
            print("Warning: No solo females found in training data!")
            return self

        # Get solo female data
        train_data = df_eng[solo_females].copy()
        train_labels = y[solo_females]

        print(f"Training on {len(train_data)} solo females")

        # Prepare features for XGBoost (x1, x2 only, like R)
        feature_cols = ["x1", "x2"]
        X_train = train_data[feature_cols].fillna(0)  # Handle any NaN values

        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=train_labels)

        # Train XGBoost with exact R parameters
        self.model = xgb.train(
            params=self.xgb_params,
            dtrain=dtrain,
            num_boost_round=self.n_rounds,
            verbose_eval=100,
        )

        # Store feature info for prediction
        self.feature_cols = feature_cols

        return self

    def predict_proba(self, X):
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")

        df_eng = self._engineer_features(X)

        # Initialize predictions with default survival rate
        n_samples = len(X)
        proba = np.full((n_samples, 2), [0.6, 0.4])  # Default: 40% survival

        # Only predict for solo females
        solo_females = (df_eng["Title"] == "woman") & (df_eng["FamilySize"] == 1)

        if solo_females.sum() > 0:
            # Get features for solo females
            X_pred = df_eng.loc[solo_females, self.feature_cols].fillna(0)

            # Predict using XGBoost
            dtest = xgb.DMatrix(X_pred)
            prob_die = self.model.predict(dtest)  # P(Survived=1)

            # Apply R's threshold: predict death if p <= 0.25
            # This means predict survival if p > 0.25
            prob_survive = 1 - prob_die

            # Update probabilities for solo females
            proba[solo_females, 0] = prob_die  # P(death)
            proba[solo_females, 1] = prob_survive  # P(survival)

        return proba

    def predict(self, X):
        """Predict classes using R's threshold (p <= 0.25 for death)"""
        # Apply R's specific threshold logic
        df_eng = self._engineer_features(X)
        solo_females = (df_eng["Title"] == "woman") & (df_eng["FamilySize"] == 1)

        # Default predictions
        predictions = np.ones(len(X))  # Default to survival

        if solo_females.sum() > 0 and self.model is not None:
            # For solo females, use the model with R's threshold
            X_pred = df_eng.loc[solo_females, self.feature_cols].fillna(0)
            dtest = xgb.DMatrix(X_pred)
            prob_survive = self.model.predict(dtest)

            # R logic: predict death if p <= 0.25
            predictions[solo_females] = (prob_survive > 0.25).astype(int)

        return predictions


def main():
    """Test the faithful R translation"""
    print("🔄 FAITHFUL R TRANSLATION TEST")
    print("=" * 50)

    # Load data
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    y = train["Survived"].astype(int)
    X = train.drop(columns=["Survived"])

    # Initialize model
    model = FaithfulRTranslation()

    # Quick fit test to see solo female count
    print("📊 Data Analysis:")
    df_eng = model._engineer_features(X)
    solo_females = (df_eng["Title"] == "woman") & (df_eng["FamilySize"] == 1)
    print(f"   Total training samples: {len(X)}")
    print(f"   Solo females found: {solo_females.sum()}")

    # Show sample of solo female data
    if solo_females.sum() > 0:
        sample_data = df_eng[solo_females][
            ["Name", "Age", "Fare", "FareAdj", "x1", "x2"]
        ].head()
        print("\n📋 Sample solo female data:")
        print(sample_data.to_string())

    # Train and evaluate
    print("\n🚀 TRAINING FAITHFUL R MODEL:")
    model.fit(X, y)

    # Cross-validation (though limited by solo female subset)
    print("\n📈 CROSS-VALIDATION RESULTS:")
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"   CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"   Individual folds: {[f'{score:.3f}' for score in cv_scores]}")
    except Exception as e:
        print(f"   CV failed: {e}")

    # Generate predictions
    print("\n💾 GENERATING PREDICTIONS:")
    test_pred = model.predict(test).astype(int)  # Ensure integer type

    # Check predictions
    df_eng_test = model._engineer_features(test)
    solo_females_test = (df_eng_test["Title"] == "woman") & (
        df_eng_test["FamilySize"] == 1
    )
    print(f"   Test solo females: {solo_females_test.sum()}")
    print(f"   Predictions summary: {np.bincount(test_pred)}")

    # Debug title extraction
    print("\n🔍 TITLE DEBUG:")
    title_counts = df_eng["Title"].value_counts()
    print(f"   Training title distribution: {title_counts.to_dict()}")
    print(f"   Female count: {(df_eng['Title'] == 'woman').sum()}")
    print(f"   Solo female count: {solo_females.sum()}")

    # Save submission
    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})

    output_path = ROOT / "outputs" / "submissions" / "submission_faithful_r.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(f"   Saved: {output_path}")


if __name__ == "__main__":
    main()
