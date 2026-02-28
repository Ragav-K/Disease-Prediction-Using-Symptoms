import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
rf = joblib.load("models/random_forest.pkl")
dt = joblib.load("models/decision_tree.pkl")
gb = joblib.load("models/gradient_boosting.pkl")
ensemble_model = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("dt", dt),
        ("gb", gb)
    ],
    voting="soft"
)
ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)
print("Ensemble Accuracy:", accuracy_score(y_test, y_pred))
joblib.dump(ensemble_model, "models/ensemble_model.pkl")