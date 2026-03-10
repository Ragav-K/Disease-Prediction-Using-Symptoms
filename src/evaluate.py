import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report


X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()


print("Loading models for evaluation...")
rf = joblib.load("models/random_forest.pkl")
dt = joblib.load("models/decision_tree.pkl")
et = joblib.load("models/extra_trees.pkl")

models = {
    "Random Forest": rf,
    "Decision Tree": dt,
    "Extra Trees": et
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n================ {name} ================")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

print("\n================ Ensemble Model (Soft Voting) ================")
rf_proba = rf.predict_proba(X_test)
dt_proba = dt.predict_proba(X_test)
et_proba = et.predict_proba(X_test)

avg_proba = (rf_proba + dt_proba + et_proba) / 3
y_pred_ensemble = np.argmax(avg_proba, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print(classification_report(y_test, y_pred_ensemble))