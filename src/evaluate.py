import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

rf = joblib.load("models/random_forest.pkl")
et = joblib.load("models/extra_trees.pkl")

models = {"Random Forest": rf, "Extra Trees": et}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n================ {name} ================")
    print("Accuracy:", accuracy_score(y_test, y_pred))

# Combined Soft-Voting Ensemble
rf_proba = rf.predict_proba(X_test)
et_proba = et.predict_proba(X_test)
avg_proba = (rf_proba + et_proba) / 2
y_pred_ensemble = np.argmax(avg_proba, axis=1)

print("\n================ Final Optimized Ensemble ================")
print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print(classification_report(y_test, y_pred_ensemble))