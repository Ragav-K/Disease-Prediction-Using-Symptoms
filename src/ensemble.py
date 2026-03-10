import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

print("Loading models...")
rf = joblib.load("models/random_forest.pkl")
dt = joblib.load("models/decision_tree.pkl")
et = joblib.load("models/extra_trees.pkl")

print("Calculating probabilities...")
rf_proba = rf.predict_proba(X_test)
dt_proba = dt.predict_proba(X_test)
et_proba = et.predict_proba(X_test)

avg_proba = (rf_proba + dt_proba + et_proba) / 3

y_pred = np.argmax(avg_proba, axis=1)

print("\n--- Ensemble Results ---")
print("Ensemble Accuracy:", accuracy_score(y_test, y_pred))
print("Note: Combined ensemble file not saved to prevent MemoryError (redundant anyway).")