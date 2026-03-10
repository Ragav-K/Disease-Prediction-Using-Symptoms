import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

print("Loading models for final evaluation...")
rf = joblib.load("models/random_forest.pkl")
et = joblib.load("models/extra_trees.pkl")

rf_proba = rf.predict_proba(X_test)
et_proba = et.predict_proba(X_test)

avg_proba = (rf_proba + et_proba) / 2
y_pred_ensemble = np.argmax(avg_proba, axis=1)

print("\n================ FINAL OPTIMIZED ENSEMBLE ================")
print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("\nIndividual Accuracies:")
print(f" - Random Forest: {accuracy_score(y_test, np.argmax(rf_proba, axis=1))}")
print(f" - Extra Trees: {accuracy_score(y_test, np.argmax(et_proba, axis=1))}")

print("\n--- Full Classification Report ---")
print(classification_report(y_test, y_pred_ensemble))