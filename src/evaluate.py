import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report


X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()


rf = joblib.load("models/random_forest.pkl")
dt = joblib.load("models/decision_tree.pkl")
nb = joblib.load("models/naive_bayes.pkl")

models = {
    "Random Forest": rf,
    "Decision Tree": dt,
    "Naive Bayes": nb
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))