import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# Load models
rf = joblib.load("random_forest.pkl")
dt = joblib.load("decision_tree.pkl")
nb = joblib.load("naive_bayes.pkl")

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