import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Reduce memory usage
X_train = X_train.astype("float32")

# Initialize models (memory optimized)
rf = RandomForestClassifier(
    n_estimators=20,
    max_depth=10,
    n_jobs=1,
    random_state=42
)

dt = DecisionTreeClassifier(max_depth=10, random_state=42)
nb = GaussianNB()

# Train models
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)
nb.fit(X_train, y_train)

# Save models
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(dt, "models/decision_tree.pkl")
joblib.dump(nb, "models/naive_bayes.pkl")

print("Models Trained & Saved Successfully")