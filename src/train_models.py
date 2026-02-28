import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

X_train = X_train.astype("float32")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

dt = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

rf.fit(X_train, y_train)
dt.fit(X_train, y_train)
gb.fit(X_train, y_train)

joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(dt, "models/decision_tree.pkl")
joblib.dump(gb, "models/gradient_boosting.pkl")

print("Models Trained & Saved Successfully")