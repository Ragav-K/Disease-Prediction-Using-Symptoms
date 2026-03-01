import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_train = X_train.astype("float32")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=2,
    random_state=42
)
dt = DecisionTreeClassifier(
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

et = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=2,
    random_state=42
)
rf.fit(X_train, y_train)
print("Random Forest trained")
dt.fit(X_train, y_train)
print("Decision Tree trained")
et.fit(X_train, y_train)
print("Extra Trees trained")
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(dt, "models/decision_tree.pkl")
joblib.dump(et, "models/extra_trees.pkl")
print("Models Trained & Saved Successfully")