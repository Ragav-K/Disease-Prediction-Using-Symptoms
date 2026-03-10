import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

X_train = X_train.astype("float32")


rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=50,       
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=2,
    random_state=42
)

et = ExtraTreesClassifier(
    n_estimators=150,
    max_depth=50,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=2,
    random_state=42
)

print("Training Random Forest...")
rf.fit(X_train, y_train)

print("Training Extra Trees...")
et.fit(X_train, y_train)

joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(et, "models/extra_trees.pkl")

print("Models Trained & Saved Successfully (Removed weak DecisionTree)")