import pandas as pd
import joblib
import gc  
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


X_train = pd.read_csv("data/processed/X_train.csv").astype("float32")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()


print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=35,        
    min_samples_leaf=3,
    n_jobs=1,            
    random_state=42
)
rf.fit(X_train, y_train)
joblib.dump(rf, "models/random_forest.pkl")


del rf
gc.collect() 
print("Random Forest saved and cleared from RAM.")


print("Training Extra Trees...")
et = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=35,
    min_samples_leaf=3,
    n_jobs=1,
    random_state=42
)
et.fit(X_train, y_train)
joblib.dump(et, "models/extra_trees.pkl")

del et
gc.collect()
print("Extra Trees saved and cleared from RAM.")

print("\nAll Models Trained Successfully!")