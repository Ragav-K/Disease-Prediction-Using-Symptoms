import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/raw/dataset.csv")

X = df.drop("diseases", axis=1)
y = df["diseases"]

class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 10].index
mask = y.isin(valid_classes)
X = X[mask]
y = y[mask]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

joblib.dump(le, "models/label_encoder.pkl")

print(f"Preprocessing Completed. {len(le.classes_)} disease classes saved.")
