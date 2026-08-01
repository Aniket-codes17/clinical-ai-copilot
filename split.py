import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("data.csv")


X = df.drop(columns=["target"])
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"Updated Accuracy: {acc:.4f}")



from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X_train, y_train)
print("forest accuracy:", rf.score(X_test, y_test))




from sklearn.metrics import classification_report
print(classification_report(y_test, rf.predict(X_test)))






import joblib
joblib.dump(rf, "clinical_model.joblib")