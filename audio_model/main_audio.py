import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.concat([
    pd.read_csv("ravdess_features.csv"),
    pd.read_csv("crema_features.csv"),
    pd.read_csv("tess_features.csv")
])

X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_params = {
    'subsample': 0.8,
    'n_estimators': 200,
    'min_samples_split': 5,
    'max_depth': 7,
    'learning_rate': 0.1
}

model = GradientBoostingClassifier(**best_params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred)

with open("results_final.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(class_report)

plt.figure(figsize=(10, 6))
plt.barh(X.columns, model.feature_importances_)
plt.xlabel("Важность признака")
plt.ylabel("Название признака")
plt.title("Важность признаков в модели")
plt.savefig("feature_importance.png")

joblib.dump(model, "audio_model.pkl")

