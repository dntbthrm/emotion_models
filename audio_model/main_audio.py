import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.concat([
    pd.read_csv("ravdess_features_2.csv"),
    pd.read_csv("crema_features_2.csv"),
    pd.read_csv("tess_features_2.csv")
])

X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_params = {'subsample': 0.8,
               'n_estimators': 400,
               'min_samples_split': 15,
               'max_depth': 7,
               'learning_rate': 0.05
               }

model = GradientBoostingClassifier(**best_params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

emotion_codes = {
    1 : "neutral",
    2 : "happy",
    3 : "sad" ,
    4 : "angry" ,
    5 : "fearful",
    6 : "disgust" ,
    7 :  "surprised"
}
target_names = [emotion_codes[i] for i in range(1, 8)]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred, target_names= target_names)

with open("final_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(class_report)

plt.figure(figsize=(10, 16))
plt.barh(X.columns, model.feature_importances_)
plt.xlabel("Важность признака")
plt.ylabel("Название признака")
plt.title("Важность признаков в модели")
plt.savefig("feature_importance_big.png")

joblib.dump(model, "audio_classifier.pkl")

