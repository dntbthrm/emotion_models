import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.inspection import permutation_importance
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

emotion_codes = {
    1: "neutral",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}
target_names = [emotion_codes[i] for i in range(1, 8)]

# загрузка датасета
data_1 = pd.read_csv("tess_features_2.csv")
data_2 = pd.read_csv("ravdess_features_2.csv")

data = pd.concat([data_1, data_2], ignore_index=True)

X = data.drop(columns=['label'])
y = data['label']

# выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.8, stratify=y_train, random_state=42)

# полный перебор параметров
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'max_depth': [5, 7, 10, 15],
    'subsample': [0.7, 0.8, 1.0],
    'min_samples_split': [2, 5, 10, 15]
}

clf = GradientBoostingClassifier()
grid_search = GridSearchCV(clf, param_grid, scoring='f1_weighted', cv=3, n_jobs=-1)
grid_search.fit(X_train_sample, y_train_sample)

# лучший классификатор и параметры
best_params = grid_search.best_params_
best_clf = grid_search.best_estimator_

# важные признаки
importance = best_clf.feature_importances_
plt.figure(figsize=(10, 16))
plt.barh(X.columns, best_clf.feature_importances_)
plt.xlabel("Важность признака")
plt.ylabel("Название признака")
plt.title("Важность признаков в модели")
plt.savefig("feature_importance_train_big.png")

top_features_idx = np.argsort(importance)[::-1][:7]
top_feature_names = X.columns[top_features_idx]



# топ-7 признаков в модели
X_train_top = X_train[top_feature_names]
X_test_top = X_test[top_feature_names]
best_clf.fit(X_train_top, y_train)

# оценка
y_pred = best_clf.predict(X_test_top)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred, target_names=target_names)

with open("results_train_big.txt", "w") as f:
    f.write(f"Лучшие гиперпараметры: {best_params}\n")
    f.write(f"Топ-7 признаков: {list(top_feature_names)}\n")
    f.write(f"Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}\n")
    f.write(class_report)

# графики
results = pd.DataFrame(grid_search.cv_results_)
sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.lineplot(x=results['param_n_estimators'], y=results['mean_test_score'], ax=axes[0])
axes[0].set_title("Влияние количества деревьев")

sns.lineplot(x=results['param_learning_rate'], y=results['mean_test_score'], ax=axes[1])
axes[1].set_title("Влияние learning rate")

sns.lineplot(x=results['param_max_depth'], y=results['mean_test_score'], ax=axes[2])
axes[2].set_title("Влияние глубины дерева")

plt.tight_layout()
plt.savefig("metrics_plot_train_big.png")
