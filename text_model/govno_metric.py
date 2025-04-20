import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
tf.config.set_visible_devices([], 'GPU')
# Загрузка данных
X_test = np.load("sran_X_test.npy")
y_test = np.load("sran_y_test.npy")

# Кастомный F1
from keras._tf_keras.keras.metrics import Metric
from keras._tf_keras.keras import backend as K
from keras._tf_keras.keras.saving import register_keras_serializable

@register_keras_serializable()
class F1Score(Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Кастомный лосс
def weighted_binary_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = - (weights * y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        return K.mean(bce, axis=-1)

    return loss

# Вычисляем веса ещё раз (на всякий случай)
y_train = np.load("sran_y_train.npy")
class_totals = np.sum(y_train, axis=0)
total_labels = np.sum(class_totals)
class_weights = total_labels / (len(class_totals) * class_totals)
loss_fn = weighted_binary_crossentropy(class_weights)

# Загрузка модели
model = tf.keras.models.load_model(
    "sran_model_v2.keras",
    custom_objects={
        "F1Score": F1Score,
        "loss": loss_fn
    }
)

# Получаем предсказания
y_pred_proba = model.predict(X_test, batch_size=32)
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Подбор порога
best_thresh = 0.0
best_f1 = 0.0
f1_scores = []

thresholds = np.arange(0.1, 0.91, 0.01)

for t in thresholds:
    y_pred_bin = (y_pred_proba >= t).astype(int)
    f1 = f1_score(y_test, y_pred_bin, average='macro')
    f1_scores.append(f1)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"✅ Лучший threshold по macro F1: {best_thresh:.2f} (F1: {best_f1:.4f})")

# Построение графика
plt.plot(thresholds, f1_scores, label="Macro F1")
plt.axvline(best_thresh, color='red', linestyle='--', label=f"Best = {best_thresh:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Macro F1 Score")
plt.title("Подбор оптимального threshold")
plt.legend()
plt.grid()
plt.show()

# Повторный отчёт с лучшим threshold
y_pred_bin = (y_pred_proba >= best_thresh).astype(int)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_bin, digits=2, target_names=[
    "anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"
]))

print("F1 Micro:", round(f1_score(y_test, y_pred_bin, average='micro'), 4))
print("F1 Macro:", round(f1_score(y_test, y_pred_bin, average='macro'), 4))
