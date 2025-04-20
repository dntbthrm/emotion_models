import numpy as np
import pickle
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation

# Параметры модели (из model_config)
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000

# Скачать нужные ресурсы один раз
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Загрузка модели
model = load_model('./emotion_model_small_v6.keras')

# Загрузка токенизатора
with open('v6/tokenizer_small.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Загрузка меток
with open('v6/label_to_index_small.pkl', 'rb') as f:
    label_to_index = pickle.load(f)

#index_to_label = {v: k for k, v in label_to_index.items()}
print(label_to_index)



# Текстовая предобработка
stop_words = stopwords.words('russian') + list(punctuation)
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower(), language='russian', preserve_line=True)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Предсказание
def predict_emotion(text):
    processed = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = model.predict(padded)[0]

    # Порог для мультилейбл классификации
    threshold = 0.5
    predicted_labels = [label_to_index[i] for i, prob in enumerate(prediction) if prob >= threshold]

    if not predicted_labels:
        # если ничего не превысило порог
        predicted_labels = [label_to_index[np.argmax(prediction)]]

    return predicted_labels

# Пример использования
if __name__ == '__main__':
    texts = [
        "Я так рад, что все получилось!",
        "Это просто ужасно, мне отвратительно",
        "Я не знаю, что теперь делать...",
        "Ну норм, в целом пойдет",
        "Это смешно, я не могу сдержать смех",
        "Я столько раз объяснял, как это должно быть сделано, и всё равно ты опять всё испортил, как будто специально!",
        "Когда я увидел, в каких условиях они хранят продукты, у меня сразу пропало всякое желание что-либо покупать у них.",
        "Я не мог дышать, когда услышал шаги за спиной в пустом подъезде — сердце колотилось так, будто вот-вот выскочит.",
        "Когда мы наконец добрались до вершины горы и увидели этот невероятный рассвет, я почувствовал, что счастлив как никогда.",
        "После того как она уехала, в доме стало так тихо и пусто, что каждый день напоминал о её отсутствии.",
        "Я и подумать не мог, что получу это письмо — оно пришло спустя десять лет и полностью изменило всё, во что я верил.",
        "Я проснулся, умылся, позавтракал овсянкой и пошёл на работу, как обычно по будням."
    ]

    for t in texts:
        print(f"\nТекст: {t}")
        print("Эмоции:", predict_emotion(t))
