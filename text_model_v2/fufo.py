from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# ✅ Загрузка модели
model_dir = "emotion_bert_model"
#model_dir = "saved_rubert_model"
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
model.eval()

# ✅ Словарь классов
basic_dict = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

# ✅ Предсказание
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return basic_dict[predicted_class]

# ✅ Пример
example_texts = ["Пошел нахер!", "Я очень скучаю по тебе, мне плохо", "Ну и гадость, меня тошнит от брокколи", "Вау, не может быть!", "Я радо что ты здесь. Спасибо", "Я ходила в магазин сегодня"]
for txt in example_texts:
    predicted = predict_emotion(txt)
    print(f"💬 Текст: {txt}\n🔮 Эмоция: {predicted}")
