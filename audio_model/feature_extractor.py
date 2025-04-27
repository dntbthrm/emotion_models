import os
import librosa
import numpy as np
import pandas as pd
import time
import logging
from tqdm import tqdm

# логирование
logging.basicConfig(
    filename="preprocessing_2.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# извлечение признаков
def extract_features(file_path):
    try:
        start_time = time.time()
        y, sr = librosa.load(file_path, sr=22050)

        # акустические признаки
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        features = np.hstack([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
            np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1),
            np.mean(zcr), np.std(zcr),
            np.mean(rms), np.std(rms)
        ])

        elapsed_time = time.time() - start_time
        logging.info(f"Признаки извлечены: {file_path} (Время: {elapsed_time:.2f} сек)")
        return features
    except Exception as e:
        logging.error(f"Ошибка при обработке {file_path}: {e}")
        return None


# обработка датасета и создания CSV
def process_dataset(dataset_path, emotions_map, output_csv, dataset):
    data = []
    total_files = sum(len(files) for _, _, files in os.walk(dataset_path))
    logging.info(f"обработка {dataset_path}. Файлов для обработки: {total_files}")

    for root, _, files in os.walk(dataset_path):
        for file in tqdm(files):
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                label = get_emotion_label(file, emotions_map, dataset)
                label_code = int(emotion_codes.get(label))

                logging.info(f"Файл: {file_path} (Эмоция: {label})")

                features = extract_features(file_path)
                if features is not None:
                    data.append(np.hstack([features, label_code]))

    columns = [f"mfcc_{i}" for i in range(13)] + \
              [f"mfcc_std_{i}" for i in range(13)] + \
              [f"chroma_{i}" for i in range(12)] + \
              [f"chroma_std_{i}" for i in range(12)] + \
              [f"spectral_contrast_{i}" for i in range(7)] + \
              [f"spectral_contrast_std_{i}" for i in range(7)] + \
              [f"tonnetz_{i}" for i in range(6)] + \
              [f"tonnetz_std_{i}" for i in range(6)] + \
              ["zero_crossing_rate", "zero_crossing_rate_std", "rms", "rms_std", "label"]

    df = pd.DataFrame(data, columns=columns)

    # CSV
    df.to_csv(output_csv, index=False)
    logging.info(f"Датасет сохранен в {output_csv} (Всего записей: {len(df)})")


# эмоции по имени файла
def get_emotion_label(filename, emotions_map, dataset):
    if dataset == 1:
        for key in emotions_map:
            emo_code = filename.split('-')[2]
            if key in emo_code:
                return emotions_map[key]
    else:
        for key in emotions_map:
            if key in filename:
                return emotions_map[key]
    return "unknown"


# карты эмоций
ravdess_map = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fearful",
               "07": "disgust", "08": "surprised"}
crema_map = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fearful", "DIS": "disgust"}
tess_map = {"neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry", "fear": "fearful",
            "disgust": "disgust", "_ps": "surprised"}


emotion_codes = {
    "neutral" : 1,
    "happy" : 2,
    "sad" : 3,
    "angry" : 4,
    "fearful" : 5,
    "disgust" : 6,
    "surprised" : 7
}

# пути к датасетам
ravdess_path = "../../dats/ravdess"
crema_path = "../../dats/CREMA-D"
tess_path = "../../dats/TESS"


# 1- ravdess
# 2 - crema
# 3 - tess

# запускть по очереди, а то комп ляжет
#process_dataset(ravdess_path, ravdess_map, "ravdess_features_2.csv", 1)
#process_dataset(crema_path, crema_map, "crema_features_2.csv", 2)
#process_dataset(tess_path, tess_map, "tess_features_2.csv", 3)
