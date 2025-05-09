import pandas as pd

df_1 = pd.read_csv("tess_features_2.csv")
df_2 = pd.read_csv("ravdess_features_2.csv")
df = pd.concat([df_1, df_2], ignore_index=True)
print(df.shape)