import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


file_path = "data/cancer_detection_dataset.csv"

df = load_data(file_path)

print(df.head())
print(df.shape)