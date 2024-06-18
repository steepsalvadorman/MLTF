import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 0

if __name__ == "__main__":
    data = pd.read_excel("C:/Users/Steep/Desktop/proyecto/Abraham/data/cefr_leveled_texts.xlsx")
    train, test = train_test_split(
        data, test_size=0.2, random_state=RANDOM_SEED, stratify=data.label
    )
    train.to_csv("C:/Users/Steep/Desktop/proyecto/Abraham/data/train.csv", index=False)
    test.to_csv("C:/Users/Steep/Desktop/proyecto/Abraham/data/test.csv", index=False)
