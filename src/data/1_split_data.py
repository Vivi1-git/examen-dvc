import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Split des données en ensemble d'entraînement et de test. 
# Notre variable cible est silica_concentrate et se trouve dans la dernière colonne du dataset. 
# L'issu de ce script seront 4 datasets (X_test, X_train, y_test, y_train) que vous pouvez stocker dans data/processed

# Chargement du dataset
df = pd.read_csv("data/raw_data/raw.csv")

# Séparation features et target
y = df['silica_concentrate']                # colonne cible
X = df.drop(columns='silica_concentrate')   # toutes les autres colonnes

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarde des datasets dans data/processed_data
os.makedirs("data/processed/split", exist_ok=True)
X_train.to_csv("data/processed/split/X_train.csv", index=False)
X_test.to_csv("data/processed/split/X_test.csv", index=False)
y_train.to_csv("data/processed/split/y_train.csv", index=False)
y_test.to_csv("data/processed/split/y_test.csv", index=False)
