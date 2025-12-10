import pandas as pd
from sklearn.preprocessing import StandardScaler

# Normalisation des données. Comme vous pouvez le noter, les données sont dans des échelles très variés donc une normalisation est nécessaire. 
# Vous pouvez utiliser des fonctions pré-existantes pour la construction de ce script. 
# En sortie, ce script créera deux nouveaux datasets : (X_train_scaled, X_test_scaled) que vous sauvegarderez également dans data/processed.

# Chargement des datasets splittés
X_train = pd.read_csv("data/processed_data/X_train.csv")
X_test = pd.read_csv("data/processed_data/X_test.csv")

# Initialisation du scaler
scaler = StandardScaler()

# Fit sur X_train et transformer
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarde des datasets normalisés dans data/processed
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("data/processed_data/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("data/processed_data/X_test_scaled.csv", index=False)
