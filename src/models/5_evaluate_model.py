import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json

# Evaluation du modèle. 
# Finalement, en utilisant le modèle entraîné on évaluera ses performances et on fera des prédictions avec ce modèle 
# de sorte qu'à la fin de ce script on aura un nouveau dataset dans data qui contiendra les predictions 
# ainsi qu'un fichier scores.json dans le dossier metrics qui récupérera les métriques d'évaluation de notre modèle (i.e. mse, r2, etc).

# Chargement des données test et du modèle
X_test = pd.read_csv("data/processed/norm/X_test_scaled.csv") # Chargement des données de test
y_test = pd.read_csv("data/processed/split/y_test.csv").values.ravel() # .ravel() pour convertir en array 1D necessaire pour sklearn
model = joblib.load("models/trained_model.pkl") # Chargement du modèle entraîné

# On garde seulement les colonnes numériques
X_test = X_test.select_dtypes(include=["float64", "int64"])

# Prédictions
y_pred = model.predict(X_test) 

# dictionnaire des métriques
metrics = {
    "mse": mean_squared_error(y_test, y_pred), # erreur quadratique moyenne
    "mae": mean_absolute_error(y_test, y_pred), # erreur absolue moyenne
    "r2": r2_score(y_test, y_pred) # coefficient de détermination
}

# Sauvegarde des métriques dans metrics/scores.json
fichier = open("metrics/scores.json", "w") # Ouverture du fichier en écriture et création s'il n'existe pas
json.dump(metrics, fichier, indent=4) # Écriture des métriques dans le fichier avec une indentation de 4 espaces
fichier.close() # Fermeture du fichier

# Sauvegarde du dataset avec les prédictions
df_pred = X_test.copy() # Copie des features

df_pred["y_true"] = y_test # Ajoute la colonne des vraies valeurs
df_pred["y_pred"] = y_pred # Ajoute la colonne des prédictions

df_pred.to_csv("data/predictions.csv", index=False) # Sauvegarde dans data/predictions.csv
