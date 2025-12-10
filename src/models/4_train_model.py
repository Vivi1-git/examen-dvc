import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Entraînement du modèle. 
# En utilisant les paramètres retrouvés à travers le GridSearch, on entraînera le modèle en sauvegardant le modèle entraîné dans le dossier models.

# Chargement des données et des meilleurs paramètres
X_train = pd.read_csv("data/processed/norm/X_train_scaled.csv") # Chargement des données d'entraînement
y_train = pd.read_csv("data/processed/split/y_train.csv").values.ravel() # .ravel() pour convertir en array 1D necessaire pour sklearn
best_params = joblib.load("models/best_params.pkl") # Chargement des meilleurs paramètres

# On garde seulement les colonnes numériques
X_train = X_train.select_dtypes(include=["float64", "int64"])

# Créeation et entraînement du modèle
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde du modèle entraîné dans le dossier models
joblib.dump(model, "models/trained_model.pkl")
