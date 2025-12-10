import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# GridSearch des meilleurs paramètres à utiliser pour la modélisation. 
# Vous déciderez le modèle de regression à implémenter et des paramètres à tester. 
# À l'issue de ce script vous aurez les meilleurs paramètres sous forme de fichier .pkl que vous sauvegarderez dans le dossier models.

# Chargement des données normalisées
X_train = pd.read_csv("data/processed/norm/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/split/y_train.csv").values.ravel()

# On garde seulement les colonnes numériques
X_train = X_train.select_dtypes(include=["float64", "int64"])

# Définition du modèle
rf = RandomForestRegressor(random_state=42)

# Grille d'hyperparamètres
param_grid = {
    "n_estimators": [100, 200, 300], # nombre d'arbres dans la forêt
    "max_depth": [None, 10, 20, 30], # profondeur maximale des arbres
    "min_samples_split": [2, 5, 10] # nombre minimum d'échantillons requis pour diviser un nœud interne
}

# GridSearch
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Sauvegarde des meilleurs paramètres dans le dossier models sous forme de fichier .pkl
joblib.dump(grid_search.best_params_, "models/best_params.pkl")
