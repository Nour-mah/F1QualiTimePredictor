import os
import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(BASE_DIR, "../../data/processed/tabular")

# Load the data
def load_data():
    # On choisit de charger le dossier le plus recent
    all_files = [f for f in os.listdir(data_folder_path) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError("Aucune Donnée tabulaire traitée pour l'apprentissage n'a été trouvée dans le dossier 'data/processed/tabular'.")
    
    # Trier les fichiers par nom, dans l'ordre décroissant pour obtenir le plus récent en premier
    all_files.sort(reverse=True)
    latest_file = all_files[0]
    latest_file_path = os.path.join(data_folder_path, latest_file)

    df = pd.read_csv(latest_file_path)
    return df

clean_data = load_data()
# On convertit la date en datetime pour pouvoir trier correctement
clean_data['Date'] = pd.to_datetime(clean_data['Date'], format='%Y-%m-%d')


# groupby('Date') trie les groupes par date croissante : un groupe = un Grand Prix
gp_groups = [group for _, group in clean_data.groupby('Date')]

features_to_drop = ['Driver', 'Event', 'Date', 'Year', 'gap_to_pole_pct']

# Paramètres de la validation croisée temporelle (par paquets de GP)
BLOCK_SIZE = 10      # nombre de GP dans chaque bloc de test
MIN_TRAIN_GPS = 20   # taille minimale du train pour qu'un fold soit exploitable

def split_xy(groups):
    """Recolle une liste de GP en un DataFrame et sépare X / y."""
    df = pd.concat(groups, ignore_index=True)
    return df.drop(columns=features_to_drop), df['gap_to_pole_pct']

# Fenêtre "expanding" : le train de chaque fold = tous les GP antérieurs au bloc de test.
n_gps = len(gp_groups)
test_starts = list(range(n_gps - BLOCK_SIZE, MIN_TRAIN_GPS - 1, -BLOCK_SIZE))[::-1]

# Boucle de validation : un entraînement + une évaluation par fold
results = []
for fold, test_start in enumerate(test_starts, start=1):
    X_train, y_train = split_xy(gp_groups[:test_start])
    X_test, y_test = split_xy(gp_groups[test_start:test_start + BLOCK_SIZE])

    xgb = XGBRegressor(random_state=42)
    xgb.fit(X_train, y_train)
    rmse_model = root_mean_squared_error(y_test, xgb.predict(X_test))

    rmse_naive = root_mean_squared_error(y_test, np.full(len(y_test), y_train.mean()))

    results.append((rmse_model, rmse_naive))
    print(f"Fold {fold}: train={test_start} GP, test=GP {test_start + 1}-{test_start + BLOCK_SIZE} "
          f"| RMSE XGBoost={rmse_model:.4f} | RMSE naïf={rmse_naive:.4f}")


rmse_models = np.array([r[0] for r in results])
rmse_naives = np.array([r[1] for r in results])
print(f"\nXGBoost : {rmse_models.mean():.4f} ± {rmse_models.std():.4f}")
print(f"Naïf    : {rmse_naives.mean():.4f} ± {rmse_naives.std():.4f}")
