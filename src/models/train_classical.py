import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
data_folder_path = os.path.join(PROJECT_ROOT, "data/processed/tabular")

TARGET = 'gap_to_pole_pct'

# Groupes de features disponibles dans le dataset : on active ceux qu'on veut tester via --features
FEATURE_GROUPS = {
    'gap': lambda c: c.endswith('_gap_to_best_pct'),   # écart en % au meilleur temps de chaque FP
    'compound': lambda c: c.endswith('_compound'),      # composé du meilleur tour (0=soft ... 4=wet)
    'laps': lambda c: c.endswith('_laps_ratio'),        # tours IsAccurate / max de la session
    'rain': lambda c: c.startswith('fp_rain'),          # pluie pendant les FP
    'temp_delta': lambda c: c.endswith('_temp_delta'),  # évolution des températures FP1 -> dernière FP
}

# Paramètres de la validation croisée temporelle (par paquets de GP)
BLOCK_SIZE = 10      # nombre de GP dans chaque bloc de test
MIN_TRAIN_GPS = 20   # taille minimale du train pour qu'un fold soit exploitable

XGB_PARAMS = {'random_state': 42}


def load_data():
    """Charge le dataset le plus récent (nom horodaté) et renvoie (df, chemin du fichier)."""
    all_files = sorted(f for f in os.listdir(data_folder_path) if f.endswith('.csv'))
    if not all_files:
        raise FileNotFoundError("Aucune Donnée tabulaire traitée pour l'apprentissage n'a été trouvée dans le dossier 'data/processed/tabular'.")
    latest_file_path = os.path.join(data_folder_path, all_files[-1])
    df = pd.read_csv(latest_file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    return df, latest_file_path


def select_features(df, groups):
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Groupes de features inconnus : {sorted(unknown)}. Disponibles : {sorted(FEATURE_GROUPS)}")
    return [c for c in df.columns if any(FEATURE_GROUPS[g](c) for g in groups)]


def split_xy(groups, features):
    """Recolle une liste de GP en un DataFrame et sépare X / y."""
    df = pd.concat(groups, ignore_index=True)
    return df[features], df[TARGET]


def run_cv(gp_groups, features):
    """Validation croisée temporelle, fenêtre "expanding" : le train = tous les GP antérieurs au bloc de test."""
    n_gps = len(gp_groups)
    test_starts = list(range(n_gps - BLOCK_SIZE, MIN_TRAIN_GPS - 1, -BLOCK_SIZE))[::-1]

    results = []
    for fold, test_start in enumerate(test_starts, start=1):
        X_train, y_train = split_xy(gp_groups[:test_start], features)
        X_test, y_test = split_xy(gp_groups[test_start:test_start + BLOCK_SIZE], features)

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        rmse_model = root_mean_squared_error(y_test, model.predict(X_test))
        rmse_naive = root_mean_squared_error(y_test, np.full(len(y_test), y_train.mean()))

        results.append((rmse_model, rmse_naive))
        print(f"Fold {fold}: train={test_start} GP, test=GP {test_start + 1}-{test_start + BLOCK_SIZE} "
              f"| RMSE XGBoost={rmse_model:.4f} | RMSE naïf={rmse_naive:.4f}")
    return np.array(results)


def main(groups):
    df, data_path = load_data()
    features = select_features(df, groups)
    # groupby('Date') trie les groupes par date croissante : un groupe = un Grand Prix
    gp_groups = [group for _, group in df.groupby('Date')]

    print(f"Dataset : {os.path.basename(data_path)} | features ({len(features)}) : {features}\n")
    results = run_cv(gp_groups, features)
    rmse_models, rmse_naives = results[:, 0], results[:, 1]
    print(f"\nXGBoost : {rmse_models.mean():.4f} ± {rmse_models.std():.4f}")
    print(f"Naïf    : {rmse_naives.mean():.4f} ± {rmse_naives.std():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne et évalue la baseline XGBoost (validation croisée temporelle).")
    parser.add_argument('--features', nargs='+', default=['gap'], choices=sorted(FEATURE_GROUPS),
                        help="Groupes de features à utiliser (défaut : gap)")
    main(parser.parse_args().features)
