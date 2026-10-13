import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRanker, XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
data_folder_path = os.path.join(PROJECT_ROOT, "data/processed/tabular")

GAP_TARGET = 'gap_to_pole_pct'
POSITION_TARGET = 'quali_position'

# Groupes de features disponibles dans le dataset : on active ceux qu'on veut tester via --features
FEATURE_GROUPS = {
    'gap': lambda c: c.endswith('_gap_to_best_pct'),   # écart en % au meilleur temps de chaque FP
    'compound': lambda c: c.endswith('_compound'),      # composé du meilleur tour (0=soft ... 4=wet)
    'laps': lambda c: c.endswith('_laps_ratio'),        # tours IsAccurate / max de la session
    'history': lambda c: c.startswith('driver_quali_'),  # position moyenne du pilote sur ses qualifications précédentes
}

# Paramètres de la validation croisée temporelle (par paquets de GP)
BLOCK_SIZE = 10      # nombre de GP dans chaque bloc de test
MIN_TRAIN_GPS = 20   # taille minimale du train pour qu'un fold soit exploitable

# Modèle volontairement très simple : avec ~700 lignes, les paramètres par défaut (profondeur 6) sur-apprennent
XGB_PARAMS = {'max_depth': 1, 'n_estimators': 100, 'learning_rate': 0.05, 'random_state': 42}


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


def evaluate_gap(train, test, features):
    """Régression de l'écart au pole. Renvoie (RMSE XGBoost, RMSE naïf = moyenne du train)."""
    # Les pilotes du top 10 sans temps en Q3 n'ont pas d'écart au pole
    train, test = train.dropna(subset=[GAP_TARGET]), test.dropna(subset=[GAP_TARGET])
    model = XGBRegressor(**XGB_PARAMS).fit(train[features], train[GAP_TARGET])
    rmse_model = root_mean_squared_error(test[GAP_TARGET], model.predict(test[features]))
    rmse_naive = root_mean_squared_error(test[GAP_TARGET], np.full(len(test), train[GAP_TARGET].mean()))
    return rmse_model, rmse_naive


def mean_spearman(df, score_col, ascending):
    """Spearman moyen par GP entre le classement déduit de score_col et la position réelle (1 = classement parfait)."""
    correlations = []
    for _, gp in df.groupby('gp_id'):
        predicted_rank = gp[score_col].rank(ascending=ascending)
        correlations.append(spearmanr(predicted_rank, gp[POSITION_TARGET]).statistic)
    return np.nanmean(correlations)


def evaluate_position(train, test, features):
    """Ranking des pilotes au sein de chaque GP. Renvoie (Spearman XGBRanker, Spearman référence = ordre des FP)."""
    # XGBRanker : un label plus grand = meilleur, donc on inverse la position (P1 -> 10, P10 -> 1).
    # qid indique à quel GP appartient chaque ligne (les lignes d'un même GP doivent être contiguës).
    model = XGBRanker(objective='rank:pairwise', **XGB_PARAMS)
    model.fit(train[features], 11 - train[POSITION_TARGET], qid=train['gp_id'])

    # Référence : classement de la FP3, à défaut celui de la FP2 puis de la FP1 si le pilote n'a pas roulé
    fp_reference = test[['FP3_gap_to_best_pct', 'FP2_gap_to_best_pct', 'FP1_gap_to_best_pct']].bfill(axis=1).iloc[:, 0]
    test = test.assign(score=model.predict(test[features]), fp_reference=fp_reference)
    return mean_spearman(test, 'score', ascending=False), mean_spearman(test, 'fp_reference', ascending=True)


# cible -> (fonction d'évaluation, nom de la métrique, nom du modèle, nom de la référence)
TARGETS = {
    'position': (evaluate_position, 'Spearman', 'XGBRanker', 'ordre FP'),
    'gap': (evaluate_gap, 'RMSE', 'XGBoost', 'naïf'),
}


def run_cv(df, features, target):
    """Validation croisée temporelle, fenêtre "expanding" : le train = tous les GP antérieurs au bloc de test."""
    evaluate, metric, model_name, reference_name = TARGETS[target]
    n_gps = df['gp_id'].nunique()
    test_starts = list(range(n_gps - BLOCK_SIZE, MIN_TRAIN_GPS - 1, -BLOCK_SIZE))[::-1]

    results = []
    for fold, test_start in enumerate(test_starts, start=1):
        train = df[df['gp_id'] < test_start]
        test = df[(df['gp_id'] >= test_start) & (df['gp_id'] < test_start + BLOCK_SIZE)]

        score_model, score_reference = evaluate(train, test, features)
        results.append((score_model, score_reference))
        print(f"Fold {fold}: train={test_start} GP, test=GP {test_start + 1}-{test_start + BLOCK_SIZE} "
              f"| {metric} {model_name}={score_model:.4f} | {metric} {reference_name}={score_reference:.4f}")

    results = np.array(results)
    print(f"\n{model_name} : {results[:, 0].mean():.4f} ± {results[:, 0].std():.4f}")
    print(f"{reference_name} : {results[:, 1].mean():.4f} ± {results[:, 1].std():.4f}")


def main(groups, target):
    df, data_path = load_data()
    features = select_features(df, groups)
    # gp_id = numéro du GP dans l'ordre chronologique (0 = le plus ancien)
    df = df.sort_values(['Date', POSITION_TARGET]).reset_index(drop=True)
    df['gp_id'] = df.groupby('Date').ngroup()

    print(f"Dataset : {os.path.basename(data_path)} | cible : {target} | features ({len(features)}) : {features}\n")
    run_cv(df, features, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne et évalue le modèle XGBoost par validation croisée temporelle.")
    parser.add_argument('--target', default='position', choices=sorted(TARGETS),
                        help="position : classement en qualification (ranking, Spearman) ; gap : écart au pole (régression, RMSE)")
    parser.add_argument('--features', nargs='+', default=['gap', 'history'], choices=sorted(FEATURE_GROUPS),
                        help="Groupes de features à utiliser (défaut : gap history)")
    args = parser.parse_args()
    main(args.features, args.target)
