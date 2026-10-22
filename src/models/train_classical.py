import os
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from scipy.stats import spearmanr
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRanker, XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
data_folder_path = os.path.join(PROJECT_ROOT, "data/processed/tabular")

# MLflow : paramètres et métriques dans une base SQLite, fichiers (modèles, JSON) dans mlartifacts/
MLFLOW_TRACKING_URI = "sqlite:///" + os.path.join(PROJECT_ROOT, "mlflow.db")
MLFLOW_ARTIFACT_LOCATION = "file:" + os.path.join(PROJECT_ROOT, "mlartifacts")
MLFLOW_EXPERIMENT = "f1-quali"
# Alias du registre de modèles désignant la version utilisée par predict.py ; on le déplace explicitement avec --promote
CHAMPION_ALIAS = "champion"

GAP_TARGET = 'gap_to_pole_pct'
POSITION_TARGET = 'quali_position'

# Groupes de features disponibles dans le dataset : on active ceux qu'on veut tester via --features
FEATURE_GROUPS = {
    'gap': lambda c: c.endswith('_gap_to_best_pct'),   # écart en % au meilleur temps de chaque FP
    'compound': lambda c: c.endswith('_compound'),      # composé du meilleur tour (0=soft ... 4=wet)
    'laps': lambda c: c.endswith('_laps_ratio'),        # tours IsAccurate / max de la session
    'history': lambda c: c.startswith('driver_quali_'),  # position moyenne du pilote sur ses qualifications précédentes
    'team': lambda c: c.startswith('team_quali_'),       # position moyenne des pilotes de l'équipe sur les GP précédents
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


def fit_regressor(train, features):
    """Régression de l'écart au pole (les pilotes du top 10 sans temps en Q3 n'ont pas d'écart, donc sont exclus)."""
    train = train.dropna(subset=[GAP_TARGET])
    return XGBRegressor(**XGB_PARAMS).fit(train[features], train[GAP_TARGET])


def fit_ranker(train, features):
    """Ranking des pilotes au sein de chaque GP."""
    # XGBRanker : un label plus grand = meilleur, donc on inverse la position (P1 -> 10, P10 -> 1).
    # qid indique à quel GP appartient chaque ligne (les lignes d'un même GP doivent être contiguës).
    model = XGBRanker(objective='rank:pairwise', **XGB_PARAMS)
    return model.fit(train[features], 11 - train[POSITION_TARGET], qid=train['gp_id'])


def evaluate_gap(train, test, features):
    """Renvoie (RMSE XGBoost, RMSE naïf = moyenne du train)."""
    model = fit_regressor(train, features)
    test = test.dropna(subset=[GAP_TARGET])
    rmse_model = root_mean_squared_error(test[GAP_TARGET], model.predict(test[features]))
    rmse_naive = root_mean_squared_error(test[GAP_TARGET], np.full(len(test), train[GAP_TARGET].dropna().mean()))
    return rmse_model, rmse_naive


def mean_spearman(df, score_col, ascending):
    """Spearman moyen par GP entre le classement déduit de score_col et la position réelle (1 = classement parfait)."""
    correlations = []
    for _, gp in df.groupby('gp_id'):
        predicted_rank = gp[score_col].rank(ascending=ascending)
        correlations.append(spearmanr(predicted_rank, gp[POSITION_TARGET]).statistic)
    return np.nanmean(correlations)


def evaluate_position(train, test, features):
    """Renvoie (Spearman XGBRanker, Spearman référence = ordre des FP)."""
    model = fit_ranker(train, features)

    # Référence : classement de la FP3, à défaut celui de la FP2 puis de la FP1 si le pilote n'a pas roulé
    fp_reference = test[['FP3_gap_to_best_pct', 'FP2_gap_to_best_pct', 'FP1_gap_to_best_pct']].bfill(axis=1).iloc[:, 0]
    test = test.assign(score=model.predict(test[features]), fp_reference=fp_reference)
    return mean_spearman(test, 'score', ascending=False), mean_spearman(test, 'fp_reference', ascending=True)


TARGETS = {
    'position': {'fit': fit_ranker, 'evaluate': evaluate_position, 'metric': 'spearman', 'higher_is_better': True,
                 'model_name': 'XGBRanker', 'reference_name': 'ordre FP', 'registered_model': 'f1-quali-ranker'},
    'gap': {'fit': fit_regressor, 'evaluate': evaluate_gap, 'metric': 'rmse', 'higher_is_better': False,
            'model_name': 'XGBoost', 'reference_name': 'naïf', 'registered_model': 'f1-quali-gap-regressor'},
}


def temporal_folds(n_gps):
    """Premiers gp_id des blocs de test, du plus ancien au plus récent. Le dernier bloc finit au dernier GP,
    et chaque bloc s'entraîne sur au moins MIN_TRAIN_GPS GP antérieurs."""
    return list(range(n_gps - BLOCK_SIZE, MIN_TRAIN_GPS - 1, -BLOCK_SIZE))[::-1]


def run_cv(df, features, target):
    """Validation croisée temporelle, fenêtre "expanding" : le train = tous les GP antérieurs au bloc de test."""
    config = TARGETS[target]
    metric, model_name, reference_name = config['metric'], config['model_name'], config['reference_name']

    results = []
    for fold, test_start in enumerate(temporal_folds(df['gp_id'].nunique()), start=1):
        train = df[df['gp_id'] < test_start]
        test = df[(df['gp_id'] >= test_start) & (df['gp_id'] < test_start + BLOCK_SIZE)]

        score_model, score_reference = config['evaluate'](train, test, features)
        results.append((score_model, score_reference))
        mlflow.log_metric(f"fold_{metric}_model", score_model, step=fold)
        mlflow.log_metric(f"fold_{metric}_reference", score_reference, step=fold)
        print(f"Fold {fold}: train={test_start} GP, test=GP {test_start + 1}-{test_start + BLOCK_SIZE} "
              f"| {metric} {model_name}={score_model:.4f} | {metric} {reference_name}={score_reference:.4f}")

    results = np.array(results)
    model_mean, reference_mean = results[:, 0].mean(), results[:, 1].mean()
    # Gain positif = le modèle fait mieux que la référence, quel que soit le sens de la métrique
    gain = model_mean - reference_mean if config['higher_is_better'] else reference_mean - model_mean
    mlflow.log_metrics({
        f"cv_{metric}_model_mean": model_mean,
        f"cv_{metric}_model_std": results[:, 0].std(),
        f"cv_{metric}_reference_mean": reference_mean,
        f"cv_{metric}_reference_std": results[:, 1].std(),
        "cv_gain_vs_reference": gain,
    })
    print(f"\n{model_name} : {model_mean:.4f} ± {results[:, 0].std():.4f}")
    print(f"{reference_name} : {reference_mean:.4f} ± {results[:, 1].std():.4f}")
    print(f"Gain par rapport à la référence : {gain:+.4f}")


def main(groups, target, promote):
    df, data_path = load_data()
    features = select_features(df, groups)
    # gp_id = numéro du GP dans l'ordre chronologique (0 = le plus ancien)
    df = df.sort_values(['Date', POSITION_TARGET]).reset_index(drop=True)
    df['gp_id'] = df.groupby('Date').ngroup()

    print(f"Dataset : {os.path.basename(data_path)} | cible : {target} | features ({len(features)}) : {features}\n")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Emplacement des artefacts fixé explicitement, sinon MLflow le place par rapport au dossier d'exécution
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACT_LOCATION)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"{target}:{'+'.join(groups)}") as run:
        mlflow.log_params({
            'target': target,
            'feature_groups': ",".join(groups),
            'dataset': os.path.basename(data_path),
            'n_gps': df['gp_id'].nunique(),
            'n_rows': len(df),
            # Dernier GP vu par le modèle final : predict.py ne l'utilise que pour des GP postérieurs (sinon fuite)
            'train_end_date': df['Date'].max().strftime('%Y-%m-%d'),
            'block_size': BLOCK_SIZE,
            'min_train_gps': MIN_TRAIN_GPS,
            **XGB_PARAMS,
        })
        mlflow.log_dict({'features': features}, "features.json")

        run_cv(df, features, target)

        # Modèle final entraîné sur tous les GP : celui qu'un script de prédiction ou une API pourra recharger
        final_model = TARGETS[target]['fit'](df, features)
        mlflow.log_param('objective', final_model.get_xgb_params()['objective'])
        # Chaque run ajoute une version au registre de modèles ; seul l'alias "champion" décide de celle utilisée en prédiction
        registered_model = TARGETS[target]['registered_model']
        model_info = mlflow.xgboost.log_model(final_model, name="model", input_example=df[features].head(5),
                                              registered_model_name=registered_model)
        version = model_info.registered_model_version
        print(f"\nRun MLflow enregistré : {run.info.run_id} | modèle {registered_model} version {version}")

        if promote:
            mlflow.MlflowClient().set_registered_model_alias(registered_model, CHAMPION_ALIAS, version)
            print(f"Version {version} promue : alias '{CHAMPION_ALIAS}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne et évalue le modèle XGBoost par validation croisée temporelle.")
    parser.add_argument('--target', default='position', choices=sorted(TARGETS),
                        help="position : classement en qualification (ranking, Spearman) ; gap : écart au pole (régression, RMSE)")
    parser.add_argument('--features', nargs='+', default=['gap', 'history'], choices=sorted(FEATURE_GROUPS),
                        help="Groupes de features à utiliser (défaut : gap history)")
    parser.add_argument('--promote', action='store_true',
                        help=f"Désigne le modèle de ce run comme '{CHAMPION_ALIAS}', celui que predict.py charge")
    args = parser.parse_args()
    main(args.features, args.target, args.promote)
