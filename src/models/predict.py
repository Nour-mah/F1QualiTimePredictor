import os
import sys
import argparse
import mlflow
import mlflow.xgboost
import pandas as pd
from scipy.stats import spearmanr

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, SRC_DIR)

from data.build_features_tabular import build_fp_features, current_driver_form, current_team_form, get_gp_date, raw_dir
from models.train_classical import (CHAMPION_ALIAS, FEATURE_GROUPS, MLFLOW_TRACKING_URI, POSITION_TARGET, PROJECT_ROOT,
                                    TARGETS, fit_ranker, load_data, select_features)

predictions_dir = os.path.join(PROJECT_ROOT, "data/predictions")
REGISTERED_MODEL = TARGETS['position']['registered_model']


def load_champion():
    """Renvoie (modèle, features, date du dernier GP d'entraînement, description) du modèle 'champion', ou None s'il n'existe pas."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None
    run = client.get_run(version.run_id)
    model = mlflow.xgboost.load_model(f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}")
    # Le modèle attend exactement les features avec lesquelles il a été entraîné
    features = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/features.json")['features']
    description = f"{REGISTERED_MODEL} version {version.version} ('{CHAMPION_ALIAS}', run {run.info.run_name})"
    return model, features, run.data.params['train_end_date'], description


def train_before(gp_date, features):
    """Entraîne le ranker sur les GP strictement antérieurs à gp_date (rejeu d'un GP passé sans fuite)."""
    df, data_path = load_data()
    train = df[df['Date'] < pd.Timestamp(gp_date)].sort_values(['Date', POSITION_TARGET]).reset_index(drop=True)
    train['gp_id'] = train.groupby('Date').ngroup()
    description = f"réentraîné à la volée sur les {train['gp_id'].nunique()} GP antérieurs ({os.path.basename(data_path)})"
    return fit_ranker(train, features), description


def get_model(gp_date, default_groups):
    """Modèle 'champion' si le GP est postérieur à ses données d'entraînement, sinon réentraînement sans fuite."""
    champion = load_champion()
    if champion is None:
        df, _ = load_data()
        features = select_features(df, default_groups)
        print(f"Aucun modèle '{CHAMPION_ALIAS}' dans le registre (lancer train_classical.py --promote) : réentraînement.")
        return (*train_before(gp_date, features), features)

    model, features, train_end_date, description = champion
    if gp_date > train_end_date:
        return model, description, features
    # Le champion a déjà vu ce GP : l'utiliser fausserait la comparaison avec le résultat réel
    print(f"Le modèle '{CHAMPION_ALIAS}' a été entraîné jusqu'au {train_end_date}, ce GP en fait partie : réentraînement.")
    return (*train_before(gp_date, features), features)


def predict_gp(year, gp_name, default_groups):
    """Prédit le classement de la qualification d'un GP à partir de ses essais libres (à lancer après fetch_fastf1.py)."""
    gp_date = get_gp_date(year, gp_name)
    new_gp = build_fp_features(year, gp_name)
    if gp_date is None or new_gp is None:
        sys.exit(f"Pas assez d'essais libres dans data/raw pour {year} {gp_name} (au moins 2 séances) : lancer d'abord src/data/fetch_fastf1.py.")
    new_gp = new_gp.merge(current_driver_form(gp_date), on='Driver', how='left')
    new_gp = new_gp.merge(current_team_form(gp_date), on='Team', how='left')

    model, model_description, features = get_model(gp_date, default_groups)

    # reindex : une séance annulée donne une colonne absente, remplacée par NaN
    new_gp['score'] = model.predict(new_gp.reindex(columns=features))
    new_gp['predicted_position'] = new_gp['score'].rank(ascending=False, method='first').astype(int)
    new_gp = new_gp.sort_values('predicted_position')

    print(f"\nPrédiction : {year} {gp_name} ({gp_date})")
    print(f"Modèle : {model_description}")
    print(f"Features : {features}\n")

    columns = ['predicted_position', 'Driver', 'score'] + features
    results_path = os.path.join(raw_dir, f"{year}_{gp_name.replace(' ', '_')}_Q_results.csv")
    if os.path.exists(results_path):
        # GP déjà couru : on compare au classement réel
        actual = pd.read_csv(results_path)[['Abbreviation', 'Position']].drop_duplicates(subset='Position', keep='first')
        actual = actual.rename(columns={'Abbreviation': 'Driver', 'Position': 'actual_position'})
        new_gp = new_gp.merge(actual, on='Driver', how='left')
        columns.insert(2, 'actual_position')

    top10 = new_gp.head(10)
    print(top10[columns].round(3).to_string(index=False))

    if 'actual_position' in new_gp:
        in_real_top10 = (top10['actual_position'] <= 10).sum()
        correlation = spearmanr(top10['predicted_position'], top10['actual_position'], nan_policy='omit').statistic
        print(f"\n{in_real_top10}/10 pilotes prédits sont dans le vrai top 10 | Spearman sur le top 10 prédit : {correlation:.3f}")

    os.makedirs(predictions_dir, exist_ok=True)
    output_path = os.path.join(predictions_dir, f"{year}_{gp_name.replace(' ', '_')}_prediction.csv")
    new_gp[columns].to_csv(output_path, index=False)
    print(f"\nPrédiction complète (toute la grille) enregistrée dans {os.path.relpath(output_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédit le top 10 de la qualification d'un Grand Prix à partir de ses essais libres.")
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--gp', required=True, help="Nom du Grand Prix tel que dans FastF1, ex. 'United States Grand Prix'")
    parser.add_argument('--features', nargs='+', default=['gap', 'history'], choices=sorted(FEATURE_GROUPS),
                        help=f"Groupes de features, utilisés seulement s'il n'existe pas de modèle '{CHAMPION_ALIAS}' (défaut : gap history)")
    args = parser.parse_args()
    predict_gp(args.year, args.gp, args.features)
