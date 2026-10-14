import os
import sys
import argparse
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRanker

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, SRC_DIR)

from data.build_features_tabular import build_fp_features, current_driver_form, get_gp_date, raw_dir
from models.train_classical import FEATURE_GROUPS, POSITION_TARGET, PROJECT_ROOT, XGB_PARAMS, load_data, select_features

predictions_dir = os.path.join(PROJECT_ROOT, "data/predictions")


def predict_gp(year, gp_name, groups):
    """Prédit le classement de la qualification d'un GP à partir de ses essais libres (à lancer après fetch_fastf1.py)."""
    gp_date = get_gp_date(year, gp_name)
    new_gp = build_fp_features(year, gp_name)
    if gp_date is None or new_gp is None:
        sys.exit(f"Pas assez d'essais libres dans data/raw pour {year} {gp_name} (au moins 2 séances) : lancer d'abord src/data/fetch_fastf1.py.")
    new_gp = new_gp.merge(current_driver_form(gp_date), on='Driver', how='left')

    # Entraînement sur les GP strictement antérieurs : on peut ainsi rejouer un GP passé sans fuite
    df, data_path = load_data()
    features = select_features(df, groups)
    train = df[df['Date'] < pd.Timestamp(gp_date)].sort_values(['Date', POSITION_TARGET]).reset_index(drop=True)
    train['gp_id'] = train.groupby('Date').ngroup()

    model = XGBRanker(objective='rank:pairwise', **XGB_PARAMS)
    model.fit(train[features], 11 - train[POSITION_TARGET], qid=train['gp_id'])

    # reindex : une séance annulée donne une colonne absente, remplacée par NaN
    new_gp['score'] = model.predict(new_gp.reindex(columns=features))
    new_gp['predicted_position'] = new_gp['score'].rank(ascending=False, method='first').astype(int)
    new_gp = new_gp.sort_values('predicted_position')

    print(f"Prédiction : {year} {gp_name} ({gp_date})")
    print(f"Modèle entraîné sur {train['gp_id'].nunique()} GP ({os.path.basename(data_path)}), features : {features}\n")

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
                        help="Groupes de features à utiliser (défaut : gap history)")
    args = parser.parse_args()
    predict_gp(args.year, args.gp, args.features)
