import os
import re
import pandas as pd
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(BASE_DIR, "../../data/raw")
processed_dir = os.path.join(BASE_DIR, "../../data/processed/tabular")
os.makedirs(processed_dir, exist_ok=True)

HISTORY_WINDOW = 5  # nombre de qualifications précédentes prises en compte dans l'historique du pilote
COMPOUND_MAP = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}

def load_session_laps(year, gp_name, session_name):
    safe_event = gp_name.replace(" ", "_")
    filename = f"{year}_{safe_event}_{session_name}_laps.csv"
    path = os.path.join(raw_dir, filename)
    
    if not os.path.exists(path):
        return None
    
    df = pd.read_csv(path)
    df = df[df["IsAccurate"]]

    df['LapTime'] = pd.to_timedelta(df['LapTime'])

    # Ligne du meilleur tour de chaque pilote (idxmin renvoie l'index de la ligne, pas seulement la valeur)
    best_laps = df.loc[df.groupby('Driver')['LapTime'].idxmin()].set_index('Driver')
    n_laps = df.groupby('Driver').size()

    out = pd.DataFrame({
        f'{session_name}_gap_to_best_pct': best_laps['LapTime'],
        f'{session_name}_compound': best_laps['Compound'].map(COMPOUND_MAP),
        f'{session_name}_laps_ratio': n_laps / n_laps.max(),
    })

    # Ecart en % par rapport au meilleur temps de la session (tous pilotes confondus)
    best_time = out[f'{session_name}_gap_to_best_pct'].min()
    out[f'{session_name}_gap_to_best_pct'] = (out[f'{session_name}_gap_to_best_pct'] - best_time) / best_time * 100

    return out

def build_gp_features(year, gp_name):
    sessions = {}
    for session_name in ['FP1', 'FP2', 'FP3']:
        sessions[session_name] = load_session_laps(year, gp_name, session_name)

    # Protection contre les sessions None
    valid_sessions = [df for df in sessions.values() if df is not None]
    if len(valid_sessions) < 2:
        return None

    # Merge des sessions valides sur la clé "Driver"
    df_practice = valid_sessions[0]
    for df in valid_sessions[1:]:
        df_practice = df_practice.merge(df, on='Driver', how='outer')

    # On conserve les pilotes ayant au moins deux séances d'essais libres valides (colonnes d'écart uniquement)
    # Les valeurs manquantes restent NaN (XGBoost les gère nativement)
    gap_cols = [c for c in df_practice.columns if c.endswith('_gap_to_best_pct')]
    df_practice = df_practice[df_practice[gap_cols].notna().sum(axis=1) >= 2]

    # Charger le CSV de résultats de quali
    safe_event = gp_name.replace(" ", "_")
    results_filename = f"{year}_{safe_event}_Q_results.csv"
    results_path = os.path.join(raw_dir, results_filename)

    if not os.path.exists(results_path):
        return None

    df_results = pd.read_csv(results_path)

    # Top 10 officiel de la qualification, y compris les pilotes sans temps en Q3.
    # Position en double (pilote disqualifié après coup, ex. Pays-Bas 2024) : on garde le premier listé, le classé officiel.
    df_q3 = df_results[df_results['Position'] <= 10].drop_duplicates(subset='Position', keep='first')
    df_q3 = df_q3[['Abbreviation', 'Position', 'Q3']].rename(columns={'Abbreviation': 'Driver', 'Position': 'quali_position'})
    df_q3['quali_position'] = df_q3['quali_position'].astype(int)

    # Ecart en % au temps de pole (NaN pour les pilotes du top 10 sans temps en Q3)
    df_q3['Q3'] = pd.to_timedelta(df_q3['Q3'])
    pole_time = df_q3.loc[df_q3['quali_position'] == 1, 'Q3'].iloc[0]
    df_q3['gap_to_pole_pct'] = (df_q3['Q3'] - pole_time) / pole_time * 100

    # Merger avec les features EL sur la clé Driver
    df_final = df_practice.merge(df_q3[['Driver', 'quali_position', 'gap_to_pole_pct']], on='Driver', how='inner')

    return df_final.sort_values('quali_position')

def discover_gps():
    """Liste les (year, gp_name) disponibles en scannant les résultats Q déjà enregistrés dans raw_dir."""
    gps = []
    pattern = re.compile(r"^(\d{4})_(.+)_Q_results\.csv$")
    for filename in sorted(os.listdir(raw_dir)):
        match = pattern.match(filename)
        if match is None:
            continue
        year = int(match.group(1))
        gp_name = match.group(2).replace("_", " ")
        gps.append((year, gp_name))
    return gps

def get_gp_date(year, gp_name):
    # Charger le CSV de résultats de quali
    safe_event = gp_name.replace(" ", "_")
    results_filename = f"{year}_{safe_event}_Q_laps.csv"
    results_path = os.path.join(raw_dir, results_filename)

    if not os.path.exists(results_path):
        return None

    df_results = pd.read_csv(results_path)

    # Récupérer la date de l'événement
    if 'LapStartDate' in df_results.columns:
        timestamp = df_results['LapStartDate'].iloc[0]
        return timestamp.split(' ')[0]  # Retourne uniquement la date
    else:
        return None


def build_driver_history(gp_dates):
    """Position moyenne de chaque pilote sur ses HISTORY_WINDOW qualifications précédentes (tous les pilotes, pas seulement le top 10)."""
    all_results = []
    for (year, gp_name), date in gp_dates.items():
        results = pd.read_csv(os.path.join(raw_dir, f"{year}_{gp_name.replace(' ', '_')}_Q_results.csv"))
        results = results[['Abbreviation', 'Position']].rename(columns={'Abbreviation': 'Driver'})
        results['Date'] = date
        all_results.append(results)
    history = pd.concat(all_results, ignore_index=True).sort_values('Date')

    # shift(1) : uniquement les GP strictement antérieurs, sinon la position à prédire fuit dans la feature
    history['driver_quali_pos_avg'] = history.groupby('Driver')['Position'].transform(
        lambda positions: positions.shift(1).rolling(HISTORY_WINDOW, min_periods=1).mean())
    return history[['Driver', 'Date', 'driver_quali_pos_avg']]


def build_dataset():
    gp_dates = {gp: get_gp_date(*gp) for gp in discover_gps()}

    datasets = []
    for (year, gp_name), date in gp_dates.items():
        df = build_gp_features(year, gp_name)
        if df is None:
            print(f"\n Skipping {year} {gp_name} due to insufficient data.")
            continue
        df['Year'] = year
        df['Event'] = gp_name
        df['Date'] = date
        datasets.append(df)
        print(f"\n Processed {year} {gp_name} with {len(df)} drivers.")

    DS = pd.concat(datasets, ignore_index=True)
    # Historique du pilote (NaN pour sa toute première qualification du dataset)
    DS = DS.merge(build_driver_history(gp_dates), on=['Driver', 'Date'], how='left')
    DS.sort_values('Date', inplace=True)
    DS.to_csv(os.path.join(processed_dir, f"F1_features_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
    print(f"\n ===== Dataset built with {len(DS)} rows across {len(datasets)} Grand Prix and saved to {processed_dir}. =====")


if __name__ == "__main__":
    build_dataset()