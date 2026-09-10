import os
import re
import pandas as pd
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(BASE_DIR, "../../data/raw")
processed_dir = os.path.join(BASE_DIR, "../../data/processed/tabular")
os.makedirs(processed_dir, exist_ok=True)

def load_session_laps(year, gp_name, session_name):
    safe_event = gp_name.replace(" ", "_")
    filename = f"{year}_{safe_event}_{session_name}_laps.csv"
    path = os.path.join(raw_dir, filename)
    
    if not os.path.exists(path):
        return None
    
    df = pd.read_csv(path)
    df = df[df["IsAccurate"]]

    df['LapTime'] = pd.to_timedelta(df['LapTime'])
    df = df.groupby(['Driver']).agg(BestTime = ('LapTime', "min")).rename(columns={'BestTime': f'{session_name}_best_time'})

    return df

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

    # On conserve les pilotes ayant au moins deux séances d'essais libres valides
    df_practice = df_practice[df_practice.isnull().sum(axis=1) < 2]

    # Imputer les sessions manquantes par la moyenne des sessions disponibles
    df_practice = df_practice.fillna(df_practice.mean(numeric_only=True))

    # Charger le CSV de résultats de quali
    safe_event = gp_name.replace(" ", "_")
    results_filename = f"{year}_{safe_event}_Q_results.csv"
    results_path = os.path.join(raw_dir, results_filename)

    if not os.path.exists(results_path):
        return None

    df_results = pd.read_csv(results_path)

    # Récupérer le temps de pole (position 1, colonne Q3)
    pole_time = pd.to_timedelta(df_results[df_results['Position'] == 1]['Q3'].iloc[0])
    
    # Ne garder que les pilotes avec un temps en Q3
    df_q3 = df_results[['Abbreviation', 'Q3']].dropna(subset=['Q3'])
    df_q3['Q3'] = pd.to_timedelta(df_q3['Q3'])

    # Calculer l'écart en % au temps de pole
    df_q3['gap_to_pole_pct'] = (df_q3['Q3'] - pole_time) / pole_time * 100

    # Merger avec les features EL sur la clé Driver
    df_q3 = df_q3.rename(columns={'Abbreviation': 'Driver'})
    df_final = df_practice.merge(df_q3[['Driver', 'gap_to_pole_pct']], on='Driver', how='inner')

    # Trier par gap_to_pole_pct et ne garder que les 10 pilotes Q3
    df_final = df_final.sort_values('gap_to_pole_pct').head(10)

    return df_final

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


def build_dataset():
    datasets = []
    for year, gp_name in discover_gps():
        df = build_gp_features(year, gp_name)
        if df is None:
            print(f"\n Skipping {year} {gp_name} due to insufficient data.")
            continue
        df['Year'] = year
        df['Event'] = gp_name
        df['Date'] = get_gp_date(year, gp_name)
        datasets.append(df)
        print(f"\n Processed {year} {gp_name} with {len(df)} drivers.")

    DS = pd.concat(datasets, ignore_index=True)
    DS.sort_values('Date', inplace=True)
    DS.to_csv(os.path.join(processed_dir, f"F1_features_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
    print(f"\n ===== Dataset built with {len(DS)} rows across {len(datasets)} Grand Prix and saved to {processed_dir}. =====")


if __name__ == "__main__":
    build_dataset()