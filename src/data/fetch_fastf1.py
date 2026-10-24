import os
import time
import datetime
import fastf1
from dotenv import dotenv_values

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Identifiants Formula 1 lus dans le fichier .env (exclu de git), sans écraser des variables déjà définies.
# Les valeurs vides sont ignorées : FastF1 demande alors les identifiants dans le terminal.
for key, value in dotenv_values(os.path.join(BASE_DIR, '../../.env')).items():
    if value and key not in os.environ:
        os.environ[key] = value

# Activation du cache
cache_dir = os.path.join(BASE_DIR, '../../data/cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Exportation des données csv
export_dir = os.path.join(BASE_DIR, '../../data/raw')
os.makedirs(export_dir, exist_ok=True)


def session_files(year, event, session_name):
    """Chemins des CSV attendus pour une session (les résultats n'existent que pour la qualification)."""
    filename_base = f"{year}_{event.replace(' ', '_')}_{session_name}"
    files = {'laps': os.path.join(export_dir, f"{filename_base}_laps.csv")}
    if session_name == 'Q':
        files['results'] = os.path.join(export_dir, f"{filename_base}_results.csv")
    return files


def session_loader():
    """
    Exporte les sessions FP1, FP2, FP3 et Q des années 2022 à 2026 (format classique de weekend)
    dans des fichiers CSV situés dans le dossier data/raw.
    Les sessions déjà téléchargées et les weekends pas encore commencés sont ignorés.
    """
    today = datetime.datetime.now()
    downloaded = 0
    for year in range(2022, 2027):
        schedule = fastf1.get_event_schedule(year)
        for row in schedule.itertuples():
            event = row.EventName
            if row.EventFormat != 'conventional':
                continue
            # EventDate = jour de la course ; les essais libres commencent 2 jours avant
            if row.EventDate - datetime.timedelta(days=2) > today:
                continue

            for session_name in ['FP1', 'FP2', 'FP3', 'Q']:
                files = session_files(year, event, session_name)
                if all(os.path.exists(path) for path in files.values()):
                    continue

                try:
                    session = fastf1.get_session(year, event, session_name)
                    # Seuls les tours (et les résultats de la qualif) servent au projet : ni météo ni télémétrie,
                    # qui ralentissaient fortement la collecte et faisaient grossir le cache
                    session.load(telemetry=False, weather=False, laps=True, messages=False)

                    # Sans télémétrie, FastF1 ne calcule pas LapStartDate : on garde l'heure de début de la séance (UTC)
                    # pour que build_features_tabular.get_gp_date puisse dater le GP
                    session.laps.assign(SessionDate=session.date).to_csv(files['laps'], index=False)
                    if session_name == 'Q':
                        session.results.to_csv(files['results'], index=False)

                    downloaded += 1
                    print(f"TÉLÉCHARGÉ : {year} {event} {session_name}")
                    time.sleep(1.5)
                except fastf1.exceptions.DataNotLoadedError:
                    print(f"DONNÉES INDISPONIBLES : {year} {event} {session_name} (session pas encore disputée ou annulée)")
                except Exception as e:
                    print(f"ERREUR : {year} {event} {session_name} : {e}")

    print(f"\n{downloaded} session(s) téléchargée(s).")


if __name__ == "__main__":
    session_loader()
