import os
import time
import fastf1

# Activation du cache
cache_dir = '../../data/cache'
if not os.path.exists(cache_dir):
	os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

# Exportation des données csv
export_dir = '../../data/raw'
if not os.path.exists(export_dir):
	os.makedirs(export_dir)


def session_loader():
    """
    Exporte toutes les sessions de FP1, FP2, FP3 et Q pour les années 2022 à 2026 (pour le format classique de weekend)
    dans des fichiers CSV. situés dans le dossier data/raw.
    """
    for year in range(2022, 2026):
        schedule = fastf1.get_event_schedule(year)
        for row in schedule.itertuples():
            event = row.EventName
            if row.EventFormat == 'conventional':
                for session_name in ['FP1', 'FP2', 'FP3', 'Q']:
                    try:
                        session = fastf1.get_session(year, event, session_name)
                        session.load(telemetry=True, weather=True, laps=True, messages=False)

                        safe_event = event.replace(" ", "_")
                        filename_base = f"{year}_{safe_event}_{session_name}"

                        laps_path = os.path.join(export_dir, f"{filename_base}_laps.csv")
                        session.laps.to_csv(laps_path, index=False)

                        weather_path = os.path.join(export_dir, f"{filename_base}_weather.csv")
                        session.weather_data.to_csv(weather_path, index=False)
                        
                        if session_name == 'Q':
                            results_path = os.path.join(export_dir, f"{filename_base}_results.csv")
                            session.results.to_csv(results_path, index=False)

                        time.sleep(1.5)
                    except fastf1.exceptions.DataNotLoadedError:
                        print("DONNÉES INDISPONIBLES (Session future ou annulée)")
                    except Exception as e:
                        print(f"ERREUR : {e}")

if __name__ == "__main__":
    session_loader()