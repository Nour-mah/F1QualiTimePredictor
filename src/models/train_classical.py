import os
import pandas as pd
import numpy as np
import datetime
import sklearn as sk


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
# On trie les données par date décroissante, et par gap_to_pole_pct croissant pour avoir les meilleurs temps en premier
clean_data = clean_data.sort_values(by=['Date', "gap_to_pole_pct"], ascending=[False, True])
# On ne garde que les 10 derniers Grand Prix (10 dates les plus récentes)
Test = pd.concat(clean_data.groupby('Date')[10], ignore_index=True)
Train = pd.concat(clean_data.groupby('Date')[10:], ignore_index=True)

X_train = Train.drop(columns=['Driver', 'Event', 'Date', 'gap_to_pole_pct', "Year"])
y_train = Train['gap_to_pole_pct', 'Year']
X_test = Test.drop(columns=['Driver', 'Event', 'Date', 'gap_to_pole_pct', "Year"])
y_test = Test['gap_to_pole_pct', 'Year']

