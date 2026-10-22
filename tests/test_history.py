"""Forme récente du pilote : le test le plus important du projet.

Une fuite (utiliser le résultat du GP qu'on prédit) gonflerait les scores sans aucune erreur visible :
seul un test peut la détecter.
"""
import math

from data.build_features_tabular import build_driver_history, current_driver_form, discover_gps, get_gp_date


def three_gps(raw):
    """Une saison 2024 de 3 GP : le pilote VER finit 1er, 3e puis 5e ; une saison 2025 de 1 GP."""
    raw.quali(2024, 'Bahrain Grand Prix', [{'Driver': 'VER', 'Position': 1}, {'Driver': 'LEC', 'Position': 2}], date="2024-03-02")
    raw.quali(2024, 'Saudi Arabian Grand Prix', [{'Driver': 'VER', 'Position': 3}, {'Driver': 'LEC', 'Position': 1}], date="2024-03-09")
    raw.quali(2024, 'Australian Grand Prix', [{'Driver': 'VER', 'Position': 5}, {'Driver': 'LEC', 'Position': 4}], date="2024-03-23")
    raw.quali(2025, 'Bahrain Grand Prix', [{'Driver': 'VER', 'Position': 2}, {'Driver': 'LEC', 'Position': 1}], date="2025-03-01")
    return {gp: get_gp_date(*gp) for gp in discover_gps()}


def form_of(history, driver, date):
    return history.loc[(history['Driver'] == driver) & (history['Date'] == date), 'driver_quali_pos_avg'].iloc[0]


def test_history_only_uses_previous_gps(raw):
    history = build_driver_history(three_gps(raw))

    # Premier GP : aucun GP antérieur, donc pas de forme (et surtout pas sa propre position, 1)
    assert math.isnan(form_of(history, 'VER', '2024-03-02'))
    # Deuxième GP : uniquement le premier (1), pas le deuxième lui-même (3)
    assert form_of(history, 'VER', '2024-03-09') == 1
    # Troisième GP : moyenne des deux premiers (1 et 3), sans le troisième (5)
    assert form_of(history, 'VER', '2024-03-23') == 2


def test_history_resets_each_season(raw):
    history = build_driver_history(three_gps(raw))

    # Premier GP de 2025 : la saison 2024 ne compte plus
    assert math.isnan(form_of(history, 'VER', '2025-03-01'))


def test_prediction_form_matches_training_form(raw):
    # La forme calculée pour prédire un GP doit être identique à celle vue par le modèle à l'entraînement,
    # sinon le modèle reçoit en production des valeurs qu'il n'a jamais apprises (training/serving skew)
    history = build_driver_history(three_gps(raw))
    form = current_driver_form('2024-03-23').set_index('Driver')['driver_quali_pos_avg']

    assert form['VER'] == form_of(history, 'VER', '2024-03-23')
    assert form['LEC'] == form_of(history, 'LEC', '2024-03-23')
