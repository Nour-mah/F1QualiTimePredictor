"""Features des essais libres et cibles de la qualification."""
import math

import pytest

from data.build_features_tabular import build_fp_features, build_gp_features, load_session_laps

GP = (2024, 'Bahrain Grand Prix')

def test_deleted_laps_are_ignored(raw):
    # Cas réel d'Austin 2026 en FP1 : le meilleur tour de VER a été annulé pour limites de piste
    raw.laps(*GP, 'FP1', [
        {'Driver': 'VER', 'seconds': 88.0, 'deleted': True},
        {'Driver': 'VER', 'seconds': 91.8},
        {'Driver': 'LEC', 'seconds': 90.0},
    ])

    session = load_session_laps(*GP, 'FP1')

    assert session.loc['LEC', 'FP1_gap_to_best_pct'] == 0
    assert session.loc['VER', 'FP1_gap_to_best_pct'] == pytest.approx(2.0)

def test_gap_is_relative_to_session_best(raw):
    raw.laps(*GP, 'FP1', [{'Driver': 'VER', 'seconds': 90.0}, {'Driver': 'LEC', 'seconds': 91.8}])

    session = load_session_laps(*GP, 'FP1')

    assert session.loc['VER', 'FP1_gap_to_best_pct'] == 0
    assert session.loc['LEC', 'FP1_gap_to_best_pct'] == pytest.approx(2.0)   # 1,8 s sur 90 s = 2 %


def test_inaccurate_laps_are_ignored(raw):
    raw.laps(*GP, 'FP1', [
        {'Driver': 'VER', 'seconds': 90.0},
        {'Driver': 'LEC', 'seconds': 85.0, 'accurate': False},   # tour non représentatif (coupe de virage...)
        {'Driver': 'LEC', 'seconds': 91.8},
    ])

    session = load_session_laps(*GP, 'FP1')

    assert session.loc['VER', 'FP1_gap_to_best_pct'] == 0
    assert session.loc['LEC', 'FP1_gap_to_best_pct'] == pytest.approx(2.0)


def test_compound_and_laps_ratio(raw):
    raw.laps(*GP, 'FP1', [
        {'Driver': 'VER', 'seconds': 92.0, 'compound': 'HARD'},
        {'Driver': 'VER', 'seconds': 90.0, 'compound': 'SOFT'},
        {'Driver': 'LEC', 'seconds': 91.0, 'compound': 'MEDIUM'},
    ])

    session = load_session_laps(*GP, 'FP1')

    # Pneu du meilleur tour, pas du premier tour
    assert session.loc['VER', 'FP1_compound'] == 0
    assert session.loc['LEC', 'FP1_compound'] == 1
    # 2 tours pour VER (le maximum), 1 pour LEC
    assert session.loc['VER', 'FP1_laps_ratio'] == 1
    assert session.loc['LEC', 'FP1_laps_ratio'] == 0.5


def test_driver_needs_two_valid_sessions(raw):
    raw.laps(*GP, 'FP1', [{'Driver': 'VER', 'seconds': 90}, {'Driver': 'LEC', 'seconds': 91}, {'Driver': 'SAI', 'seconds': 92}])
    raw.laps(*GP, 'FP2', [{'Driver': 'VER', 'seconds': 89}, {'Driver': 'LEC', 'seconds': 90}])
    raw.laps(*GP, 'FP3', [{'Driver': 'VER', 'seconds': 88}])

    features = build_fp_features(*GP).set_index('Driver')

    # SAI n'a roulé qu'en FP1 : exclu. LEC a roulé en FP1 et FP2 : gardé, avec une FP3 manquante en NaN
    assert set(features.index) == {'VER', 'LEC'}
    assert math.isnan(features.loc['LEC', 'FP3_gap_to_best_pct'])


def test_gp_with_less_than_two_sessions_is_skipped(raw):
    raw.laps(*GP, 'FP1', [{'Driver': 'VER', 'seconds': 90}])

    assert build_fp_features(*GP) is None


def write_weekend(raw, results):
    for session in ['FP1', 'FP2']:
        raw.laps(*GP, session, [{'Driver': r['Driver'], 'seconds': 90 + i} for i, r in enumerate(results)])
    raw.quali(*GP, results, date="2024-03-02")


def test_top10_keeps_driver_without_q3_time(raw):
    results = [{'Driver': f"D{p:02d}", 'Position': p, 'q3_seconds': 89 + p / 10} for p in range(1, 10)]
    results += [{'Driver': 'D10', 'Position': 10}, {'Driver': 'D11', 'Position': 11}]   # P10 sans temps en Q3 (accident)
    write_weekend(raw, results)

    gp = build_gp_features(*GP).set_index('Driver')

    assert len(gp) == 10 and 'D11' not in gp.index
    assert gp.loc['D10', 'quali_position'] == 10
    assert math.isnan(gp.loc['D10', 'gap_to_pole_pct'])
    assert gp.loc['D01', 'gap_to_pole_pct'] == 0


def test_duplicate_position_keeps_first_listed(raw):
    # Cas réel des Pays-Bas 2024 : un pilote disqualifié reste listé avec la même position que le classé officiel
    results = [{'Driver': f"D{p:02d}", 'Position': p, 'q3_seconds': 89 + p / 10} for p in range(1, 11)]
    results.insert(10, {'Driver': 'DSQ', 'Position': 10, 'q3_seconds': 89.5})
    write_weekend(raw, results)

    gp = build_gp_features(*GP)

    assert len(gp) == 10
    assert 'DSQ' not in set(gp['Driver'])
