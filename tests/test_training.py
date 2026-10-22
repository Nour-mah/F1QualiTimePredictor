"""Évaluation et entraînement du modèle de classement."""
import numpy as np
import pandas as pd
import pytest

from models.train_classical import (BLOCK_SIZE, MIN_TRAIN_GPS, POSITION_TARGET, fit_ranker, mean_spearman,
                                    select_features, temporal_folds)


def one_gp(gp_id, positions, scores):
    return pd.DataFrame({'gp_id': gp_id, POSITION_TARGET: positions, 'score': scores})


def test_spearman_is_one_for_perfect_order_and_minus_one_for_reversed():
    gp = one_gp(0, [1, 2, 3, 4], [4.0, 3.0, 2.0, 1.0])   # score le plus haut = P1

    assert mean_spearman(gp, 'score', ascending=False) == pytest.approx(1)
    assert mean_spearman(gp, 'score', ascending=True) == pytest.approx(-1)


def test_spearman_is_averaged_per_gp():
    perfect = one_gp(0, [1, 2, 3], [3.0, 2.0, 1.0])
    reversed_ = one_gp(1, [1, 2, 3], [1.0, 2.0, 3.0])

    assert mean_spearman(pd.concat([perfect, reversed_]), 'score', ascending=False) == pytest.approx(0)


def test_select_features_by_group():
    df = pd.DataFrame(columns=['Driver', 'FP1_gap_to_best_pct', 'FP2_gap_to_best_pct', 'driver_quali_pos_avg', 'FP1_compound'])

    assert select_features(df, ['gap']) == ['FP1_gap_to_best_pct', 'FP2_gap_to_best_pct']
    assert select_features(df, ['gap', 'history']) == ['FP1_gap_to_best_pct', 'FP2_gap_to_best_pct', 'driver_quali_pos_avg']
    with pytest.raises(ValueError):
        select_features(df, ['meteo'])


@pytest.mark.parametrize('n_gps', [30, 82, 95])
def test_folds_never_train_on_the_future(n_gps):
    starts = temporal_folds(n_gps)

    assert starts[0] >= MIN_TRAIN_GPS                     # assez de GP pour le premier entraînement
    assert starts[-1] + BLOCK_SIZE == n_gps                # le dernier bloc finit au dernier GP
    assert all(b - a == BLOCK_SIZE for a, b in zip(starts, starts[1:]))   # blocs contigus, sans chevauchement
    # Le train d'un bloc = tous les gp_id < start, le test = [start, start + BLOCK_SIZE) : toujours après le train


def test_ranker_puts_best_driver_first():
    # 30 GP où la feature ordonne parfaitement les pilotes : le modèle doit retrouver l'ordre.
    # Échoue si l'on oublie d'inverser la position (P1 doit avoir le label le plus haut).
    rng = np.random.default_rng(0)
    rows = []
    for gp_id in range(30):
        pace = np.sort(rng.uniform(0, 3, 10))   # le plus petit écart en FP = P1
        rows += [{'gp_id': gp_id, POSITION_TARGET: p + 1, 'FP3_gap_to_best_pct': pace[p]} for p in range(10)]
    train = pd.DataFrame(rows)

    model = fit_ranker(train, ['FP3_gap_to_best_pct'])
    new_gp = pd.DataFrame({'FP3_gap_to_best_pct': [0.0, 1.0, 2.5]})
    scores = model.predict(new_gp)

    assert scores[0] > scores[1] > scores[2]
