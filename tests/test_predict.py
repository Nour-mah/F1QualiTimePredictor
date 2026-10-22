"""Choix du modèle dans predict.py : le champion ne doit jamais prédire un GP qu'il a déjà vu."""
import pytest

from models import predict

FEATURES = ['FP3_gap_to_best_pct', 'driver_quali_pos_avg']


@pytest.fixture
def fake_models(monkeypatch):
    """Remplace le registre MLflow et le réentraînement par des faux, pour savoir lequel get_model a choisi."""
    monkeypatch.setattr(predict, 'load_champion', lambda: ('champion', FEATURES, '2026-10-03', 'champion v1'))
    monkeypatch.setattr(predict, 'train_before', lambda gp_date, features: ('réentraîné', f"avant {gp_date}"))


def test_champion_is_used_for_an_upcoming_gp(fake_models):
    model, _, features = predict.get_model('2026-10-24', ['gap'])

    assert model == 'champion'
    assert features == FEATURES   # les features du champion, pas celles passées en option


def test_retrain_when_champion_already_saw_the_gp(fake_models):
    model, description, features = predict.get_model('2026-10-03', ['gap'])

    assert model == 'réentraîné'
    assert description.startswith("avant 2026-10-03")
    assert features == FEATURES


def test_retrain_when_no_champion(monkeypatch):
    monkeypatch.setattr(predict, 'load_champion', lambda: None)
    monkeypatch.setattr(predict, 'train_before', lambda gp_date, features: ('réentraîné', ''))
    monkeypatch.setattr(predict, 'load_data', lambda: (__import__('pandas').DataFrame(columns=FEATURES + ['FP1_compound']), 'x.csv'))

    model, _, features = predict.get_model('2026-10-24', ['gap', 'history'])

    assert model == 'réentraîné'
    assert features == FEATURES
