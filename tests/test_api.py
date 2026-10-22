"""API : forme des réponses et codes d'erreur (la prédiction elle-même est remplacée par un faux)."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main
from models.predict import MissingPracticeData, Prediction

client = TestClient(main.app)


def fake_prediction(with_results):
    ranking = pd.DataFrame({
        'predicted_position': range(1, 12),
        'Driver': [f"D{i:02d}" for i in range(1, 12)],
        'Team': ['Team A'] * 11,
        'score': [1 - i / 10 for i in range(11)],
    })
    if with_results:
        ranking['actual_position'] = [2, 1] + list(range(3, 12))
    return Prediction(2026, 'Bahrain Grand Prix', '2026-10-03', 'champion v1', ['FP3_gap_to_best_pct'], ranking)


@pytest.mark.parametrize('with_results', [False, True])
def test_predict_returns_top10_and_full_grid(monkeypatch, with_results):
    monkeypatch.setattr(main, 'predict_ranking', lambda year, gp: fake_prediction(with_results))

    response = client.get("/predict", params={'year': 2026, 'grand_prix': 'Bahrain Grand Prix'})

    assert response.status_code == 200
    body = response.json()
    assert len(body['top10']) == 10 and len(body['grid']) == 11
    assert body['top10'][0]['driver'] == 'D01'
    if with_results:
        assert body['evaluation']['drivers_in_real_top10'] == 10
    else:
        assert body['evaluation'] is None and body['top10'][0]['actual_position'] is None


def test_predict_returns_404_without_practice_data(monkeypatch):
    def missing(year, gp):
        raise MissingPracticeData("pas d'essais libres")
    monkeypatch.setattr(main, 'predict_ranking', missing)

    response = client.get("/predict", params={'year': 2026, 'grand_prix': 'Atlantis Grand Prix'})

    assert response.status_code == 404


def test_predict_requires_parameters():
    assert client.get("/predict").status_code == 422


def test_health_without_champion(monkeypatch):
    monkeypatch.setattr(main, 'champion_info', lambda: None)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'champion': None}
