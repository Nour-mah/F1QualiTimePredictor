"""API de prédiction du top 10 de la qualification.

Lancement (depuis la racine du projet) : uvicorn api.main:app --app-dir src
Documentation interactive : http://127.0.0.1:8000/docs
"""
import math
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from data.build_features_tabular import discover_practice_gps
from models.predict import MissingPracticeData, champion_info, predict_ranking

app = FastAPI(
    title="F1 Quali Predictor",
    description="Prédit l'ordre de la qualification d'un Grand Prix de F1 à partir de ses essais libres.",
    version="1.0.0",
)


class Champion(BaseModel):
    name: str
    version: int
    run_name: str
    train_end_date: str


class Health(BaseModel):
    status: str
    champion: Optional[Champion]


class GrandPrix(BaseModel):
    year: int
    grand_prix: str


class DriverPrediction(BaseModel):
    predicted_position: int
    driver: str
    team: Optional[str]
    score: float
    actual_position: Optional[int] = None


class Evaluation(BaseModel):
    drivers_in_real_top10: int
    spearman_top10: float


class PredictionResponse(BaseModel):
    year: int
    grand_prix: str
    date: str
    model: str
    features: list[str]
    top10: list[DriverPrediction]
    grid: list[DriverPrediction]
    evaluation: Optional[Evaluation] = None   # seulement si le GP a déjà eu lieu


def optional_int(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else int(value)


@app.get("/health", response_model=Health)
def health():
    """État du service et modèle 'champion' utilisé pour les GP à venir."""
    return Health(status="ok", champion=champion_info())


@app.get("/grands-prix", response_model=list[GrandPrix])
def grands_prix():
    """Grands Prix dont les essais libres sont disponibles, donc prédictibles."""
    return [GrandPrix(year=year, grand_prix=name) for year, name in discover_practice_gps()]


@app.get("/predict", response_model=PredictionResponse)
def predict(year: int = Query(..., examples=[2026]),
            grand_prix: str = Query(..., examples=["United States Grand Prix"])):
    """Classement prédit de toute la grille. Pour un GP déjà couru, ajoute le résultat réel et l'évaluation."""
    try:
        prediction = predict_ranking(year, grand_prix)
    except MissingPracticeData as error:
        raise HTTPException(status_code=404, detail=str(error))

    grid = [DriverPrediction(
        predicted_position=int(row.predicted_position),
        driver=row.Driver,
        team=row.Team if isinstance(row.Team, str) else None,
        score=float(row.score),
        actual_position=optional_int(getattr(row, 'actual_position', None)),
    ) for row in prediction.ranking.itertuples()]

    evaluation = prediction.evaluation()
    return PredictionResponse(
        year=year,
        grand_prix=grand_prix,
        date=prediction.gp_date,
        model=prediction.model_description,
        features=prediction.features,
        top10=grid[:10],
        grid=grid,
        evaluation=Evaluation(drivers_in_real_top10=evaluation[0], spearman_top10=evaluation[1]) if evaluation else None,
    )
