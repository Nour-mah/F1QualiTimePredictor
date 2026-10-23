"""Données fabriquées pour les tests : de faux CSV au format de data/raw, écrits dans un dossier temporaire.

Les tests ne lisent jamais le vrai data/raw (exclu de git, donc absent d'un dépôt fraîchement cloné) : chaque valeur est choisie
à la main, on sait donc exactement quel résultat attendre.
"""
import pandas as pd
import pytest

from data import build_features_tabular


class FakeRaw:
    """Écrit des CSV au format produit par fetch_fastf1.py dans un dossier temporaire."""

    def __init__(self, path):
        self.path = path

    def _file(self, year, gp_name, suffix):
        return self.path / f"{year}_{gp_name.replace(' ', '_')}_{suffix}.csv"

    def laps(self, year, gp_name, session, laps, date="2024-03-01"):
        """laps : liste de dicts {Driver, seconds, accurate=True, compound='SOFT', team='Team A'}."""
        rows = [{
            'Driver': lap['Driver'],
            'LapTime': pd.to_timedelta(lap['seconds'], unit='s'),
            'IsAccurate': lap.get('accurate', True),
            'Compound': lap.get('compound', 'SOFT'),
            'Team': lap.get('team', 'Team A'),
            'LapStartDate': f"{date} 14:00:00.000",
        } for lap in laps]
        pd.DataFrame(rows).to_csv(self._file(year, gp_name, f"{session}_laps"), index=False)

    def quali(self, year, gp_name, results, date):
        """results : liste de dicts {Driver, Position, q3_seconds=None, team='Team A'}.
        Écrit aussi un fichier de tours de qualification, d'où get_gp_date tire la date du GP."""
        rows = [{
            'Abbreviation': r['Driver'],
            'TeamName': r.get('team', 'Team A'),
            'Position': float(r['Position']),
            'Q3': pd.to_timedelta(r['q3_seconds'], unit='s') if r.get('q3_seconds') else None,
        } for r in results]
        pd.DataFrame(rows).to_csv(self._file(year, gp_name, "Q_results"), index=False)
        pd.DataFrame({'LapStartDate': [f"{date} 15:00:00.000"]}).to_csv(self._file(year, gp_name, "Q_laps"), index=False)


@pytest.fixture
def raw(tmp_path, monkeypatch):
    """Redirige build_features_tabular vers un data/raw temporaire et vide."""
    monkeypatch.setattr(build_features_tabular, 'raw_dir', str(tmp_path))
    return FakeRaw(tmp_path)
