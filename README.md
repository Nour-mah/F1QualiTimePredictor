# F1 Quali Time Predictor

Prédire l'ordre du top 10 de la qualification (Q3) d'un Grand Prix de Formule 1 à partir des essais libres (FP1 à FP3) et de la forme récente de chaque pilote. Les données viennent de [FastF1](https://github.com/theOehrly/Fast-F1).

Ce projet de portfolio met autant l'accent sur les **pratiques MLOps et Docker** (pipelines reproductibles, suivi des expériences, déploiement en conteneur) que sur la modélisation. Il est en cours de construction et documenté au fil de l'eau.

## Ce que fait le pipeline

Pour chaque Grand Prix :
1. Il récupère les séances FP1, FP2, FP3 et la qualification via FastF1.
2. Il calcule, pour chaque séance d'essais libres et pour chaque pilote (par rapport à tous les pilotes de la séance) :
   - `FP{n}_gap_to_best_pct` : l'écart en % entre son meilleur tour et le meilleur tour de la séance
   - `FP{n}_compound` : le pneu utilisé pour ce meilleur tour (0 = tendre, 1 = médium, 2 = dur, 3 = intermédiaire, 4 = pluie)
   - `FP{n}_laps_ratio` : son nombre de tours propres divisé par le maximum de la séance
3. Il ajoute la forme récente, calculée uniquement sur les GP antérieurs **de la même saison** :
   - `driver_quali_pos_avg` : la position moyenne du pilote sur ses 5 dernières qualifications
   - `team_quali_pos_avg` : la position moyenne des pilotes de son équipe sur ses 5 derniers GP
4. Il calcule deux cibles à partir des résultats officiels de la qualification :
   - `quali_position` : la position finale dans le top 10 (cible principale)
   - `gap_to_pole_pct = (temps Q3 du pilote - temps de la pole) / temps de la pole * 100` (cible secondaire)
5. Il construit une ligne par pilote et par Grand Prix, limitée au top 10 officiel, et trie le dataset chronologiquement.

Un pilote doit avoir au moins 2 séances d'essais libres valides. Les valeurs manquantes restent en `NaN` (pas d'imputation : XGBoost les gère nativement).

Période couverte : saisons 2022 à 2026, week-ends classiques uniquement, sans les week-ends sprint (82 Grands Prix, 820 lignes).

## Approche de modélisation

- **Apprentissage d'un classement (learning to rank).** La tâche principale est d'ordonner les 10 pilotes de chaque Grand Prix. Le modèle est donc un `XGBRanker`, entraîné avec un groupe par Grand Prix. Il est évalué par la **corrélation de Spearman** entre l'ordre prédit et l'ordre réel, calculée pour chaque Grand Prix puis moyennée (1 = ordre parfait).
- **Une référence exigeante.** Le modèle est comparé à l'ordre de la dernière séance d'essais libres (FP3, à défaut FP2 puis FP1). C'est une référence bien plus difficile à battre que la prédiction d'une valeur constante.
- **Validation temporelle.** Pas de découpage aléatoire : il ferait fuiter de l'information entre pilotes d'un même Grand Prix, et du futur vers le passé. L'évaluation se fait par **validation croisée temporelle**, par blocs de 10 Grands Prix (fenêtre d'entraînement croissante, 6 blocs).
- **Des features relatives.** Les temps au tour dépendent du circuit : le modèle reçoit donc des écarts par rapport au meilleur tour de la séance.
- **Une forme remise à zéro chaque saison.** Les changements d'équipe et de réglementation rendent l'historique de la saison précédente trompeur (par exemple Pérez, chez Red Bull en 2024 puis chez Cadillac en 2026). Au premier GP d'une saison, la forme vaut `NaN` pour tous les pilotes et le modèle s'appuie uniquement sur les essais libres.
- **Un modèle volontairement simple.** Avec environ 800 lignes, les paramètres par défaut de XGBoost (profondeur 6) sur-apprennent et font moins bien que la référence. Le modèle utilise des arbres de profondeur 1, 100 arbres et un pas d'apprentissage de 0,05.

### Résultats actuels (validation croisée temporelle, 6 blocs)

Classement du top 10 (Spearman, plus haut = meilleur) :

| Modèle | Spearman (moyenne ± écart-type) |
|---|---|
| Référence : ordre des essais libres | 0,544 ± 0,110 |
| XGBRanker, écarts FP seuls | 0,578 ± 0,086 |
| XGBRanker, forme du pilote seule | 0,566 ± 0,111 |
| XGBRanker, écarts FP + forme de l'équipe | 0,606 ± 0,059 |
| **XGBRanker, écarts FP + forme du pilote** | **0,627 ± 0,073** |
| + forme de l'équipe | 0,614 ± 0,079 |
| + pneus + ratio de tours | 0,627 ± 0,062 |

Le rythme en essais libres et la forme récente se complètent : combinés, ils battent l'ordre des essais libres en moyenne. Le modèle est devant sur 5 des 6 blocs, et à égalité sur le plus récent (0,667 contre 0,669). L'amélioration est réelle mais modeste. Les hyperparamètres ont été choisis sur ces mêmes blocs, ce qui rend les scores légèrement optimistes.

La remise à zéro par saison améliore surtout le début de la saison 2026, la première de la nouvelle réglementation (Spearman de 0,54 à 0,62 au premier GP, de 0,77 à 0,86 au deuxième). Elle pénalise en revanche les débuts de saison stables, où l'historique de l'année précédente restait utile (premier GP de 2024 : de 0,75 à 0,36).

Prédiction de l'écart au pole (RMSE en points de pourcentage, plus bas = meilleur) : 0,858 avec les écarts FP et la forme du pilote, contre 0,902 pour un modèle naïf qui prédit toujours la moyenne.

### Ce qui n'a pas marché

- **Les temps au tour absolus** : ils dépendent du circuit et font bien pire que la référence naïve.
- **La météo** (retirée) : identique pour tous les pilotes d'un GP, elle ne peut pas changer leur ordre. Ses valeurs absolues servent en plus d'empreinte du circuit et provoquent du sur-apprentissage.
- **La forme de l'équipe** : utile seule, mais redondante avec celle du pilote, car deux coéquipiers qualifient en général proches l'un de l'autre. Elle reste disponible en option.
- **Le pneu et le ratio de tours** : pas de gain mesurable (disponibles en option).

## Suivi des expériences et registre de modèles (MLflow)

Chaque lancement de `train_classical.py` est enregistré comme un run MLflow :
- **paramètres** : cible, groupes de features, fichier de dataset, nombre de GP et de lignes, date du dernier GP d'entraînement, découpage des blocs, hyperparamètres XGBoost ;
- **métriques** : score de chaque bloc pour le modèle et la référence, moyennes, écarts-types et `cv_gain_vs_reference` (positif quand le modèle fait mieux que la référence, quelle que soit la métrique) ;
- **artefacts** : la liste des features (`features.json`) et le modèle final, entraîné sur tous les GP ;
- **automatiquement** : le commit git du code et la date.

Les métadonnées sont stockées dans une base SQLite (`mlflow.db`) et les fichiers dans `mlartifacts/`, à la racine du projet, hors de git.

Chaque run ajoute une version au **registre de modèles** (`f1-quali-ranker` pour le classement). Une seule version porte l'alias `champion` : c'est elle que `predict.py` utilise. L'alias ne change que sur décision explicite (`--promote`), pour qu'une expérience ratée ne passe jamais en prédiction par accident.

## Prédire un Grand Prix à venir

`src/models/predict.py` prédit le top 10 de la qualification d'un Grand Prix dès que ses essais libres sont disponibles (entre la FP3 et la qualification). Il classe toute la grille et enregistre la prédiction dans `data/predictions/`.

- **Pour un Grand Prix à venir**, il charge le modèle `champion` du registre, avec ses features.
- **Pour un Grand Prix déjà vu par le champion**, il réentraîne le modèle sur les Grands Prix strictement antérieurs, pour éviter toute fuite d'information. Il affiche alors aussi les positions réelles, ce qui permet de rejouer un Grand Prix passé.

Grands Prix rejoués (modèle entraîné uniquement sur les GP précédents) :

| Grand Prix | Pilotes du top 10 prédit réellement dans le top 10 | Spearman sur le top 10 prédit |
|---|---|---|
| Bahreïn 2026 | 10 / 10 | 0,75 |
| Espagne 2026 | 9 / 10 | 0,81 |
| Italie 2026 | 8 / 10 | 0,73 |
| Azerbaïdjan 2026 | 8 / 10 | 0,25 |
| Singapour 2025 | 9 / 10 | 0,72 |

Le modèle identifie de façon fiable les pilotes qui accèdent à la Q3, mais l'ordre à l'intérieur du top 10 varie beaucoup d'un Grand Prix à l'autre.

Limite connue : le modèle est entraîné sur les pilotes du top 10 uniquement, alors que la prédiction classe toute la grille. L'entraînement sur toute la grille est la prochaine étape de modélisation.

## Avancement

- [x] Collecte des données FastF1 (`src/data/fetch_fastf1.py`)
- [x] Construction des features par GP et du dataset complet (`src/data/build_features_tabular.py`)
- [x] Modèle de classement avec validation croisée temporelle et groupes de features au choix (`src/models/train_classical.py`)
- [x] Prédiction d'un Grand Prix à venir (`src/models/predict.py`)
- [ ] Améliorer le modèle (entraînement sur toute la grille, écart au coéquipier, saisons supplémentaires)
- [ ] Pipeline tensoriel pour un modèle PyTorch (phase 2)
- [x] Suivi des expériences et registre de modèles (MLflow)
- [x] Tests (pytest, données fabriquées)
- [ ] API de prédiction
- [ ] Docker / docker-compose
- [ ] CI/CD
- [ ] Versioning des données (DVC)

## Outils

Python, pandas, FastF1, scikit-learn, SciPy, XGBoost, MLflow, pytest. Prévus : PyTorch, Docker, DVC.

## Structure du dépôt

```
src/
├── data/
│   ├── fetch_fastf1.py           # Récupère les séances depuis FastF1
│   └── build_features_tabular.py # Construit le dataset tabulaire par GP
└── models/
    ├── train_classical.py        # Classement / régression XGBoost + validation croisée temporelle
    └── predict.py                # Prédit le top 10 de la qualification d'un GP à partir de ses essais libres
tests/           # tests pytest sur des données fabriquées (ne dépendent pas de data/)

data/            # données brutes, traitées et prédictions, exclues de git
mlflow.db        # base MLflow (runs, registre de modèles), exclue de git
mlartifacts/     # artefacts MLflow (modèles, features.json), exclus de git
```

## Lancer le pipeline

Installation (Python 3.12) :

```bash
pip install -r requirements-dev.txt
```

```bash
cd src/data
python fetch_fastf1.py            # remplit data/raw/ en ignorant les séances déjà téléchargées et les week-ends pas encore commencés
python build_features_tabular.py  # construit data/processed/tabular/F1_features_dataset_<horodatage>.csv

cd ../models
python train_classical.py                                      # classement du top 10, features : écarts FP + forme du pilote
python train_classical.py --features gap history team compound laps   # groupes de features : gap, history, team, compound, laps
python train_classical.py --target gap                         # tâche secondaire : régression de l'écart au pole
python train_classical.py --promote                            # désigne le modèle de ce run comme champion
python predict.py --year 2026 --gp "United States Grand Prix"  # top 10 prédit (lancer d'abord fetch_fastf1.py après la FP3)
```

Explorer les runs dans l'interface MLflow (depuis la racine du projet, puis ouvrir http://127.0.0.1:5000) :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Après chaque nouveau Grand Prix couru : relancer `fetch_fastf1.py`, `build_features_tabular.py` puis `train_classical.py --promote`, pour que le champion profite des nouvelles données.

## Tests

```bash
pytest
```

Les tests n'utilisent jamais les vraies données (`data/raw/` est exclu de git, donc absent de la CI) : chacun écrit de petits CSV fabriqués dans un dossier temporaire, avec des valeurs choisies à la main. Ils vérifient en priorité ce qui pourrait casser sans erreur visible :
- **pas de fuite** : la forme d'un pilote n'utilise que les GP antérieurs, repart à zéro chaque saison, et vaut la même chose à l'entraînement et en prédiction ;
- **features et cibles** : écarts relatifs, tours non représentatifs ignorés, règle des 2 séances, pilote sans temps en Q3, position en double ;
- **évaluation** : Spearman, blocs de validation toujours postérieurs à l'entraînement, ranker qui place bien le meilleur pilote en tête ;
- **prédiction** : le modèle champion n'est jamais utilisé sur un GP qu'il a déjà vu.
