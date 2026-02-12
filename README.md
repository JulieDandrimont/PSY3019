# Analyse des données psychologiques — Pandemic Study

Projet d'analyse de données issu du cours **PSY3019-H23** (Université de Montréal, hiver 2023). Le script principal effectue un pipeline complet de nettoyage, d'analyse statistique, de visualisation et de classification sur un jeu de données psychologiques collectées pendant la pandémie.

---

## Structure du projet

```
.
├── main.py                                          # Script principal
├── JulieDandrimont_donnees_psy3019-H23_20230212_Pandemic.csv   # Données brutes
├── Fonctions/
│   ├── fonctions.py                                 # Fonctions utilitaires
│   └── classe.py                                    # Classe Preprocessing
└── README.md
```

---

## Données

Le fichier CSV contient des données auto-rapportées sur des participants canadiens, incluant :

- **Variables démographiques** : `age`, `sex`, `genderid`, `genderD`, `SexOrientationR`, `raceD`, `canada`, `provinceCA`
- **Appartenance LGBTQ+** : `LGBTQ`, `LGBT_Subgroup`
- **Mesures psychologiques** :
  - `ARM_P`, `ARM_R`, `ARM_Total` — Mesures de résilience ARM
  - `WCC_P`, `WCC_SS`, `WCC_B`, `WCC_W`, `WCC_A` — Stratégies d'adaptation (WCC)
  - `Resilience_1` à `Resilience_15` — Items d'une échelle de résilience

---

## Pipeline d'analyse

### 1. Chargement et nettoyage
- Lecture du CSV avec `pandas`
- Remplacement des espaces vides par `NaN`
- Détection et suppression des outliers via la classe `Preprocessing` (seuil Z-score = 2.5)
- Conversion des colonnes en types numériques appropriés
- Imputation des valeurs manquantes par `0`
- Remplacement des valeurs infinies par le maximum de la colonne

### 2. Ingestion en base de données
- Conversion du DataFrame en base SQLite via `convert_to_sqlite_and_analyze()`

### 3. Feature engineering
- Transformation logarithmique de `ARM_Total` → `ARMlog`
- Calcul de la moyenne des 15 items de résilience → `Resilience_mean`

### 4. Analyses statistiques
- Statistiques descriptives via `perform_stats()`
- Visualisations via `plot_stats()`

### 5. Classification
- **SVM** sur l'ensemble des données via `svm_classification()`
- **Double validation croisée** (nested cross-validation) sur les variables numériques clés avec différentes valeurs de `n_clusters` (2, 3, 4, 5, 6)

---

## Variables utilisées pour la classification

| Variable    | Description                          |
|-------------|--------------------------------------|
| `age`       | Âge du participant                   |
| `ARM_Total` | Score total de résilience ARM        |
| `WCC_P`     | Adaptation — résolution de problèmes |
| `WCC_SS`    | Adaptation — soutien social          |
| `WCC_B`     | Adaptation — comportemental          |
| `WCC_W`     | Adaptation — retrait                 |
| `WCC_A`     | Adaptation — évitement               |

**Variable cible** : `LGBTQ` (appartenance à la communauté LGBTQ+)

---

## Dépendances

```bash
pip install pandas numpy matplotlib scikit-learn
```

| Librairie      | Usage                                  |
|----------------|----------------------------------------|
| `pandas`       | Manipulation des données               |
| `numpy`        | Calculs numériques                     |
| `matplotlib`   | Visualisations                         |
| `scikit-learn` | SVM, validation croisée                |
| `sqlite3`      | Persistance en base de données         |

---

## Utilisation

```bash
python main.py
```

Assurez-vous que le fichier CSV et le dossier `Fonctions/` se trouvent dans le même répertoire que le script principal.

---

## Auteure

**Julie Dandrimont** — PSY3019, Hiver 2023
