# Importation des librairies
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

# Filtrer les avertissements de la bibliothèque scikit-learn
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Importation de mes fonctions
from Fonctions.fonctions import convert_to_sqlite_and_analyze, check_missing_and_plot, svm_classification, add_mean_column, double_cross_validation, perform_stats, plot_stats

from Fonctions.classe import Preprocessing

data = pd.read_csv('JulieDandrimont_donnees_psy3019-H23_20230212_Pandemic.csv', index_col=0)
data = data.replace(' ', np.nan)
preprocessor = Preprocessing(data)
cleaned_data = preprocessor.preprocess(threshold=2.5)

convert_to_sqlite_and_analyze(data, db_name='my_database.db', table_name='my_table')

"""check_missing_and_plot(data)"""

data = data.drop('date', axis=1) # on utilise pas la date pour les analyses

# nettoyer les colonnes object
data['age'] = pd.to_numeric(data['age'], errors='coerce')
data['ARM_P'] = pd.to_numeric(data['ARM_P'], errors='coerce')
data['ARM_R'] = pd.to_numeric(data['ARM_R'], errors='coerce')
data['ARM_Total'] = pd.to_numeric(data['ARM_Total'], errors='coerce')
data['WCC_P'] = pd.to_numeric(data['WCC_P'], errors='coerce')
data['WCC_SS'] = pd.to_numeric(data['WCC_SS'], errors='coerce')
data['WCC_B'] = pd.to_numeric(data['WCC_B'], errors='coerce')
data['WCC_W'] = pd.to_numeric(data['WCC_W'], errors='coerce')
data['WCC_A'] = pd.to_numeric(data['WCC_A'], errors='coerce')

# convertir les colonnes object en float
data['age'] = data['age'].astype(float)
data['ARM_P'] = data['ARM_P'].astype(float)
data['ARM_R'] = data['ARM_R'].astype(float)
data['ARM_Total'] = data['ARM_Total'].astype(float)
data['WCC_P'] = data['WCC_P'].astype(float)
data['WCC_SS'] = data['WCC_SS'].astype(float)
data['WCC_B'] = data['WCC_B'].astype(float)
data['WCC_W'] = data['WCC_W'].astype(float)
data['WCC_A'] = data['WCC_A'].astype(float)

# remplacer les valeurs manquantes par zéro
data = data.fillna(0)

# remplacer les valeurs infinies par la valeur maximale de la colonne
max_value = data.select_dtypes(include=np.number).max().max()

data = data.replace([np.inf, -np.inf], max_value)

# convertir les colonnes object en int
data['age'] = data['age'].astype(int)
data['genderid'] = pd.to_numeric(data['genderid'], errors='coerce').astype(int)
data['genderD'] = pd.to_numeric(data['genderD'], errors='coerce').astype(int)
data['SexOrientationR'] = pd.to_numeric(data['SexOrientationR'], errors='coerce').astype(int)
data['raceD'] = data['raceD'].astype(int)
data['canada'] = data['canada'].astype(int)
data['provinceCA'] = data['provinceCA'].astype(int)
data['sex'] = data['sex'].astype(int)
data['genderid'] = data['genderid'].astype(int)
data['genderD'] = data['genderD'].astype(int)
data['SexOrientationR'] = data['SexOrientationR'].astype(int)
data['LGBTQ'] = data['LGBTQ'].astype(int)
data['LGBT_Subgroup'] = data['LGBT_Subgroup'].astype(int)
data['ARMlog'] = data['ARM_Total'].apply(log_n)

columns = ['Resilience_1', 'Resilience_2', 'Resilience_3', 'Resilience_4', 'Resilience_5', 'Resilience_6', 'Resilience_7', 'Resilience_8', 'Resilience_9', 'Resilience_10', 'Resilience_11', 'Resilience_12', 'Resilience_13', 'Resilience_14', 'Resilience_15']
new_column_name = 'Resilience_mean'
data = add_mean_column(data, columns, new_column_name)

# Appeler les fonctions d'analyse statistique et de visualisation
perform_stats(data)
plot_stats(data)

svm_classification(data)

# Extraire les colonnes numériques
numerical_cols = ['age', 'ARM_Total', 'WCC_P', 'WCC_SS', 'WCC_B', 'WCC_W', 'WCC_A']
X = data[numerical_cols].to_numpy()

# Extraire les labels
Y = data['LGBTQ'].to_numpy()

n_clusters_list = [2, 3, 4, 5, 6]
scores = double_cross_validation(X, Y, n_clusters_list)

