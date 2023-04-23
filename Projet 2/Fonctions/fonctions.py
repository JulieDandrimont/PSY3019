import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
import os
import datetime
import warnings
from sklearn.exceptions import ConvergenceWarning
import math

# Filtrer les avertissements de la bibliothèque scikit-learn
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def convert_to_sqlite_and_analyze(data, db_name='my_database.db', table_name='my_table'):
    # Convertir la base de données en dataframe Pandas
    df = pd.DataFrame(data)

    # Créer une connexion à la base de données SQLite
    conn = sqlite3.connect(db_name)

    # Insérer les données dans une nouvelle table en remplaçant une table existante s'il y en a une
    df.to_sql(table_name, conn, if_exists='replace')

    # Fermer la connexion
    conn.close()

    # Analyser la base de données à partir de la nouvelle table
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Calculer les statistiques descriptives pour chaque colonne
    stats = pd.read_sql_query(f"SELECT * FROM {table_name}", conn).describe(include='all')

    # Enregistrer le tableau de statistiques sous forme de fichier CSV
    stats.to_csv('Donnees/stats.csv')

    # Fermer la connexion
    conn.close()

    # Afficher les statistiques
    print("Statistiques descriptives :\n", stats)


def check_missing_and_plot(data):
    try:
        # Vérifier les données manquantes
        missing_count = data.isnull().sum()

        # Afficher les données manquantes
        print("Nombre de données manquantes :\n", missing_count)

        # Créer un graphique montrant l'emplacement des données manquantes
        plt.figure(figsize=(12,8))
        plt.title('Emplacement des données manquantes')
        plt.imshow(data.isnull(), cmap='viridis', aspect='auto')
        plt.xticks(range(len(data.columns)), data.columns, rotation=90)
        plt.yticks(range(len(data.index)), data.index)
        plt.colorbar()
        plt.show()
    except Exception as e:
        print("Une erreur s'est produite :", e)

def svm_classification(data):
    # Sélectionner les colonnes d'intérêt
    X = data[['age','raceD', 'canada', 'provinceCA', 'SexOrientationR', 'LGBTQ', 'LGBT_Subgroup']]
    y = data['sex']

    # Séparer les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle SVM
    clf = SVC(kernel='linear', C=1, probability=True)
    clf.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = clf.predict(X_test)

    # Calculer la précision du modèle
    accuracy = accuracy_score(y_test, y_pred)
    print('Précision : {:.2f}'.format(accuracy))

    # Tracer la courbe ROC
    y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Enregistrer la courbe ROC dans un fichier PNG
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='Courbe ROC (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC SVM')
    plt.legend(loc='lower right')

    output_dir = 'Donnees'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_svm_classification.png'
    output_path = os.path.join(output_dir, output_filename)

    fig.savefig(output_path, bbox_inches='tight')

    # Enregistrer les performances du modèle dans un fichier texte
    output_filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_svm_classification.txt'
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as f:
        f.write('Précision : {:.2f}\n'.format(accuracy))
        f.write('Courbe ROC (AUC = {:.2f})\n'.format(roc_auc))
        f.write('Taux de faux positifs : {}\n'.format(fpr))
        f.write('Taux de vrais positifs : {}\n'.format(tpr))

    plt.show()


def add_mean_column(data, columns, new_column_name):
    # Sélectionner les colonnes qui contiennent des valeurs numériques
    numeric_columns = data[columns].select_dtypes(include=np.number).columns

    # Calculer la moyenne des colonnes spécifiées
    mean_values = data[numeric_columns].mean(axis=1)

    # Ajouter la nouvelle colonne à la base de données
    data[new_column_name] = mean_values

    # Retourner la base de données modifiée
    return data

import os

def double_cross_validation(X, Y, n_clusters_list, save_dir='Donnees'):
    """
    Effectue une double validation croisée pour estimer les performances d'un algorithme non supervisé (K-means).
    Paramètres :
        - X : tableau numpy des données d'entrée
        - Y : tableau numpy des labels
        - n_clusters_list : liste des nombres de clusters à tester
        - save_dir : dossier où enregistrer les résultats (par défaut : 'Donnees')
    Retourne :
        - tableau numpy des scores de précision (1 score par nombre de clusters)
        - graphique représentant l'évolution du score en fonction du nombre de clusters
    """
    # Nombre de splits pour la validation croisée
    n_splits = 5

    # Initialisation du tableau de scores
    scores = np.zeros(len(n_clusters_list))

    # Validation croisée externe
    kf_outer = KFold(n_splits=n_splits, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf_outer.split(X)):
        X_train_outer, X_test_outer = X[train_index], X[test_index]
        Y_train_outer, Y_test_outer = Y[train_index], Y[test_index]

        # Validation croisée interne
        kf_inner = KFold(n_splits=n_splits, shuffle=True)
        best_score = 0
        for n_clusters in n_clusters_list:
            score = 0
            for j, (train_index_inner, test_index_inner) in enumerate(kf_inner.split(X_train_outer)):
                X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
                Y_train_inner, Y_test_inner = Y_train_outer[train_index_inner], Y_train_outer[test_index_inner]

                # Application de l'algorithme non supervisé (K-means)
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(X_train_inner)

                # Évaluation des performances sur le fold interne
                labels_pred = kmeans.predict(X_test_inner)
                score += np.sum(labels_pred == Y_test_inner) / len(Y_test_inner)

            # Calcul de la moyenne des scores pour tous les folds internes
            score /= n_splits

            # Si le score est meilleur que le meilleur score précédent, on le sauvegarde
            if score > best_score:
                best_score = score

        # Ajout du meilleur score pour cette itération au tableau de scores
        scores += best_score / n_splits

    # Calcul de la moyenne des scores pour toutes les itérations
    scores /= n_splits

    # Génération du graphique
    plt.plot(n_clusters_list, scores)
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Score de précision')
    plt.title('Score en fonction du nombre de clusters')
    plt.savefig(os.path.join(save_dir, 'graphique_knn.png'))
    plt.show()

    # Enregistrement des données dans un fichier texte
    np.savetxt(os.path.join(save_dir, 'Precision_scores_clusters_knn.txt'), scores)

    return scores

import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import os

def perform_stats(data):
    # Régression linéaire multiple
    X = data[['age', 'raceD', 'canada', 'provinceCA', 'sex']]
    Y = data['Resilience_mean']
    model = sm.OLS(Y, sm.add_constant(X)).fit()
    
    # Enregistrer les résultats de la régression dans un fichier texte
    results_path = os.path.join('Donnees', 'results_regression.txt')
    with open(results_path, 'w') as f:
        f.write(model.summary().as_text())
    
    # Test d'hypothèse pour la moyenne
    ttest_result = stats.ttest_ind(data[data['LGBTQ']==1]['Resilience_mean'], data[data['LGBTQ']==0]['Resilience_mean'], equal_var=False)
    
    # Ajouter le résultat du test dans le fichier texte
    with open(results_path, 'a') as f:
        f.write(f"\n\nTest d'hypothèse pour la moyenne : {ttest_result}")
    
    print(model.summary())
    print(ttest_result)
    
def plot_stats(data):
    # Histogramme des âges
    sns.histplot(data['age'], kde=False)
    plt.title("Distribution des âges")
    plt.xlabel("Âge")
    plt.ylabel("Nombre d'individus")
    
    # Enregistrer le graphique dans un fichier image
    plot_path = os.path.join('Donnees', 'age_distribution.png')
    plt.savefig(plot_path)
    
    plt.show()
   

def log_n(x):
    '''
    prend une valeur et retourne son log

    fonction récursive
    '''
    if x <= 1:
        return 0
    else:
        return 1 + log_n(x/math.e)
    
def tri_liste(T):
    """
    
    """
    n = len(T)
    for i in range(n-1):
        min_index = i
        for j in range(i+1, n):
            if T[j] < T[min_index]:
                min_index = j
        T[i], T[min_index] = T[min_index], T[i]
    return T


def select_values_above_threshold(df, col_name, threshold):
    '''
    fontion qui prend le dataframe, le nom de la colonne et un threshold et retourne seulement les valeurs qui sont au dessus du seuil
    col_name must be between brackets
    dans le projet, le threshold est déterminé dans le manuel d'utilisation du test de l'ARM page 25
    https://cyrm.resilienceresearch.org/files/CYRM_&_ARM-User_Manual.pdf

    2e fonction anonyme
    '''
    # create a new column with NaN values where the threshold is not met
    df[f'{col_name}_res'] = df[col_name].apply(lambda x: x if int(x) > threshold else np.nan)
    
    # drop the NaN values and convert to a list
    values_above_threshold = df[f'{col_name}_res'].dropna().tolist()
    
    # return the list
    return values_above_threshold