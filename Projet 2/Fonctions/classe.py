import pandas as pd

import numpy as np

class Preprocessing:
    def __init__(self, data):
        self.data = data

    def check_missing(self):
        """
        Vérifie le nombre de données manquantes pour chaque variable
        """
        missing_count = self.data.isnull().sum()
        if missing_count.sum() > 0:
            print("Les colonnes suivantes contiennent des valeurs manquantes :\n", 
                  missing_count[missing_count > 0])
        else:
            print("Aucune valeur manquante détectée dans la base de données.")

    def check_aberrant(self, threshold=3):
        """
        Vérifie si des valeurs aberrantes sont présentes dans chaque variable en comparant la moyenne 
        et l'écart-type à un seuil fixé par l'utilisateur.
        """
        stats = self.data.describe()
        mean = stats.loc['mean']
        std = stats.loc['std']
        lower_threshold = mean - threshold * std
        upper_threshold = mean + threshold * std
        aberrant_cols = []
        for col in self.data.columns:
            if self.data[col].dtype in [np.int64, np.float64]:
                aberrant_rows = self.data[(self.data[col] < lower_threshold[col]) | 
                                          (self.data[col] > upper_threshold[col])].index
                if len(aberrant_rows) > 0:
                    aberrant_cols.append(col)
                    print(f"La variable {col} contient {len(aberrant_rows)} valeur(s) aberrante(s).")
        if len(aberrant_cols) == 0:
            print("Aucune valeur aberrante détectée dans la base de données.")
        return aberrant_cols

    def preprocess(self, threshold=3):
        """
        Nettoie la base de données en supprimant les lignes avec des données manquantes et les colonnes
        avec des valeurs aberrantes.
        """
        self.check_missing()
        aberrant_cols = self.check_aberrant(threshold)
        cleaned_data = self.data.dropna().drop(aberrant_cols, axis=1)
        return cleaned_data

