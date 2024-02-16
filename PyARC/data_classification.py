import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForest:

    @staticmethod
    def model_training(df):
        # Carica il tuo DataFrame
        # Supponendo che il DataFrame si chiami "df"
        # Se non hai già scikit-learn installato, puoi farlo con: pip install scikit-learn

        # Estrai la colonna "Cluster" come target e le rimanenti colonne come features
        X = df.drop("Cluster", axis=1)
        y = df["Cluster"]

        # Suddividi il dataset in set di addestramento e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Crea un modello di Random Forest
        rf_model = RandomForestClassifier(random_state=42)

        # Definisci la griglia degli iperparametri da esplorare
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Crea un oggetto GridSearchCV per la ricerca degli iperparametri
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')

        # Esegui la ricerca degli iperparametri sul set di addestramento
        grid_search.fit(X_train, y_train)

        # Visualizza i migliori parametri trovati
        best_params = grid_search.best_params_
        print("Migliori parametri:", best_params)

        # Ottieni il modello con i migliori parametri
        best_rf_model = grid_search.best_estimator_

        # Fai previsioni sul set di test
        y_pred = best_rf_model.predict(X_test)

        # Calcola e stampa l'accuracy sul set di test
        accuracy_test = accuracy_score(y_test, y_pred)
        print("Accuracy sul set di test:", accuracy_test)

        # Calcola e stampa l'accuracy sul set di addestramento
        accuracy_train = best_rf_model.score(X_train, y_train)
        print("Accuracy sul set di addestramento:", accuracy_train)

        return best_rf_model