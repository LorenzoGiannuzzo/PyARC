import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import matplotlib.pyplot as plt
import  numpy as np
import seaborn as sns

class RandomForest:



    @staticmethod
    def model_training(df):
        # Carica il tuo DataFrame
        # Supponendo che il DataFrame si chiami "df"
        # Se non hai già scikit-learn installato, puoi farlo con: pip install scikit-learn

        def convert_to_word(number):
            # Funzione per convertire un numero in una parola
            words = ["Zero", "One", "Two", "Three", "Four", "Five","Six","Seven","Eight","Nine"]  # Aggiungi le parole mancanti secondo necessità
            return words[number]

        # Sostituisci gli elementi nella colonna "Cluster" con le parole corrispondenti
        df['Cluster'] = df['Cluster'].apply(convert_to_word)

        # Estrai la colonna "Cluster" come target e le rimanenti colonne come features
        X = df.drop("Cluster", axis=1)
        y = df["Cluster"]

        # Suddividi il dataset in set di addestramento e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Crea un modello di Random Forest
        rf_model = RandomForestClassifier(criterion='gini', max_depth=None, random_state=42)

        # Definisci la griglia degli iperparametri da esplorare
        param_grid = {
            'n_estimators': [50, 500],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [20, 30, 40, 80],
            'max_features': [1, 2, 3, 4, 5]
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

        # Calcola e stampa l'accuracy sul set di addestramento
        accuracy_train = best_rf_model.score(X_train, y_train)
        print("Accuracy sul set di addestramento:", accuracy_train)
        print("Accuracy sul set di test:", accuracy_test)

        model_folder = "Pre-trained Model"
        # Percorso completo per il file del modello
        model_filename = os.path.join("..", model_folder, "random_forest_model.joblib")

        # Salvare il modello
        dump(best_rf_model, model_filename)
        # Se stai utilizzando scikit-learn >= 0.24.0, sostituisci la riga sopra con:
        # dump(best_rf_model, model_filename)

        print(f"Modello salvato in {model_filename}")

        def plot_feature_importance(importance, names, model_type):
            # Crea un DataFrame con le variabili e le relative importanze
            feature_importance = np.array(importance)
            feature_names = np.array(names)

            data = {'feature_names': feature_names, 'feature_importance': feature_importance}
            fi_df = pd.DataFrame(data)

            # Ordina il DataFrame in base all'importanza delle feature
            fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

            # Crea un grafico a barre
            plt.figure(figsize=(10, 8))
            sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
            # Aggiungi etichette e titolo
            plt.title(model_type + ' - Feature Importance')
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature Names')

            # Salva il grafico su un file
            plot_filename = os.path.join("..", "plots" , "feature_importance_plot.png")
            plt.savefig(plot_filename)

        # Chiamata alla funzione prima del return del modello
        feature_names = X.columns
        plot_feature_importance(best_rf_model.feature_importances_, feature_names, 'Random Forest')

        return best_rf_model