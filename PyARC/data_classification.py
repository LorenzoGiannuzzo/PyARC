import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class RandomForest:

    def __init__(self):
        self.model = None
        self.accuracy_train_history = []
        self.accuracy_test_history = []

    def _load_dataframe(self, df):
        self.df = df

    def _convert_cluster_to_word(self):
        def convert_to_word(number):
            words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
            return words[number]

        self.df['Cluster'] = self.df['Cluster'].apply(convert_to_word)

    def _extract_features_target(self):
        self.X = self.df.drop("Cluster", axis=1)
        self.y = self.df["Cluster"]

    def _split_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    def _train_model(self):
        rf_model = RandomForestClassifier(criterion='gini', max_depth=None, random_state=42)

        param_grid = {
            'n_estimators': [50, 500],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [20, 30, 40, 80],
            'max_features': [1, 2, 3, 4, 5]
        }

        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        self.model = grid_search.best_estimator_

    def _evaluate_model(self):
        y_pred = self.model.predict(self.X_test)

        # Calculate and print accuracy on the test set
        accuracy_test = accuracy_score(self.y_test, y_pred)
        self.accuracy_test_history.append(accuracy_test)

        # Calculate and print accuracy on the training set
        accuracy_train = self.model.score(self.X_train, self.y_train)
        self.accuracy_train_history.append(accuracy_train)

        print("Accuracy on training set:", accuracy_train)
        print("Accuracy on test set:", accuracy_test)

    def _save_model(self):
        model_folder = "User-trained Model"
        model_filename = os.path.join("..", model_folder, "random_forest_model.joblib")
        dump(self.model, model_filename)

        print(f"Model saved to {model_filename}")

        # Save accuracy history to a text file
        metrics_folder = os.path.join("..", "docs", "User-trained Model Metrics")
        os.makedirs(metrics_folder, exist_ok=True)

        metrics_filename = os.path.join(metrics_folder, "User_trained_model_metrics.txt")
        with open(metrics_filename, "w") as file:
            file.write("Training Accuracy:\n")
            for accuracy in self.accuracy_train_history:
                file.write(f"{accuracy}\n")

            file.write("\nTesting Accuracy:\n")
            for accuracy in self.accuracy_test_history:
                file.write(f"{accuracy}\n")

        print(f"Accuracy history saved to {metrics_filename}")

    def _plot_feature_importance(self):
        def plot_feature_importance(importance, names, model_type):
            feature_importance = np.array(importance)
            feature_names = np.array(names)

            data = {'feature_names': feature_names, 'feature_importance': feature_importance}
            fi_df = pd.DataFrame(data)
            fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

            plt.figure(figsize=(10, 8))
            sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
            plt.title(model_type + ' - Feature Importance')
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature Names')

            plot_filename = os.path.join("..", "plots", "feature_importance_plot.png")
            plt.savefig(plot_filename)

        feature_names = self.X.columns
        plot_feature_importance(self.model.feature_importances_, feature_names, 'Random Forest')

    @staticmethod
    def model_training(df):
        random_forest_instance = RandomForest()

        random_forest_instance._load_dataframe(df)

        random_forest_instance._convert_cluster_to_word()

        random_forest_instance._extract_features_target()

        random_forest_instance._split_dataset()

        random_forest_instance._train_model()

        random_forest_instance._evaluate_model()

        random_forest_instance._save_model()

        random_forest_instance._plot_feature_importance()

        return random_forest_instance.model