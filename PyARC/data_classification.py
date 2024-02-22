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


    def _load_dataframe(self, df):
        self.df = df

    def _convert_cluster_to_word(self):
        def convert_to_word(number):
            # Function to convert a number to a word
            words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]  # Add missing words as needed
            return words[number]

        # Replace elements in the "Cluster" column with corresponding words
        self.df['Cluster'] = self.df['Cluster'].apply(convert_to_word)

    def _extract_features_target(self):
        # Extract features and target from the DataFrame
        self.X = self.df.drop("Cluster", axis=1)
        self.y = self.df["Cluster"]

    def _split_dataset(self):
        # Split the dataset into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    def _train_model(self):
        # Create a Random Forest model
        rf_model = RandomForestClassifier(criterion='gini', max_depth=None, random_state=42)

        # Define the hyperparameter grid to explore
        param_grid = {
            'n_estimators': [50, 500],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [20, 30, 40, 80],
            'max_features': [1, 2, 3, 4, 5]
        }

        # Create a GridSearchCV object for hyperparameter tuning
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')

        # Perform hyperparameter tuning on the training set
        grid_search.fit(self.X_train, self.y_train)

        # Get the model with the best parameters
        self.model = grid_search.best_estimator_

    def _evaluate_model(self):
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)

        # Calculate and print accuracy on the test set
        accuracy_test = accuracy_score(self.y_test, y_pred)

        # Calculate and print accuracy on the training set
        accuracy_train = self.model.score(self.X_train, self.y_train)
        print("Accuracy on training set:", accuracy_train)
        print("Accuracy on test set:", accuracy_test)

    def _save_model(self):
        model_folder = "User-trained Model"
        # Full path for the model file
        model_filename = os.path.join("..", model_folder, "random_forest_model.joblib")

        # Save the model
        dump(self.model, model_filename)
        # If you're using scikit-learn >= 0.24.0, replace the line above with:
        # dump(self.model, model_filename)

        print(f"Model saved to {model_filename}")

    def _plot_feature_importance(self):
        def plot_feature_importance(importance, names, model_type):
            # Create a DataFrame with variables and their importance
            feature_importance = np.array(importance)
            feature_names = np.array(names)

            data = {'feature_names': feature_names, 'feature_importance': feature_importance}
            fi_df = pd.DataFrame(data)

            # Sort the DataFrame by feature importance
            fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

            # Create a bar plot
            plt.figure(figsize=(10, 8))
            sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
            # Add labels and title
            plt.title(model_type + ' - Feature Importance')
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature Names')

            # Save the plot to a file
            plot_filename = os.path.join("..", "plots", "feature_importance_plot.png")
            plt.savefig(plot_filename)

        # Call the function before returning the model
        feature_names = self.X.columns
        plot_feature_importance(self.model.feature_importances_, feature_names, 'Random Forest')


        # Call the function before returning the model
        feature_names = self.X.columns
        plot_feature_importance(self.model.feature_importances_, feature_names, 'Random Forest')

    @staticmethod
    def model_training(df):
        random_forest_instance = RandomForest()

        # Load your DataFrame
        # Assuming the DataFrame is named "df"
        # If you don't have scikit-learn installed, you can do so with: pip install scikit-learn
        random_forest_instance._load_dataframe(df)

        random_forest_instance._convert_cluster_to_word()

        random_forest_instance._extract_features_target()

        random_forest_instance._split_dataset()

        random_forest_instance._train_model()

        random_forest_instance._evaluate_model()

        random_forest_instance._save_model()

        random_forest_instance._plot_feature_importance()

        return random_forest_instance.model