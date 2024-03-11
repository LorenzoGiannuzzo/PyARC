
## PyARC - Python Algorithm for Residential load profiles reConstruction (Python 3.12.1)

*Lorenzo Giannuzzo (a,b), Francesco Demetrio Minuto (a,b), Daniele Salvatore Schiera (a,b), Andrea Lanzini (a,b).

_(a) Energy Center Lab, Polytechnic of Turin, via Paolo Borsellino 38/16, 10152, Turin, Italy_ _(b) Department of Energy (DENERG), Polytechnic of Turin, Corso Duca degli Abruzzi 24, 10129, Turin, Italy_

PyARC Software - Python Algorithm for Residential load profiles reConstruction (Python 3.12.1).

## Overview

PyARC is a Python-developed software designed for the reconstruction of residential aggregate electrical load profiles. It leverages an algorithm for Association Rule Mining to model complex relationships between Time-of-Use (ToU) data and electrical consumption profiles. The software uses a non-intrusive machine learning methodology that can be used to generate residential electrical consumption profiles at an hourly resolution level using only monthly consumption data (i.e., billed energy). The methodology is mainly composed of three phases: first, identifying the typical load patterns of residential users through k-Means clustering, supported by evaluation metrics to identify the optimal number of clusters such as the Davies-Boudin Index, Elbow Method, and Silhouette Score, then implementing a Random Forest algorithm, based on features extracted from monthly energy bills, to identify typical load patterns and, finally, reconstructing the hourly electrical load profile through a rescaling factor.

## PyARC Workflow

- **Model Training:**

    - Execute the `train_model` method in `PyARC.py` to train a new PyARC model. This method trains the classification model used to reconstruct the aggregate electrical load profiles of residential users using generic input data contained in the `Input Training Data` folder (electrical energy measures on an hourly scale and Time of Use subdivision)

- **Reconstruction using Pre-trained Model:**

    - Execute the `reconstruct_profiles` method in `PyARC.py` to reconstruct profiles using the pre-trained model contained in the `Pre-trained Model` folder and the input data contained in the `Input Data` folder.

    - The PyARC instance is initialized, and the pre-trained model is used for profile reconstruction.

- **Reconstruction using User-trained Model:**

    - Execute the `user_trained_model` method in `PyARC.py` to reconstruct profiles using a user-trained model contained in the `User-trained Model` folder. If no model is inside the folder, the **Model Training** features have to be executed with the required `Input Data`.

    - The PyARC instance is initialized, and the user-trained model is utilized for profile reconstruction.

## How to Run:

1. Install necessary libraries typing in the terminal: `pip install -r requirements.txt`

2. Ensure the necessary libraries are installed (required libraries are defined in the `requirement.txt` file)

3. Run `main.py`.

4. Choose an option (1, 2, or 3) based on the desired action:

    - 1: Reconstruct profiles using the pre-trained model.

    - 2: Train a new PyARC model.

    - 3: Reconstruct profiles using a user-trained model.

## Input

**Model Training:**

- Hourly electrical consumption measures expressed in [kWh] in CSV format used to train the `User-trained Model` must be contained in the `Input Training Data` folder. (Data structure and requirements are specified in **Directory Structure/data** section).

- Time of Use (ToU) data in CSV format used to train the `User-trained Model` must be contained in the `Input Training Data` folder. (Data structure and requirements are specified in **Directory Structure/data** section).

**Reconstruction using Pre-trained Model:**

- Monthly electrical energy bills expressed in [kWh] in CSV format used to reconstruct the residential aggregate load profile of users must be contained in the `Input Data` folder. (Data structure and requirements are specified in **Directory Structure/data** section).

A pre-trained classification model is required to perform the load profile reconstruction algorithm. The pre-trained model is already placed and available in the `Pre-trained Model` folder.

- Time of Use (ToU) data in CSV format is required and already placed in the `Input Data` folder. (Data structure and requirements are specified in **Directory Structure/data** section).

**Reconstruction using User-trained Model:**

- Monthly electrical energy bills expressed in [kWh] in CSV format used to reconstruct the residential aggregate load profile of users must be contained in the `Input Data` folder. (Data structure and requirements are specified in **Directory Structure/data** section).

A user-trained classification model is required to perform the load profile reconstruction algorithm. The User-trained model must be trained before performing the reconstruction process using the `User-trained Model`. To do this, it's necessary to execute the **Model Training** process (see **Input/Model Training** section). Once executed, the model will be saved in the `User-trained Model` folder and used to reconstruct the residential aggregate load profile.

- Time of Use (ToU) data in CSV format is required and must be in the `Input Data` folder. (Data structure and requirements are specified in **Directory Structure/data** section).

## Output

**Model Training:**

The main output of the model training process is a RandomForest classifier which is used during the reconstruction of the residential aggregate load profile. The model is saved in the `User-trained Model` folder.

The secondary output is:

- Cluster centroids obtained during the clustering process, which are saved in the `plots` folder in .png format as `Cluster_Centroids_Profiles.png`;

- The Davies-Bouldin Index values used to evaluate the optimal number of clusters, which is saved in the `plots` folder in .png format as `Davies-Boulding Index vs Number of CLusters.png` ;

- The "Elbow Method" metric values used to evaluate the optimal number of clusters, which are saved in the `plots` folder in .png format as `Elbow Method vs Number of Clusters.png`;

- The Silhouette score values used to evaluate the optimal number of clusters, which are saved in the `plots` folder in .png format as `Silhouette Score vs Number of Clusters.png`;

- The importance of the features used by the RandomForest to perform the classification task, save in the `plots` folder as `feature_importance_plot.png` ;

- A file .txt containing the accuracy of the RandomForest during training and test, which is saved in the `docs/User-trained Model Metrics` folder as `User_trained_model_metrics.txt`.

**Reconstruction using Pre-trained Model:**

The output of the reconstruction of the residential aggregate load profile process is as follows:

- A file containing the time series of the aggregate electrical load profile in CSV format, which is saved in the `data/Output Data` folder.

- The plot of the aggregate electrical load profiles for each month, which is saved in the `plots` folder in .png format as `Aggregate load profiles.png`.

- **Reconstruction using User-trained Model:**

The output of the process is the same as the - **Reconstruction using Pre-trained Model** process, namely:

- A file containing the time series of the aggregate electrical load profile in CSV format, which is saved in the `data/Output Data` folder structured as follows:

`Year, Month,Day,Hour,Aggregate load   2012,1,1,0,5362.737504088515   2012,1,1,1,4699.1362917288525   2012,1,1,2,4299.034252481011   2012,1,1,3,4143.05778396232   2012,1,1,4,4043.022294386959   2012,1,1,5,4402.389390089283 ...`

where: - "User": is a char/string which contains usernames; - "Year": is a numerical value which represents the year when electrical energy was measured; - "Month": is a numerical value which represents the months when electrical energy was measured; - "Day": is a numerical value which represents the number of days when electrical energy was measured; - "Hour": is a numerical value which represents the hours when electrical energy was measured; -"Aggregate load": contains electrical energy consumption of the residential aggregate expressed in [kWh].

- The plot of the aggregate electrical load profiles for each month, which is saved in the `plots` folder in .png format as `Aggregate load profiles.png`.

## Directory Structure:

- **data:** Contains input data required to run the code in CSV format.

    - _Input Training Data:_ This folder contains the required data to train the user-trained model, which can be used to reconstruct the electrical load profile of residential aggregates.

        - `train_data.csv`: Training data file in CSV format. This data must be structured as follows:

`"User","Year","Month","Day","Hour","Consumption" "MAC000016",2012,1,1,0,0.0275 "MAC000016",2012,1,1,1,0.0735 "MAC000016",2012,1,1,2,0.0395 "MAC000016",2012,1,1,3,0.0305 ...`

```

        where:

        - "User": is a char/string which contains usernames;

        - "Year": is a numerical value that represents the year when electrical     energy was measured;

        - "Month": is a numerical value that represents the months when electrical energy was measured;

        - "Day": is a numerical value that represents the number of days when electrical energy was measured;

        - "Hour": is a numerical value that represents the hours when electrical energy was measured;

        - -"Consumption": contains electrical energy values expressed in [kWh].

    - `train_tou.csv`: Training Time of Use (ToU) data in CSV format. This file represents the specific Time of Use on a weekday related to the train data (train_data.csv). The ToU must be structured as follows:

```

  `"Hour","ToU"   0,"F3"   1,"F3"   2,"F3"   3,"F3"   4,"F3"   5,"F3"   6,"F3"   7,"F2"   8,"F2"   9,"F2"   10,"F2"   11,"F1"   12,"F1"   13,"F1"   14,"F2"   15,"F2"   16,"F1"   17,"F1"   18,"F1"   19,"F2"   20,"F2"   21,"F2"   22,"F2"   23,"F3"`     

```

        where:

        - "Hour": is a numerical value that represents the Hour is which the ToU is divided;

        - "ToU": contains the Time of Use subgroup names expressed as char/string.

```

- _Input Data:_ This folder contains the required data to use the user-trained, which can be used to reconstruct the electrical load profile of residential aggregates.

    ```

     - `data.csv`: file in CSV format containing the electrical energy bills on a monthly scale, divided based on the ToU. The file must be structured as follows:

    ```

`User,Year,Month,F1,F2,F3   MAC000016,2012,1,32.27275,42.496,9.63   MAC000016,2012,2,7.841,9.5075,5.58 MAC000016,2012,4,3.456,4.502,3.31   MAC000016,2012,5,2.8345,4.06825,3.46 MAC000016,2012,6,3.504,4.82775,2.8235 ...`

```

        where:

        - "User": is a char/string which contains usernames;

        - "Year": is a numerical value that represents the year when electrical     energy was measured;

        - "Month": is a numerical value that represents the months when electrical energy was measured;

        -"F1,F2,F3": contains electrical energy bills expressed in [kWh] and divided based on the ToU. The column name must be the same as the ToU timeframes.

            - `tou.csv`: file in CSV format containing the electrical energy bills on a monthly scale, divided based on the ToU. The file must be structured as follows:

```

  `"Hour","ToU"   0,"F3"   1,"F3"   2,"F3"   3,"F3"   4,"F3"   5,"F3"   6,"F3"   7,"F2"   8,"F2"   9,"F2"   10,"F2"   11,"F1"   12,"F1"   13,"F1"   14,"F2"   15,"F2"   16,"F1"   17,"F1"   18,"F1"   19,"F2"   20,"F2"   21,"F2"   22,"F2"   23,"F3"`     

```

        where:

        - "Hour": is a numerical value that represents the Hour is which the ToU is divided;

        - "ToU": contains the Time of Use subgroup names expressed as char/string.

```

- _Default Training Data:_ This folder contains the data used to train the pre-trained or the pre-trained model, which can be used to reconstruct the electrical load profile of residential aggregates. The files inside that folder are structured as the files in the `Input Training Data` folder.

- _Centroids_: This folder contains the data containing the information about cluster centroids' shape, obtained during the clustering process. The folder contains a file in CSV format that is structured as follows:

`Cluster,Hour,Centroid   1,0,0.2637207555426242   1,1,0.22434655768686296   1,2,0.1997015622259806   1,3,0.18870161496496365   1,4,0.18341704840807113   ...  2,0,0.6829219648948441   2,1,0.6445736349804456   2,2,0.6290312587566406   2,3,0.6235405579184536   2,4,0.6243397833412112   ... 3,0,0.15704705883607284   3,1,0.13702333683359103   3,2,0.12460339667704806   3,3,0.12093336923863841   3,4,0.11674785616109161   ...  4,0,0.3947443102687575   4,1,0.33696615765171994   4,2,0.30173250234809096   4,3,0.2840255217189579   4,4,0.27625352359190597   ...`

```

        where:

        - "Cluster" is a numerical value that represents the cluster/subgroup obtained during the clustering process;

        - "Hour": is a numerical value that represents the hours;

        - "Centroids": contains normalized numerical values (range from 0 to 1) which are representative of the normalized electrical consumption of the respective cluster.

```

- **PyARC:** Folder containing the scripts of the algorithms and related functionalities.

```

-  `main.py:`The provided code defines a Python script for managing and utilizing PyARC models. Below is a succinct description:

```

*ModelManager Class:

```

   - The `ModelManager` class manages instances of the `PyARC` class for handling PyARC models.

   - It has methods for using a pre-trained model (`use_pretrained_model`), training a new model (`train_new_model`), and using a user-trained model (`use_user_trained_model`).

```

*start_program Function:

```

- The `start_program` function serves as the entry point for the script.

- It prompts the user to choose an action (1. Use a pre-trained model, 2. Train a new model, 3. Use a user-trained model) and instantiates the `ModelManager` accordingly.

- Based on the user's choice, it executes the corresponding method and provides a success message.

```

*Script Execution:

```

- The `__main__` block ensures that the script runs when executed directly.

- It calls the `start_program` function to initiate user interaction and model management.

```

*Usage:

```

- Users interact with the script by entering a number corresponding to their desired action.

- The script then uses the `ModelManager` to perform the chosen action: using a pre-trained model, training a new model, or using a user-trained model.

```

This script acts as a user-friendly interface for managing PyARC models, allowing users to perform different actions based on their requirements.

```

-`PyARC.py:` The `PyARC` class contains a set of functionalities for training and using a Random Forest model, conducting data preprocessing, clustering, feature extraction, and generating aggregate load profiles. Below is a concise description of the key functionalities:

```

*Initialization:

```

- The `__init__` method initializes a PyARC instance with a `model` attribute set to `None`.

```

*Training the Model:

```

- The `train_model` method trains a Random Forest model using input data and Time of Use (TOU) information.

- It involves loading and processing data, handling TOU data, conducting data preprocessing, normalization, filtering, and visualization using plots.

- K-means clustering is applied to identify optimal cluster numbers, and the resulting centroids are exported to a CSV file.

```

*Reconstructing Profiles:

```

- The `reconstruct_profiles` method reconstructs user profiles using a pre-trained Random Forest model.

- It loads the model, input data, TOU data, and centroids.

- Features are extracted and filtered based on the model's feature names, and cluster predictions are made.

- Further data manipulation, identification of Time of Use, and additional feature engineering are performed.

- Output data is aggregated, and aggregate load profiles are visualized and exported to a CSV file.

```

*User-Trained Model:

```

- The `user_trained_model` method uses a user-trained Random Forest model for profile reconstruction.

- It loads the user-trained model, input data, TOU data, and centroids.

- Features are extracted, and predictions are made based on the model's feature names.

- Data is merged, Time of Use values are spotted, and additional feature engineering is conducted.

- The output includes expanded data, load profiles, and aggregated load profiles, which are visualized and exported.

```

This comprehensive class facilitates the end-to-end process of training and utilizing a model for load profile reconstruction and analysis.

```

 -  `data_aggregation.py`: This script defines a class called `Aggregator` that contains three static methods for data manipulation using the `Pandas` library:

 *Expand_dataframe(df):

  - Takes a DataFrame `df` as input, which presumably has data merged from different sources.

  - Expands the DataFrame by adding columns for "Day" and "Hour."

  - Generates a new DataFrame by combining the input DataFrame with a DataFrame containing hours from 0 to 23.

  - Calculates the "Day" column based on the month and returns the expanded DataFrame.

 *load_profile_generator(df):*

  - Takes a DataFrame `df` as input.

  - Generates load profiles by multiplying the values in the "weight" column by corresponding values in the "main ToU" column.

  - Iterates through unique elements in the "main ToU" column, multiplies the "weight" column, and updates the "load" column in the DataFrame.

 *aggregate_load(df):*

  - Takes a DataFrame `df` with load profiles as input.

  - Aggregates the "load" column for each combination of "Year," "Month," "Day," and "Hour."

  - Returns a new DataFrame (`aggregated_df`) with columns renamed to 'Aggregate load' and 'Hour.'

 Each method includes error handling with try-except blocks, raising a `ValueError` with an error message if an exception occurs. The code seems to be designed for data preprocessing and manipulation, likely in the context of time-series data analysis.

-  `data_classification.py`: This script defines a class called `RandomForest` that encapsulates the process of training a RandomForest classifier, evaluating its performance, saving the model, and plotting feature importance.

 *Initialization:

  - Initializes the `RandomForest` class with attributes for the model, training accuracy history, and testing accuracy history.

 *Data Loading and Preprocessing:

  - `_load_dataframe(df)`: Loads the input DataFrame into the class attribute.

  - `_convert_cluster_to_word()`: Converts numerical cluster labels to corresponding words.

  - `_extract_features_target()`: Extracts features and targets from the DataFrame.

  - `_split_dataset()`: Splits the dataset into training and test sets.

 *Model Training:

  - `_train_model()`: Creates a Random Forest model and performs hyperparameter tuning using GridSearchCV.

 *Model Evaluation:*

  - `_evaluate_model()`: Makes predictions on the test set, calculates and stores accuracy on both training and test sets.

 *Model Saving and Metrics:*

  - `_save_model()`: Saves the trained model in a specified folder and also saves accuracy history to a text file.

 *Feature Importance Plotting:

  - `_plot_feature_importance()`: Plots and saves the feature importance of the trained model.

 *Model Training Process:

  - `model_training(df)`: Performs the entire process of loading data, training the model, evaluating performance, saving the model and metrics, and plotting feature importance. Returns the trained Random Forest model.

The script uses external libraries like Pandas, scikit-learn (for RandomForestClassifier, train_test_split, GridSearchCV), joblib (for model persistence), and Matplotlib and Seaborn for plotting.

```

- `data_clustering.py`: This script defines a class called `Clustering` that provides methods for determining the optimal number of clusters using metrics such as the Davies-Bouldin Index, Silhouette Score, and the Elbow Method, and for performing k-means clustering.

    *Optimal Cluster Number Determination:

    ```

    - find_optimal_cluster_number(data)`: Takes a DataFrame `data` as input.

        - Checks if the required column 'M_consumption' exists.

    - Select the "M_consumption" column for clustering.

    - Iterates over a range of cluster numbers (4 to 8) and calculates the Davies-Bouldin Index, Silhouette Score, and Distortion (Elbow Method).

    - Plots and saves the trends of these metrics against the number of clusters.

    - Counts votes for each cluster number based on the metrics and determines the optimal number of clusters by consensus.

    - Returns the optimal number of clusters and the votes.

    ```

    *K-Means Clustering:

    ```

    - `kmeans_clustering(df, optimal_cluster_number)`: Takes a DataFrame `df` and the optimal number of clusters.

    - Checks if the required column 'M_consumption' exists.

    - Creates a temporary dataframe for clustering features.

    - Performs k-means clustering with the optimal number of clusters.

    - Obtains cluster centroids and adds them to the result dataframe.

    - Transposes the dataframe into a "long" format for plotting.

    - Merges the results with the original dataframe.

    - Returns the result dataframe and the dataframe with cluster centroids.

    ```

    The code utilizes external libraries such as Pandas, scikit-learn (for KMeans, davies_bouldin_score, silhouette_score), Matplotlib, and NumPy for data manipulation and visualization.

- `data_normalization`: This script defines a class named `DataNormalization` that contains data normalization operations on a pandas DataFrame. The class is initialized with a DataFrame, and it provides a method called `normalize_consumption`. Here's a detailed breakdown:

    *Constructor (`__init__` method):

- Takes a pandas DataFrame (`dataframe`) as an argument and assigns it to the class attribute `self.dataframe`.

*Method (`normalize_consumption`):

```

- Begins with a try-except block to handle potential exceptions.

- Checks if the 'Consumption' column is present in the DataFrame. If not, raise a `ValueError` with an appropriate message.

- Verifies if the DataFrame is not empty. If it is empty, raise a `ValueError` with an appropriate message.

- If the above checks pass:

    - Groups the DataFrame by 'User', 'Year', and 'Month'.

    - Calculates the maximum consumption for each group using the `transform` method.

    - Creates a new column named 'Norm_consumption' in the DataFrame, containing normalized values. Normalization is done by dividing the 'Consumption' values by the corresponding maximum consumption per group. A replacement of 0 with 1 is performed to prevent division by zero.

- The method returns the modified DataFrame.

```

*Exception Handling:

```

- If any exception occurs during the normalization process, an error message is printed, and the method returns `None`.

```

In summary, the class is designed to normalize the 'Consumption' column of a DataFrame by dividing its values by the maximum consumption within each group defined by 'User', 'Year', and 'Month'. It includes checks for the presence of the column and the non-emptiness of the DataFrame, with error handling to address potential issues during the normalization process.

- `data_preparation.py`: This script defines a class called `DataFrameProcessor` for handling and processing pandas DataFrames.

*Class Definition (`DataFrameProcessor`):

```

- The class has a constructor (`__init__` method) that takes a DataFrame as input and initializes the class attribute `self.dataframe` with it.

```

*Method (`process_dataframe`):

```

- This method processes the DataFrame with the following steps:

    - Checks if required columns ("User", "Year", "Month", "Day", "Hour", "Consumption") are present. If any are missing, it raises a `ValueError` with a message indicating the missing columns.

    - Creates a new column named "Dayname" based on the columns "Year", "Month", and "Day" by converting them to a datetime object.

    - Drops the intermediate "Date" column if not needed.

    - Returns the processed DataFrame.

```

*Static Method (`check_tou`):

```

- A static method that checks if the "Hour" and "ToU" columns are present in the given DataFrame (`dataframe`):

    - Raises a `ValueError` if either "Hour" or "ToU" is missing.

    - Prints a message if both columns are present.

```

The `DataFrameProcessor` class is designed to perform various operations on a DataFrame, such as checking for required columns, creating a "Dayname" column, dropping unnecessary columns, creating data subsets based on the "Month" column, and checking the presence of specific columns. The class aims to enhance the functionality and usability of working with pandas DataFrames.

- `data_preprocessing.py`: This script defines a class named `DataPreprocessing` that contains various data preprocessing methods using pandas, numpy, and scikit-learn libraries.

*Class Definition (`DataPreprocessing`):

```

- The class is initialized with a DataFrame, and the DataFrame is stored as an attribute (`self.dataframe`).

-

```

*Method (`get_negative_values`):

```

- Checks if the "Consumption" column is present in the DataFrame.

- If present, replace negative values in the "Consumption" column with NaN.

- Returns the modified DataFrame. If the "Consumption" column is not present, it prints a message and returns None.

```

*Static Method (`replace_max_daily_zero_consumption`):

```

- Identifies profiles with the maximum daily zero consumption and replaces their "Consumption" values with NaN.

- Returns the modified DataFrame.

```

*Static Method (`interpolate_missing_values`):

```

- Sorts the DataFrame by user-related columns.

- Converts the "Consumption" column to numeric, handling errors with NaN.

- Groups the DataFrame by the user and interpolates missing values in the "Consumption" column linearly within each group.

- Returns the DataFrame with interpolated values.

```

*Static Method (`fill_missing_values_with_monthly_mean`):

```

- Sorts the DataFrame by user-related columns.

- Computes monthly means of the "Consumption" column for each user.

- Merges the DataFrame with its monthly means and fills missing values in the "Consumption" column with the corresponding monthly mean.

- Returns the DataFrame with filled missing values.

-

```

*Static Method (`remove_outliers_iqr`):

```

- Uses the Interquartile Range (IQR) method to identify and replace outliers in the "Consumption" column with None.

- Returns the DataFrame with outliers removed.

```

*Static Method (`filter_users`):

```

- Sorts the DataFrame by user-related columns.

- Converts the "Hour" column to numeric if it is not already numeric.

- Groups the DataFrame by user, year, month, and day and checks if all hours from 0 to 23 are present for each group.

- Creates a new DataFrame indicating valid users based on the presence of all hours.

- Filters the original DataFrame based on valid users.

- Prints the number of eliminated users and a success message.

- Returns the filtered DataFrame.

```

*Static Method (`monthly_average_consumption`):

```

- Creates a copy of the input DataFrame.

- Groups the DataFrame by user, year, month, and hour.

- Calculates the monthly average normalized consumption for each hour.

- Returns the DataFrame with an additional "M_consumption" column.

```

*Static Method (`reshape_dataframe`):

```

- Groups the input DataFrame by user, year, month, and hour, calculating the mean of "M_consumption" for each group.

- Returns the reshaped DataFrame.

```

```

*Static Method (`merge_clusters`):

```

- Merges two DataFrames on user, year, month, and hour, adding a "Cluster" column from the smaller DataFrame to the main DataFrame.

- Returns the main DataFrame with the additional "Cluster" column.

```

The `DataPreprocessing` class provides a collection of static methods for various data preprocessing tasks, such as handling negative values, replacing specific daily profiles, interpolating missing values, filling missing values with monthly means, removing outliers, filtering users, calculating monthly averages, reshaping the DataFrame, and merging clusters. Each method serves a specific purpose in preparing and cleaning the input DataFrame for further analysis.

- `export_data.py`: This script defines a class named `Export` that contains two static methods for exporting pandas DataFrames to CSV files.

*Class Definition (`Export`):

```

- The class doesn't have a constructor (`__init__`) as it only contains static methods for exporting data.

```

*Static Method (`export_centroid_csv`):

```

- Takes a pandas DataFrame (`data`) as input.

- Retrieves the directory of the script using `os.path.dirname(__file__)`.

- Defines a directory path for centroids (`"../data/Centroids"`) relative to the script directory.

- Creates the specified directory if it doesn't exist (`os.makedirs(data_dir, exist_ok=True)`).

- Defines the path for the CSV file within the centroids directory (`csv_path = os.path.join(data_dir, "centroid_data.csv")`).

- Exports the input DataFrame to the CSV file without including the index (`data.to_csv(csv_path, index=False)`).

```

*Static Method (`export_output_csv`):

```

- Similar to `export_centroid_csv` but tailored for output data.

- Retrieves the script directory.

- Defines a directory path for output data (`"../data/Output Data"`).

- Creates the specified directory if it doesn't exist.

- Defines the path for the CSV file within the output data directory (`csv_path = os.path.join(data_dir, "Aggregated Data.csv")`).

- Exports the input DataFrame to the CSV file without including the index.

```

The `Export` class provides methods for exporting pandas DataFrames to CSV files. The `export_centroid_csv` method is designed for exporting centroid data, and the `export_output_csv` method is tailored for exporting aggregated output data. Both methods use the `os` module to handle file paths and directory creation.

- `get_features.py`:

*Method (spot_tou):

```

- Merges two DataFrames (`main_df` and `tou_df`) based on the 'Hour' column.

- Extracts unique elements in the "ToU" column and assigns an "Extension" rank based on the count.

- Augments the main DataFrame with the "Extension" information.

```

*Method (get_features):

```

- Computes the sum of 'Consumption' for each unique combination of 'User', 'Year', and 'Month' concerning 'ToU' values.

- Introduces a 'Monthly_consumption' column as the sum of dynamically created columns for each 'ToU'.

- Establishes columns as ratios between the created columns and 'Monthly_consumption'.

-

```

*Method (create_permutation_ratios):

```

- Determines the ratio between all pairs of 'ToU' columns, factoring in the 'Extension' information.

- Integrates these ratio columns into the DataFrame.

```

*Method (get_selected_features_and_cluster):

```

- Selects specific columns, including 'Cluster', 'Monthly_consumption', 'ToU' columns, and dynamically created ratio columns.

- Filters out duplicates and replaces infinite values with NaN.

```

*Method (get_features2):

```

- Select numeric columns (excluding "User", "Year", "Month").

- Generates ratio columns for all feasible combinations of numeric columns.

- Combines the original DataFrame with the new columns.

```

*Method (identify_main_ToU):

```

- Chooses rows with 'Hour' in the range from 10 to 18.

- Identifies the most frequent 'ToU' during the chosen interval.

- Computes the total duration of the interval for the main 'ToU'.

- Computes the extension per unique cluster and introduces new columns ('main ToU', 'Extension').

```

*Method (calculate_sum_column):

```

- Filters the DataFrame for the most frequent 'ToU'.

- Computes the sum of elements in the 'Centroid' column for each unique value in the 'Cluster' column.

- Merges the sum for each cluster into the original DataFrame.

```

*Method (calculate_weight_coefficient):

```

- Computes the 'weight' column as 'Centroid' / (Extension * sum).

```

*Method (numeric_to_words):

```

- Maps numeric values in the 'Cluster' column to their word representations.

```

These methods are designed for specific feature engineering tasks, contributing to a modular and organized approach to data manipulation and analysis.

- `get_file.py:`The `CSVHandler` class facilitates the handling of CSV files within Python, particularly focusing on the efficient loading of data into a Pandas DataFrame. The class is initialized with a specified file path, intending to streamline subsequent operations. It maintains attributes such as `file_path` to store the designated CSV file path and `data` to hold the loaded information. A key method, `load_csv`, attempts to read the CSV file into a Pandas DataFrame. Employing a try-except block ensures graceful handling of potential errors. In the event of a successful load, a confirmation message is displayed. If the file is not found (`FileNotFoundError`), an appropriate error message is printed. For other exceptions, a generic loading error message is presented, including details of the encountered exception. Another method, `get_data`, allows external components to retrieve the loaded DataFrame effortlessly. This provides a straightforward means for users to access and manipulate the loaded data. The `CSVHandler` class serves as a modular and informative tool for working with CSV files, offering clear feedback during the loading process and easy access to the resulting data.

- `get_tou.py:`The `CSVHandler` class provides a streamlined approach for handling CSV files. Upon initialization, it stores the specified file path as `file_path` and initializes the `data` attribute as `None`. The primary functionalities include a method named `load_csv` and another named `get_data`. The `load_csv` method attempts to read the CSV file located at `file_path` into a Pandas DataFrame. In case of a successful load, a confirmation message is printed. If the file is not found (`FileNotFoundError`), an error message is displayed. For other exceptions during loading, a generic error message is presented along with details of the encountered exception. The `get_data` method allows external components to retrieve the loaded DataFrame effortlessly, offering a convenient means for users to access and manipulate the loaded data.

- `plots.py:`The `Plots` class in Python facilitates the creation of informative visualizations using the seaborn and matplotlib libraries. This class contains three static methods, each designed to generate specific types of plots based on input data.

*plot_norm_avg_cons:

```

- This method creates a line plot depicting the normalized average monthly consumption profiles for different users.

- It utilizes seaborn's lineplot, specifying the x-axis as "Hour," y-axis as "M_consumption," and differentiating lines by the "User" parameter.

- The resulting plot is saved as a PNG file named "Normalized Average Monthly Consumption Profiles.png" in the "plots" directory.

```

*plot_cluster_centroids:

```

- This method visualizes the profiles of cluster centroids, showcasing how each cluster's central tendency varies across different hours.

- A FacetGrid is employed to create a grid of subplots, with each subplot representing a distinct cluster.

- The line plot within each subplot displays the centroid values over the hours.

- The resulting plot is saved as "Cluster_Centroids_Profiles.png" in the "plots" directory.

```

*plot_aggregate_loads:

```

- This method generates a set of subplots, each representing a different month, to visualize the distribution of aggregate loads over hours.

- It uses seaborn's boxplot to show the distribution of loads and overlays line plots for each day in the month without dashes.

- The resulting set of subplots is saved collectively as "Aggregate load profiles.png" in the "plots" directory.