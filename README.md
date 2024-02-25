
## PyARC - Python Algorythm for Residential load profiles reConstruction (Python 3.12.1)

*Lorenzo Giannuzzo (a,b), Francesco Demetrio Minuto (a,b), Daniele Salvatore Schiera (a,b), Andrea Lanzini (a,b). 

*(a) Energy Center Lab, Polytechnic of Turin, via Paolo Borsellino 38/16, 10152, Turin, Italy* 
*(b) Department of Energy (DENERG), Polytechnic of Turin, Corso Duca degli Abruzzi 24, 10129, Turin, Italy*

PyARC Software - Python Algorythm for Residential load profiles reConstruction ==(Python 3.12.1).==
## Overview:

PyARC is a Python-developed software designed for the reconstruction of residential aggregate electrical load profiles. It leverages an algorithm for Association Rule Mining to model complex relationships between Time-of-Use (ToU) data and electrical consumption profiles. The software uses a non-intrusive machine learning methodology that can be used to generate residential electrical consumption profiles at an hourly resolution level using only monthly consumption data (i.e., billed energy). The methodology is mainly composed by three phases: first, identifying the typical load patterns of residential users through kMeans clustering, then implementing a Random Forest algorithm, based on monthly energy bills, to identify typical load patterns and, finally, reconstructing the hourly electrical load profile through a data-driven rescaling procedure.

## Input:

**Model Training:**

- Hourly eletrical consumption measures expressed in [kWh] in CSV format used to train the `User-trained Model` must be contained in the `Input Training Data` folder. (Data structure and requirements are specified in **Directory Structure/data** section).

- Time of Use (ToU) data in CSV format used to train the `User-trained Model` must be contained in the `Input Training Data` folder. (Data structure and requirements are specified in **Directory Structure/data** section).

 **Reconstruction using Pre-trained Model:**

- Monthly electrical energy bills expressed in [kWh] in CSV format used to reconstruct the residential aggregate load profile of users must be contained in the `Input Data` folder. (Data structure and requirements are specified in **Directory Structure/data** section).

- Pre-trained classification model is required to perform the load profile reconstruction algorythm. The pre-trained model is already placed and available in the `Pre-trained Model` folder.

 - Time of Use (ToU) data in CSV format is required and already placed in the `Input Data` folder. (Data structure and requirements are specified in **Directory Structure/data** section).

 **Reconstruction using User-trained Model:**

- Monthly electrical energy bills expressed in [kWh] in CSV format used to reconstruct the residential aggregate load profile of users must be contained in the `Input Data` folder. (Data structure and requirements are specified in **Directory Structure/data** section).

- User-trained classification model is required to perform the load profile reconstruction algorythm. The User-trained model must be trained before performing the reconstruction process using the `User-trained Model`. In order to do this, it's necessary to execute the **Model Training** process (see **Input/Model Training** section). Once exectued, the model will be saved in the `User-trained Model` folder, and used to reconstruct the residential aggregate load profile.

 - Time of Use (ToU) data in CSV format is required and must be in the `Input Data` folder. (Data structure and requirements are specified in **Directory Structure/data** section).

## PyARC Workflow:

- **Model Training:**
    
    - Execute the `train_model` method in `PyARC.py` to train a new PyARC model. This method train the classification model used to reconstruct the aggregate eletrical load profiles of residential users using generic input data contained in the `Input Training Data` folder (electrical energy measures on an hourly scale and Time of Use subdivision)

- **Reconstruction using Pre-trained Model:**
    
    - Execute the `reconstruct_profiles` method in `PyARC.py` to reconstruct profiles using the pre-trained model contained in the `Pre-trained Model` folder and the input data contained in the `Input Data` folder.
    - The PyARC instance is initialized, and the pre-trained model is used for profile reconstruction.
    
- **Reconstruction using User-trained Model:**
    
    - Execute the `user_trained_model` method in `PyARC.py` to reconstruct profiles using a user-trained model contained in `User-trained Model` folder. If no model is inside the folder, the **Model Training** features has to be executed with the required `Input Data`.
    - The PyARC instance is initialized, and the user-trained model is utilized for profile reconstruction.

## How to Run:

1. Install necessary lirbaries typing in the terminal: ==`pip install -r requirements.txt`== 
2. Ensure the necessary libraries are installed (required libraries are defined in the `requirement.txt` file)
3. Run `main.py`.
4. Choose an option (1, 2, or 3) based on the desired action:
    - 1: Reconstruct profiles using the pre-trained model.
    - 2: Train a new PyARC model.
    - 3: Reconstruct profiles using a user-trained model.


## Directory Structure:

- **data:** Contains input data required to run the code in CSV format.
    
    - _Input Training Data:_ This folder contains the required data to train the user-trained model, which can be used to reconstruct the eletrical load profile of residential aggregates.
    
        - `train_data.csv`: Training data file in CSV format. This data must be structured as follow:
```    
"User","Year","Month","Day","Hour","Consumption"
"MAC000016",2012,1,1,0,0.0275
"MAC000016",2012,1,1,1,0.0735
"MAC000016",2012,1,1,2,0.0395
"MAC000016",2012,1,1,3,0.0305
```
            where:
            - "User": is a char/string which contains usernames;
            - "Year": is a numerical value which respresent the year when eletrical     energy was measured;
            - "Month": is a numerical value which represent the months when electrical energy was measured;
            - "Day": is a numerical value which represent the number of days when electrical energy was measured;
            - "Hour": is a numerical value which represent the hours when eletrical energy was measured;
            - -"Consumption": contains electrical energy values expressed in [kWh].
        
        - `train_tou.csv`: Training Time of Use (ToU) data in CSV format. This file represent the specific Time of Use of a weekday related to the train data (train_data.csv). The ToU must be structured as follow:
```
  "Hour","ToU"  
0,"F3"  
1,"F3"  
2,"F3"  
3,"F3"  
4,"F3"  
5,"F3"  
6,"F3"  
7,"F2"  
8,"F2"  
9,"F2"  
10,"F2"  
11,"F1"  
12,"F1"  
13,"F1"  
14,"F2"  
15,"F2"  
16,"F1"  
17,"F1"  
18,"F1"  
19,"F2"  
20,"F2"  
21,"F2"  
22,"F2"  
23,"F3"      
```
            where:
            - "Hour": is a numerical value that represent the Hour is which the ToU is divided;
            - "ToU": contains the Time of Use subgroup names expressed as char/string.

 - *Input Data:* This folder contains the required data to use the user-trained  or the pre-trained model, which can be used to reconstruct the eletrical load profile of residential aggregates.

        - `data.csv`: Input data file in CSV format. This data must be structured as follow:
        - 
    
- **PyARC:** Folder containing the scripts of the algorithms and related functionalities.
    
- **main.py:** Main script for user interaction and execution of PyARC functionalities.
    


## Notes:

- The `PyARC` package contains the core implementation of the Apriori algorithm for Association Rule Mining.
- Data preprocessing is performed using functions in `data_preprocessing.py`.
- Various feature extraction methods are defined in `features.py` and `get_features.py`.
- Export functionalities for centroids and output data are managed by `export.py`.
- The main interaction and execution of PyARC functionalities occur in `main.py`.

