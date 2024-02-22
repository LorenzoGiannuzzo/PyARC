import os
from PyARC import PyARC

class ModelManager:
    def __init__(self):
        self.pyarc_instance = PyARC()

    def use_pretrained_model(self):
        self.pyarc_instance.reconstruct_profiles()
        # Aggiungi eventuali logiche aggiuntive specifiche per l'utilizzo del modello pre-allenato

    def train_new_model(self):
        data_file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Input Training Data")
        data_file_path = os.path.join(data_file_directory, "train_data.csv")

        tou_file_directory = os.path.join(os.path.dirname(__file__), "..", "data", "Input Training Data")
        tou_file_path = os.path.join(tou_file_directory, "train_tou.csv")

        self.pyarc_instance.train_model(data_file_path, tou_file_path)
        # Aggiungi eventuali logiche aggiuntive specifiche per l'addestramento di un nuovo modello

    def use_user_trained_model(self):

        self.pyarc_instance.user_trained_model()

def start_program():
    print("Welcome to PyARC! What would you like to do?")
    print("1. Reconstruct Residential Aggregate Electrical Load Profiles using the pre-trained model")
    print("2. Train a new model")
    print("3. Reconstruct Residential Aggregate Electrical Load Profiles using the user-trained model")

    choice = input("Enter the number corresponding to the desired action: ")

    model_manager = ModelManager()

    if choice.strip() == "1":
        model_manager.use_pretrained_model()
    elif choice.strip() == "2":
        model_manager.train_new_model()
    elif choice.strip() == "3":
        model_manager.use_user_trained_model()
    else:
        print("Invalid choice. Please enter a valid number.")

if __name__ == "__main__":
    start_program()




