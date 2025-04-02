import pickle
import logging
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from models.DataPreprocessing import DataPreprocessing
from models.PredictiveModelling import PredictiveModelling
from models.ModelScoringEvaluation import ModelScoringEvaluation
from models.DataVisualization import DataVisualization

from logger_config import setup_logger

class SaveLoadBestModel:
    def __init__(self, best_model, X_test, y_test, best_model_name, filename="best_model.pkl"):
        """
        Initializes the SaveLoadBestModel class.

        Args:
            best_model: The best trained model.
            X_test: Test features.
            y_test: Test target.
            best_model_name: Name of the best model.
            filename (str): Filename to save/load the model.
        """
        self.logger = setup_logger(__name__)
        self.best_model = best_model
        self.X_test = X_test
        self.y_test = y_test
        self.best_model_name = best_model_name
        self.filename = filename

    def save_model(self):
        """Saves the best model to a pickle file."""
        try:
            with open(self.filename, 'wb') as file:
                pickle.dump(self.best_model, file)
            self.logger.info(f"Best model '{self.best_model_name}' saved to '{self.filename}'")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

    def load_model(self):
        """Loads the best model from a pickle file."""
        try:
            with open(self.filename, 'rb') as file:
                loaded_model = pickle.load(file)
            self.logger.info(f"Model loaded from '{self.filename}'")
            return loaded_model
        except FileNotFoundError:
            self.logger.error(f"File '{self.filename}' not found.")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None

    def is_model_close(self, model1, model2):
        """
        Verifies if two models are close (same).

        Args:
            model1: The first model.
            model2: The second model.

        Returns:
            bool: True if models are close, False otherwise.
        """
        if model1 is None or model2 is None:
            return False

        try:
            # Compare model parameters if possible
            if hasattr(model1, 'get_params') and hasattr(model2, 'get_params'):
                params1 = model1.get_params()
                params2 = model2.get_params()
                if params1 != params2:
                    return False

            # Compare predictions on the test set
            y_pred1 = model1.predict(self.X_test)
            y_pred2 = model2.predict(self.X_test)

            if not np.array_equal(y_pred1, y_pred2):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            return False

    def evaluate_loaded_model(self, loaded_model):
        """Evaluates the loaded model on the test set."""
        if loaded_model is None:
            self.logger.error("No model to evaluate.")
            return

        try:
            y_pred = loaded_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, loaded_model.predict_proba(self.X_test)[:, 1])

            self.logger.info(f"Evaluation of loaded model '{self.best_model_name}':")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"ROC AUC: {roc_auc:.4f}")
            self.logger.info(f"Classification Report:\n{report}")
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")

    def run(self):
        """Runs the save, load, and evaluate methods."""
        self.save_model()
        loaded_model = self.load_model()
        if loaded_model:
            if self.is_model_close(self.best_model, loaded_model):
                self.logger.info("Loaded model is identical to the saved model.")
                self.evaluate_loaded_model(loaded_model)
            else:
                self.logger.warning("Loaded model is different from the saved model.")


data_path = '../data/total_ksi.csv'
data_preprocessor = DataPreprocessing(data_path)
modelling = PredictiveModelling(data_preprocessor)
tuned_models, X_test, y_test, best_params, training_times = modelling.run() #get training times.
evaluation = ModelScoringEvaluation(tuned_models, X_test, y_test, modelling.X_train, modelling.y_train)
model_results, best_model_name, best_model_roc_auc = evaluation.run(best_params)

# Create SaveLoadBestModel instance and run
save_load = SaveLoadBestModel(evaluation.tuned_models[best_model_name], X_test, y_test, best_model_name)
save_load.run()