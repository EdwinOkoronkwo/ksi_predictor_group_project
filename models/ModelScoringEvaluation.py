import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix,
                             roc_curve, auc)

from models.DataPreprocessing import DataPreprocessing
from models.PredictiveModelling import PredictiveModelling
from models.DataVisualization import DataVisualization
import logging
from logger_config import setup_logger

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

from logger_config import setup_logger



class ModelScoringEvaluation:
    def __init__(self, tuned_models, X_test, y_test, X_train, y_train):
        """
        Initializes the ModelScoringEvaluation class.

        Args:
            tuned_models (dict): Dictionary of tuned models.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
        """
        self.logger = setup_logger(__name__)
        self.tuned_models = tuned_models
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.model_results = pd.DataFrame()
        self.visualizer = DataVisualization(pd.DataFrame(X_test))

    def evaluate_model(self, model, model_name):
        """
        Evaluates a single model and returns test and train accuracies.

        Args:
            model: The model to evaluate.
            model_name (str): Name of the model.

        Returns:
            tuple: Test accuracy and train accuracy.
        """
        y_pred = model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        self.visualizer.plot_confusion_matrix(cm, model_name)
        fpr, tpr, thresholds = roc_curve(self.y_test, model.predict_proba(self.X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        y_train_pred = model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)

        new_row = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [test_accuracy],
            'Precision': [report['weighted avg']['precision']],
            'Recall': [report['weighted avg']['recall']],
            'F1-Score': [report['weighted avg']['f1-score']],
            'ROC AUC': [roc_auc],
            'Train Accuracy': [train_accuracy],
        })

        self.model_results = pd.concat([self.model_results, new_row], ignore_index=True)
        self.visualizer.plot_roc_curve(fpr, tpr, roc_auc, model_name)

        return test_accuracy, train_accuracy

    def evaluate_all_models(self):
        """
        Evaluates all tuned models and returns a dictionary of test accuracies.

        Returns:
            dict: Dictionary of test accuracies.
        """
        accuracy_scores = {}
        for model_name, model in self.tuned_models.items():
            self.logger.info(f"Evaluating {model_name}...")
            test_accuracy, train_accuracy = self.evaluate_model(model, model_name)
            accuracy_scores[model_name] = test_accuracy
            self.visualizer.plot_train_test_metrics(train_accuracy, test_accuracy, model_name)
        return accuracy_scores

    def recommend_best_model(self):
        """
        Recommends the best model based on ROC AUC.

        Returns:
            str: Name of the best model.
        """
        best_model_row = self.model_results.loc[self.model_results['ROC AUC'].idxmax()]
        best_model_name = best_model_row['Model']
        best_roc_auc = best_model_row['ROC AUC']

        self.logger.info(f"Recommended best model: {best_model_name} (ROC AUC: {best_roc_auc:.3f})")
        print(f"Recommended best model: {best_model_name} (ROC AUC: {best_roc_auc:.3f})")
        return best_model_name

    def get_model_results(self):
        """
        Returns the model evaluation results.

        Returns:
            pd.DataFrame: DataFrame containing model evaluation results.
        """
        return self.model_results

    def run(self, best_params):
        """
        Runs the model evaluation process.

        Args:
            best_params (dict): Dictionary of best parameters for each model.

        Returns:
            tuple: Model results DataFrame, best model name, and best model ROC AUC.
        """
        self.logger.info("Starting model evaluation...")
        accuracy_scores = self.evaluate_all_models()
        best_model_name = self.recommend_best_model()
        best_model_row = self.model_results[self.model_results['Model'] == best_model_name].iloc[0]
        self.logger.info("Model evaluation complete.")
        print("Model Results:")
        print(self.model_results)
        print("\nBest Model:")
        print(best_model_name)
        print("\nBest Model Metrics:")
        print(best_model_row)
        print("\nBest Model Parameters:")
        print(best_params.get(best_model_name, "Parameters not found"))
        self.visualizer.plot_model_comparison(accuracy_scores)
        self.visualizer.plot_roc_auc_comparison(self.model_results)
        return self.model_results, best_model_name, best_model_row['ROC AUC']

# # Example Usage (replace with your actual data and pipeline)
# data_path = '../data/total_ksi.csv'
# data_preprocessor = DataPreprocessing(data_path)
# modelling = PredictiveModelling(data_preprocessor)
# tuned_models, X_test, y_test, best_params = modelling.run()
# evaluation = ModelScoringEvaluation(tuned_models, X_test, y_test, modelling.X_train, modelling.y_train)
# model_results, best_model_name, best_model_roc_auc = evaluation.run(best_params)


# class ModelScoringEvaluation:
#     def __init__(self, tuned_models, X_test, y_test, X_train, y_train):
#         self.logger = setup_logger(__name__)
#         self.tuned_models = tuned_models
#         self.X_test = X_test
#         self.y_test = y_test
#         self.X_train = X_train
#         self.y_train = y_train
#         self.model_results = pd.DataFrame()
#         self.visualizer = DataVisualization(pd.DataFrame(X_test))
#
#     def evaluate_model(self, model, model_name):
#         y_pred = model.predict(self.X_test)
#         test_accuracy = accuracy_score(self.y_test, y_pred)
#         report = classification_report(self.y_test, y_pred, output_dict=True)
#         cm = confusion_matrix(self.y_test, y_pred)
#         self.visualizer.plot_confusion_matrix(cm, model_name)
#         fpr, tpr, thresholds = roc_curve(self.y_test, model.predict_proba(self.X_test)[:, 1])
#         roc_auc = auc(fpr, tpr)
#
#         y_train_pred = model.predict(self.X_train)
#         train_accuracy = accuracy_score(self.y_train, y_train_pred)
#
#         new_row = pd.DataFrame({
#             'Model': [model_name],
#             'Accuracy': [test_accuracy],
#             'Precision': [report['weighted avg']['precision']],
#             'Recall': [report['weighted avg']['recall']],
#             'F1-Score': [report['weighted avg']['f1-score']],
#             'ROC AUC': [roc_auc],
#             'Train Accuracy': [train_accuracy],
#         })
#
#         self.model_results = pd.concat([self.model_results, new_row], ignore_index=True)
#         self.visualizer.plot_roc_curve(fpr, tpr, roc_auc, model_name)
#
#         return test_accuracy, train_accuracy
#
#     def evaluate_all_models(self):
#         accuracy_scores = {}
#         for model_name, model in self.tuned_models.items():
#             self.logger.info(f"Evaluating {model_name}...")
#             test_accuracy, train_accuracy = self.evaluate_model(model, model_name)
#             accuracy_scores[model_name] = test_accuracy
#             self.visualizer.plot_train_test_metrics(train_accuracy, test_accuracy, model_name) #call the DataVisualization plot.
#         return accuracy_scores
#
#     def recommend_best_model(self):
#         best_model_row = self.model_results.loc[self.model_results['ROC AUC'].idxmax()]
#         best_model_name = best_model_row['Model']
#         best_roc_auc = best_model_row['ROC AUC']
#
#         self.logger.info(f"Recommended best model: {best_model_name} (ROC AUC: {best_roc_auc:.3f})")
#         print(f"Recommended best model: {best_model_name} (ROC AUC: {best_roc_auc:.3f})")  # print to console as well.
#         return best_model_name
#
#     def get_model_results(self):
#         return self.model_results
#
#     def run(self, best_params):
#         self.logger.info("Starting model evaluation...")
#         accuracy_scores = self.evaluate_all_models()
#         best_model_name = self.recommend_best_model()
#         best_model_row = self.model_results[self.model_results['Model'] == best_model_name].iloc[0]
#         self.logger.info("Model evaluation complete.")
#         print("Model Results:")
#         print(self.model_results)
#         print("\nBest Model:")
#         print(best_model_name)
#         print("\nBest Model Metrics:")
#         print(best_model_row)
#         print("\nBest Model Parameters:")
#         print(best_params.get(best_model_name, "Parameters not found"))
#         self.visualizer.plot_model_comparison(accuracy_scores)
#         self.visualizer.plot_roc_auc_comparison(self.model_results)
#         return self.model_results, best_model_name, best_model_row['ROC AUC']
#
# data_path = '../data/total_ksi.csv'
# data_preprocessor = DataPreprocessing(data_path)
# modelling = PredictiveModelling(data_preprocessor)
# tuned_models, X_test, y_test, best_params = modelling.run()
# evaluation = ModelScoringEvaluation(tuned_models, X_test, y_test, modelling.X_train, modelling.y_train)
# evaluation.run(best_params)


