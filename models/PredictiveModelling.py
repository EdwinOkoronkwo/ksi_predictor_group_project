import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier


from models.DataVisualization import DataVisualization
from models.DataPreprocessing import DataPreprocessing
from logger_config import setup_logger


import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from logger_config import setup_logger


class PredictiveModelling:
    def __init__(self, data_preprocessor, topN=None):
        self.logger = setup_logger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.data_preprocessor = data_preprocessor
        self.data_preprocessor.run()
        self.X_train, self.X_test, self.y_train, self.y_test = data_preprocessor.get_processed_data()
        self.preprocessing_feature_importance = self.data_preprocessor.get_feature_importance()
        self.topN = topN
        self.top_features = self.select_top_features() if topN is not None else self.X_train.columns.tolist()
        self.X_train = self.X_train[self.top_features]
        self.X_test = self.X_test[self.top_features]
        self.models = {
            "LogReg": LogisticRegression(random_state=42),
            "DTC": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(random_state=42, probability=True),
            "Forest": RandomForestClassifier(random_state=42),
            "ANN": MLPClassifier(random_state=42),
            "NBayes": GaussianNB(),
            "KNN": KNeighborsClassifier(),
            "Voting": None,
            "Stacking": None
        }
        self.tuned_models = {}
        self.visualizer = DataVisualization(self.X_test)


    def select_top_features(self):
        """Selects the top N features based on preprocessing feature importance."""
        sorted_features = self.preprocessing_feature_importance.sort_values(ascending=False)
        top_features = sorted_features.index[:self.topN].tolist()
        return top_features

    def train_and_tune_all_models(self):
        """Trains and tunes all models, extracting best parameters."""
        best_params = {}
        training_times = {}  # Dictionary to store training times
        for model_name, model in self.models.items():
            self.logger.info(f"Training and tuning {model_name}...")
            start_time = time.time()
            if model_name != "Voting" and model_name != "Stacking":
                tuned_model = self.train_and_tune_model(model, model_name)
                self.tuned_models[model_name] = tuned_model
                if hasattr(tuned_model, 'best_params_'):
                    best_params[model_name] = tuned_model.best_params_
            elif model_name == "Voting":
                self.tuned_models[model_name] = self.create_voting()
            elif model_name == "Stacking":
                self.tuned_models[model_name] = self.create_stacking()
            end_time = time.time()
            training_times[model_name] = end_time - start_time
            self.logger.info(f"{model_name} training time: {training_times[model_name]:.2f} seconds")
        return best_params, training_times  # Return training times



    def train_and_tune_model(self, model, model_name):
        """Trains and tunes a single model based on its name."""
        if model_name == "LogReg":
            return self.tune_logistic_regression(model)
        elif model_name == "DTC":
            return self.tune_decision_tree(model)
        elif model_name == "SVM":
            return self.tune_svm(model)
        elif model_name == "Forest":
            return self.tune_random_forest(model)
        elif model_name == "ANN":
            return self.tune_neural_network(model)
        elif model_name == "NBayes":
            return self.tune_naive_bayes(model)
        elif model_name == "KNN":
            return self.tune_knn(model)
        else:
            return model


    def tune_logistic_regression(self, model):
        param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'max_iter': [100, 200, 500, 1000]}
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=0, return_train_score=True)
        grid_search.fit(self.X_train, self.y_train)
        self.logger.info(f"Best parameters for Logistic Regression: {grid_search.best_params_}")
        self.logger.info(f"Cross Validation Results: {grid_search.cv_results_['mean_test_score']}")
        best_model = grid_search.best_estimator_
        feature_importance = pd.Series(best_model.coef_[0], index=self.X_train.columns)
        self.logger.info("Feature Importance in Predictive Modelling (Logistic Regression):")
        self.logger.info(feature_importance.to_string())
        self.visualizer.plot_feature_importance(feature_importance, title="Feature Importance in Predictive Modelling (Logistic Regression)")
        return best_model

    def tune_decision_tree(self, model):
        param_grid = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        self.logger.info(f"Best parameters for Decision Tree: {grid_search.best_params_}")
        self.logger.info(f"Cross Validation Results: {grid_search.cv_results_['mean_test_score']}")
        best_model = grid_search.best_estimator_
        feature_importance = pd.Series(best_model.feature_importances_, index=self.X_train.columns)
        self.logger.info("Feature Importance in Predictive Modelling (Decision Tree):")
        self.logger.info(feature_importance.to_string())
        self.visualizer.plot_feature_importance(feature_importance, title="Feature Importance in Predictive Modelling (Decision Tree)")
        return best_model

    def tune_svm(self, model):
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 0.1, 1]}
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        self.logger.info(f"Best parameters for SVM: {grid_search.best_params_}")
        self.logger.info(f"Cross Validation Results: {grid_search.cv_results_['mean_test_score']}")
        return grid_search.best_estimator_

    def tune_random_forest(self, model):
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        self.logger.info(f"Best parameters for Random Forest: {grid_search.best_params_}")
        self.logger.info(f"Cross Validation Results: {grid_search.cv_results_['mean_test_score']}")
        best_model = grid_search.best_estimator_
        feature_importance = pd.Series(best_model.feature_importances_, index=self.X_train.columns)
        self.logger.info("Feature Importance in Predictive Modelling (Random Forest):")
        self.logger.info(feature_importance.to_string())
        self.visualizer.plot_feature_importance(feature_importance, title="Feature Importance in Predictive Modelling (Random Forest)")
        return best_model

    def tune_neural_network(self, model):
        param_grid = {
            'hidden_layer_sizes': [(64, 64,), (128, 128), (256, 256)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        self.logger.info(f"Best parameters for Neural Network: {grid_search.best_params_}")
        self.logger.info(f"Cross Validation Results: {grid_search.cv_results_['mean_test_score']}")
        return grid_search.best_estimator_

    def tune_knn(self, model):
        """Tunes the KNN model and plots optimum k."""
        param_grid = {'n_neighbors': range(1, 21)}
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        self.logger.info(f"Best parameters for KNN: {grid_search.best_params_}")
        self.logger.info(f"Cross Validation Results: {grid_search.cv_results_['mean_test_score']}")
        # Plotting optimum k
        self.visualizer.plot_knn_optimum_k(grid_search.cv_results_['mean_test_score'], param_grid['n_neighbors'])
        return grid_search.best_estimator_

    def tune_naive_bayes(self, model):
        """Tunes the Naive Bayes model."""
        param_grid = {
            'var_smoothing': [ 1e-6, 1e-5, 1e-4, 1e-3, 0.005, 1e-2, 1e-1]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        self.logger.info(f"Best parameters for Naive Bayes: {grid_search.best_params_}")
        self.logger.info(f"Cross Validation Results: {grid_search.cv_results_['mean_test_score']}")

        return grid_search.best_estimator_

    def create_voting(self):
        """Creates the voting ensemble model using VotingClassifier."""
        estimators = [(name, model) for name, model in self.tuned_models.items() if
                      name not in ["Voting", "Stacking", "ANN"]]  # removed ensemble, added voting
        voting_model = VotingClassifier(estimators=estimators, voting='soft')
        voting_model.fit(self.X_train, self.y_train)
        return voting_model

    def create_stacking(self):
        """Creates the stacking model using StackingClassifier with RandomForest as blender."""
        estimators = [(name, model) for name, model in self.tuned_models.items() if
                      name not in ["Voting", "Stacking", "ANN"]]

        # Define the RandomForest blender with the specified hyperparameters
        rf_blender = RandomForestClassifier(
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=200,
            random_state=42
        )

        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=rf_blender
        )
        stacking_model.fit(self.X_train, self.y_train)
        return stacking_model

    def plot_training_times(self, training_times):
        """Creates a bar plot of training times."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(training_times.keys()), y=list(training_times.values()))
        plt.title("Model Training Times")
        plt.xlabel("Model")
        plt.ylabel("Training Time (seconds)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def run(self):
        self.logger.info("Starting model training and tuning...")
        best_params, training_times = self.train_and_tune_all_models()
        self.logger.info("Model training and tuning complete.")
        self.visualizer.plot_training_times(training_times)  # Plot training times.
        return self.tuned_models, self.X_test, self.y_test, best_params, training_times


    # def run(self):
    #     self.logger.info("Starting model training and tuning...")
    #     best_params = self.train_and_tune_all_models()
    #     self.logger.info("Model training and tuning complete.")
    #     return self.tuned_models, self.X_test, self.y_test, best_params


# data_path = '../data/total_ksi.csv'
# data_preprocessor = DataPreprocessing(data_path)
# visualizer = DataVisualization(pd.DataFrame(data_path)) # Pass the needed data
# modelling = PredictiveModelling(data_preprocessor, visualizer, topN=15)
# tuned_models, X_test, y_test, best_params = modelling.run()
