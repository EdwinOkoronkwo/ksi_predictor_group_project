import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE  # For handling imbalanced classes
from models.DataExploration import DataExploration
from models.CombinedFeaturesAdder import CombinedFeaturesAdder
from models.DataVisualization import DataVisualization
import copy
import logging
from logger_config import setup_logger #Import logger
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier

import pandas as pd
import logging
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from models.DataExploration import DataExploration
from logger_config import setup_logger


class DataPreprocessing:
    def __init__(self, data_path, drop_threshold=0.5, sample_reduction=0.1):
        """
        Initializes the DataModelling object.

        Args:
            data_path (str): Path to the CSV file containing the dataset.
            drop_threshold (float): Threshold for dropping columns with missing values.
            sample_reduction (float): Fraction of sample to use. 1.0 means no reduction.
        """
        # Setup logging
        self.logger = setup_logger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        data_explorer = DataExploration(data_path)
        self.data = data_explorer.data
        self.target_variable = 'ACCLASS'
        self.drop_threshold = drop_threshold
        self.sample_reduction = sample_reduction  # Initialize sample_reduction here
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.visualizer = DataVisualization(self.data)

    def sample_data(self, reduction_factor=None):
        """Samples the data with optional reduction factor."""
        self.logger.info("--- Before sample_data ---")
        self.logger.info(f"Data shape: {self.data.shape}")
        self.logger.info(f"Number of columns: {len(self.data.columns)}")

        if reduction_factor is None:
            reduction_factor = self.sample_reduction
        self.data = self.data.sample(frac=reduction_factor, random_state=42)
        self.data = self.data.reset_index(drop=True)

        self.logger.info("--- After sample_data ---")
        self.logger.info(f"Data shape: {self.data.shape}")
        self.logger.info(f"Number of columns: {len(self.data.columns)}")


    def _handle_missing_data(self):
        """Handles missing data."""
        self.logger.info("--- Before _handle_missing_data ---")
        self.logger.info(f"Data shape: {self.data.shape}")
        self.logger.info(f"Number of columns: {len(self.data.columns)}")

        # Drop columns with more than drop_threshold missing values
        na_percentages = self.data.isna().sum() / len(self.data)
        cols_to_drop = na_percentages[na_percentages > self.drop_threshold].index
        self.data = self.data.drop(columns=cols_to_drop)

        # Separate numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=np.number).columns
        categorical_cols = self.data.select_dtypes(exclude=np.number).columns

        # Impute missing values
        for col in numerical_cols:
            self.data[col] = self.data[col].fillna(self.data[col].median())
        for col in categorical_cols:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

        self.logger.info("--- After _handle_missing_data ---")
        self.logger.info(f"Data shape: {self.data.shape}")
        self.logger.info(f"Number of columns: {len(self.data.columns)}")


    def _split_data(self, X, y):
            """Splits data into train and test sets."""
            self.logger.info("--- Before _split_data ---")
            self.logger.info(f"X shape: {X.shape}")
            self.logger.info(f"Number of columns in X: {len(X.columns)}")
            self.logger.info(f"y shape: {y.shape}")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.logger.info("--- After _split_data ---")
            self.logger.info(f"X_train shape: {self.X_train.shape}")
            self.logger.info(f"X_test shape: {self.X_test.shape}")
            self.logger.info(f"y_train shape: {self.y_train.shape}")
            self.logger.info(f"y_test shape: {self.y_test.shape}")
            self.logger.info(f"Number of columns in X_train: {len(self.X_train.columns)}")
            self.logger.info(f"Number of columns in X_test: {len(self.X_test.columns)}")


    def _manage_imbalanced_classes(self):
        """Handles imbalanced classes using SMOTE."""
        self.logger.info("Class distribution before SMOTE:")
        self.logger.info(pd.Series(self.y_train).value_counts())
        self.visualizer.plot_class_distribution(self.y_train, 'Class Distribution Before SMOTE')  # Plot before SMOTE
        # Create copies of X_train and y_train
        X_train_original = copy.deepcopy(self.X_train)
        y_train_original = copy.deepcopy(self.y_train)
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        self.logger.info("Class distribution after SMOTE:")
        self.logger.info(pd.Series(self.y_train).value_counts())
        self.visualizer.plot_class_distribution(self.y_train, 'Class Distribution After SMOTE')  # Plot after SMOTE

    def _feature_selection(self):
        """Performs feature selection based on correlation and domain knowledge."""
        self.logger.info("--- Before _feature_selection ---")
        self.logger.info(f"Data shape: {self.data.shape}")
        self.logger.info(f"Number of columns: {len(self.data.columns)}")

        self._drop_high_missing_cols()
        self._drop_irrelevant_cols()
        self._drop_highly_correlated_cols()

        self.logger.info("--- After _feature_selection ---")
        self.logger.info(f"Data shape: {self.data.shape}")
        self.logger.info(f"Number of columns: {len(self.data.columns)}")

    def _drop_high_missing_cols(self):
        """Drops columns with a high percentage of missing values."""
        self.logger.info("--- Dropping columns with high missing values ---")
        missing_percentage = (self.data.isnull().sum() / len(self.data)) * 100
        cols_to_drop = missing_percentage[missing_percentage > self.drop_threshold * 100].index
        self.logger.info(f"Columns to drop: {list(cols_to_drop)}")
        self.data.drop(columns=cols_to_drop, inplace=True)
        self.logger.info(f"Number of columns after dropping: {len(self.data.columns)}")

    def _drop_irrelevant_cols(self):
        """
        Drops irrelevant columns based on domain knowledge and to prevent target leakage/overfitting.

        Columns dropped:
        - 'OBJECTID': Unique identifier, not relevant for prediction.
        - 'INDEX': Index column, not relevant for prediction.
        - 'INJURY': Directly related to the target variable 'ACCLASS', causing target leakage.
                    'INJURY FATAL' is the target, and 'INJURY MAJOR' is a strong indicator of 'FATAL'.
                    Therefore, including it leads to overfitting and unrealistic model performance.
        - 'ACCNUM': Accident number, a unique identifier with no predictive value.
        - 'DATE': High cardinality feature that is unlikely to generalize well.
                  Also, it may contain temporal patterns that are specific to the training set.
                  If the date is used as is, it can lead to overfitting.
                  If the date is used to create features such as day of week, month, etc, those features should be created elsewhere, and not in this function.
        - 'STREET1', 'STREET2': High cardinality categorical features that can lead to overfitting.
                             They may contain local patterns that are specific to the training set and not generalize well.
                             They can also act as proxies for location, which can lead to data leakage if not handled carefully.
                             These features can also cause issues with some models, such as linear models, that assume a linear relationship between features and the target variable.
        """
        self.logger.info("--- Dropping irrelevant columns ---")
        irrelevant_cols = ['OBJECTID', 'INDEX', 'ACCNUM', 'INJURY', 'DATE', 'STREET1', 'STREET2']
        self.logger.info(f"Columns to drop: {irrelevant_cols}")
        self.data.drop(columns=irrelevant_cols, inplace=True, errors='ignore')
        self.logger.info(f"Number of columns after dropping: {len(self.data.columns)}")

    def _drop_highly_correlated_cols(self):
        """Drops highly correlated features."""
        corr_matrix = self.data.corr()
        highly_correlated_features = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    colname = corr_matrix.columns[i]
                    highly_correlated_features.add(colname)

        self.logger.info(f"Columns removed due to high correlation: {highly_correlated_features}")
        self.data.drop(columns=highly_correlated_features, inplace=True)


    def create_pipeline(self):
        """Creates a pipeline for data transformations."""
        # Separate numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns

        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('combiner', CombinedFeaturesAdder()),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Full pipeline (apply pipelines to respective columns)
        full_pipeline = ColumnTransformer([
            ('numerical', numerical_pipeline, numerical_cols),  # Apply numerical pipeline to numerical columns
            ('categorical', categorical_pipeline, categorical_cols)  # Apply categorical pipeline to categorical columns
        ])

        return full_pipeline

    def get_processed_data(self):
        """Returns the processed data."""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def _print_pre_transformation_info(self):
        """Prints data info before transformation."""
        self.logger.info("Missing data handled.")
        self.logger.info("Feature selection completed.")
        self.logger.info(f"Columns after feature selection: {self.data.columns}")
        self.logger.info("\nData Info before transformation:")
        self.logger.info(self.data.info())

    def _print_post_transformation_info(self):
        """Prints data info after transformation."""
        self.logger.info(f"Columns before pipeline: {self.data.columns}")
        self.logger.info(f"Columns after pipeline: {self.data.columns}")
        self.logger.info("\nData Info after transformation:")
        self.logger.info(self.data.info())

    def _print_processed_data_info(self):
        """Prints info for preprocessed training and test data."""
        self.logger.info("Data split into train and test sets.")
        self.logger.info("SMOTE applied to handle imbalanced classes.")
        self.logger.info("\nPreprocessed Training Data:")
        self.logger.info(f"Columns: {self.X_train.columns.tolist()}")
        self.logger.info(f"Training Data Shape: {self.X_train.shape}")
        self.logger.info(f"Training Data dtypes:\n{self.X_train.dtypes}")  # Log the dtypes
        self.logger.info("\nPreprocessed Test Data:")
        self.logger.info(f"Columns: {self.X_test.columns.tolist()}")
        self.logger.info(f"Test Data Shape: {self.X_test.shape}")
        self.logger.info(f"Test Data dtypes:\n{self.X_test.dtypes}")  # Log the dtyp

    def _print_final_data_info(self, X_train, X_test, y_train, y_test):
        """Prints final processed data info."""
        self.logger.info("\nFinal Processed Data Info:")
        self.logger.info(
            f"X_train shape: {X_train.shape}, dtype: {X_train.dtypes.iloc[0] if not X_train.empty else None}")
        self.logger.info(f"X_test shape: {X_test.shape}, dtype: {X_test.dtypes.iloc[0] if not X_test.empty else None}")
        self.logger.info(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
        self.logger.info(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")
        self.logger.info(f"y_train value counts:\n{pd.Series(y_train).value_counts()}")
        self.logger.info(f"y_test value counts:\n{pd.Series(y_test).value_counts()}")

    def convert_target_to_int32(self, y_train, y_test):
        """Converts y_train and y_test to int32."""
        y_train = y_train.astype('int32')
        y_test = y_test.astype('int32')
        return y_train, y_test

    def train_random_forest(self):
        """Trains a RandomForest model and extracts feature importance."""
        self.logger.info("Training RandomForest model...")
        if self.X_train is None or self.y_train is None:
            raise ValueError("X_train or y_train is not set. Run 'run' method first.")
        model = RandomForestClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        self.feature_importance = pd.Series(model.feature_importances_, index=self.X_train.columns)
        self.logger.info("RandomForest model trained. Feature importance extracted.")

    def get_feature_importance(self):
        """Returns the feature importance calculated by train_random_forest."""
        return self.feature_importance

    def run(self):
        """Runs all data preprocessing and training steps."""
        self.logger.info("Starting data preprocessing...")
        self.logger.info("Starting Data Preprocessing...")
        self.sample_data()
        self._handle_missing_data()
        self._feature_selection()
        self._print_pre_transformation_info()
        y = self.data[self.target_variable]
        self.data = self.data.drop(columns=[self.target_variable])
        pipeline = self.create_pipeline()
        self.data = pd.DataFrame(pipeline.fit_transform(self.data), columns=pipeline.get_feature_names_out())
        self.data = self.data.astype('float32')
        self._print_post_transformation_info()
        self._split_data(self.data, y)
        self._manage_imbalanced_classes()
        self.y_train, self.y_test = self.convert_target_to_int32(self.y_train, self.y_test)
        self._print_processed_data_info()
        self._print_final_data_info(self.X_train, self.X_test, self.y_train, self.y_test)
        self.train_random_forest()  # train the random forest.
        self.visualizer.plot_feature_importance(self.feature_importance)
        self.logger.info("Data Preprocessing Complete.")


# data_path = '../data/total_ksi.csv'
# preprocessor = DataPreprocessing(data_path)
# preprocessor.run()
