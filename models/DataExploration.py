import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import logging
from logger_config import setup_logger #Import logger


from models.DataVisualization import DataVisualization

import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from models.DataVisualization import DataVisualization
from logger_config import setup_logger

class DataExploration:
    def __init__(self, data_path):
        self.logger = setup_logger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.data = pd.read_csv(data_path)
        self._convert_acclass_to_numeric()
        self.visualizer = DataVisualization(self.data)

    def load_and_describe_data(self):
        self.logger.info("--- Data Loading and Description ---")
        self.logger.info(f"First few rows of the data:\n{self.data.head()}")
        self.logger.info(f"Data shape: {self.data.shape}")
        self.logger.info(f"Column information:\n{self.data.info()}")

        for col in self.data.columns:
            self.logger.info(f"\n--- Column: {col} ---")
            self.logger.info(f"Data type: {self.data[col].dtype}")
            self.logger.info(f"Unique values: {self.data[col].nunique()}")
            if self.data[col].dtype in ['int64', 'float64']:
                self.logger.info(f"Range: [{self.data[col].min()}, {self.data[col].max()}]")
                self.logger.info(f"Mean: {self.data[col].mean()}")
                self.logger.info(f"Median: {self.data[col].median()}")
                self.logger.info(f"Standard deviation: {self.data[col].std()}")
            else:
                self.logger.info(f"Sample values: {self.data[col].unique()[:5]}")


    def _convert_acclass_to_numeric(self):
        """Converts the 'ACCLASS' column to numeric (Fatal = 1, otherwise 0)."""
        self.data['ACCLASS'] = self.data['ACCLASS'].apply(lambda x: 1 if x == 'Fatal' else 0)

    def statistical_assessments(self):
        """
        Performs statistical assessments including means, averages, and correlations.
        """
        self.logger.info("\n--- Statistical Assessments ---")
        self.logger.info("Descriptive statistics:")
        self.logger.info(self.data.describe())

        self.logger.info("\nCorrelation matrix:")
        corr_matrix = self.data.corr()
        self.logger.info(corr_matrix)

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def missing_data_evaluation(self):
        """
        Evaluates missing data in the dataset.
        """
        self.logger.info("\n--- Missing Data Evaluation ---")
        self.logger.info("Missing values per column:")
        self.logger.info(self.data.isnull().sum())

        missing_percentage = (self.data.isnull().sum() / len(self.data)) * 100
        self.logger.info("\nMissing percentage per column:")
        self.logger.info(missing_percentage)

        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Data Heatmap")
        plt.show()

    def run(self):

        """
        Runs all data exploration tasks.
        """
        self.logger.info("Starting Data Exploration...")
        self.load_and_describe_data()
        # self.statistical_assessments()
        self.missing_data_evaluation()

        # Call the plotting functions from the DataVisualization instance
        self.visualizer.plot_missing_data_heatmaps()
        self.visualizer.plot_histograms()
        self.visualizer.plot_bar_plots()
        self.visualizer.plot_pie_charts()
        self.visualizer.plot_scatter_plots()
        self.visualizer.plot_box_plots()
        self.visualizer.plot_grouped_bar_plots()
        self.visualizer.plot_heatmaps()
        self.visualizer.plot_time_series()
        self.visualizer.plot_missing_percentages()
        self.visualizer.plot_pairwise_scatter_plots()
        self.logger.info("Data Exploration Complete.")
        return self.data


data_path = '../data/total_ksi.csv'
explorer = DataExploration(data_path)
explorer.run()