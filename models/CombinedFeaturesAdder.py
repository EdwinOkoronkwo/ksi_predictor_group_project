import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CombinedFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # No fitting required

    def get_feature_names_out(self, input_features=None):
        """
        Returns the feature names after the transformation.
        """
        if input_features is None:
            input_features = ['x', 'y', 'TIME']  # Provide default input features if None

        if isinstance(input_features, np.ndarray):
            output_features = input_features.tolist()  # Convert to list if it's a NumPy array
        else:
            output_features = input_features.copy()  # Otherwise, create a copy

        output_features.append('combined_xy')
        output_features.append('hour_of_day')
        output_features.append('day_of_week')
        if 'TIME' in output_features:
            output_features.remove('TIME')
        return np.array(output_features)

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)  # Convert to DataFrame if needed

        # Replace 'x' and 'y' with the actual column names
        X['combined_xy'] = np.sqrt(X['LATITUDE'] ** 2 + X['LONGITUDE'] ** 2)

        # Convert TIME to datetime and extract hour_of_day and day_of_week
        X['TIME'] = pd.to_datetime(X['TIME'])
        X['hour_of_day'] = X['TIME'].dt.hour
        X['day_of_week'] = X['TIME'].dt.dayofweek

        # Drop the original 'TIME' column
        X = X.drop('TIME', axis=1)

        return X

