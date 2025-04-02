## Data Modelling Analysis Report

This report documents the data modelling process, including data preprocessing, feature engineering, and handling imbalanced classes for the collision dataset.

**1. Data Sampling:**

* The original dataset contained approximately 18,957 rows and 54 columns.
* To manage computational resources, the dataset was sampled down to 10% of its original size, resulting in 1,896 rows and 54 columns.

**2. Handling Missing Data:**

* Initially, the dataset had 1,896 rows and 54 columns.
* 19 columns were dropped due to a high percentage of missing values, exceeding the defined threshold.
* The dataset was reduced to 1,896 rows and 35 columns.
* Remaining missing values were imputed using the median for numerical columns and the mode for categorical columns.

**3. Feature Selection:**

* The dataset initially contained 1,896 rows and 35 columns.
* Feature selection was performed in three stages:
    * **Handling High Missing Values:** No further columns were dropped at this stage.
    * **Dropping Irrelevant Columns:** The columns `OBJECTID` and `INDEX` were dropped based on domain knowledge.
    * **Removing Highly Correlated Columns:** The columns `x` and `y` were removed due to high correlation with other features.
* The final dataset after feature selection contained 1,896 rows and 31 columns.
* **Justification for Features Used:**
    * `ACCNUM`: Unique accident identifier.
    * `DATE`, `TIME`: Temporal information for trend analysis.
    * `STREET1`, `STREET2`, `ROAD_CLASS`, `DISTRICT`, `LATITUDE`, `LONGITUDE`, `ACCLOC`: Location-related features for risk assessment.
    * `TRAFFCTL`, `VISIBILITY`, `LIGHT`, `RDSFCOND`: Environmental and road condition factors.
    * `ACCLASS`: Target variable (Fatal or Non-fatal).
    * `IMPACTYPE`, `INVTYPE`, `INVAGE`, `INJURY`, `INITDIR`, `VEHTYPE`, `MANOEUVER`, `DRIVACT`, `DRIVCOND`: Accident and participant characteristics.
    * `AUTOMOBILE`, `AG_DRIV`: Vehicle and driver details.
    * `HOOD_158`, `NEIGHBOURHOOD_158`, `HOOD_140`, `NEIGHBOURHOOD_140`, `DIVISION`: Additional location context.
* **Justification for Data Columns Discarded:**
    * Columns with >50% missing values were dropped.
    * `OBJECTID` and `INDEX` were deemed irrelevant.
    * `x` and `y` were removed due to high correlation.

**4. Data Transformation Pipeline:**

* The data was transformed using a `ColumnTransformer` within a `Pipeline`.
* Numerical features were scaled using `StandardScaler`.
* Categorical features were one-hot encoded, resulting in 3,861 columns.
* All data types were converted to float64.

**5. Data Splitting:**

* The dataset was split into training (approximately 1,516 rows, 3,861 columns) and test (approximately 380 rows, 3,861 columns) sets.

**6. Data Evolution:**

* The original dataset contained approximately 18,957 rows.
* To manage computational resources, the dataset was sampled down to 10% of its original size, resulting in 1,896 rows.
* During the data preprocessing phase, columns with high missing values were dropped, and irrelevant columns were removed, reducing the number of columns.
* One-hot encoding of categorical features significantly increased the number of columns, resulting in 3,861 features.
* The dataset was then split into training and test sets.
* SMOTE was applied to the training set to balance the target variable, resulting in an increase in the number of training samples from approximately 1,516 to 2,618.
* The test set was not modified by SMOTE, so it retains its original number of rows (380).

**7. Handling Imbalanced Classes:**

* The training data exhibited class imbalance, with a significantly higher number of non-fatal accidents.
* SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data.
* Before SMOTE, the training data had 1,309 non-fatal and 207 fatal accidents.
* After SMOTE, the training data was balanced, with 1,309 instances of each class, and the total training row count was increased to 2,618.
* The test data retained its original class distribution.

**8. Final Data and Results:**

* Refer to the "Data Evolution" section for a detailed explanation of how the data reached its final state.
* **X_train:**
    * Shape: (2,618, 3,861)
    * Data Type: float64
* **X_test:**
    * Shape: (380, 3,861)
    * Data Type: float64
* **y_train:**
    * Shape: (2,618,)
    * Data Type: int64
    * Value Counts:
        * 1: 1,309
        * 0: 1,309
* **y_test:**
    * Shape: (380,)
    * Data Type: int64
    * Value Counts:
        * 0: 332
        * 1: 48

**9. Feature Selection Recommendations for Model Building:**

* Given the high dimensionality (3,861 columns), feature selection should be integrated into the model building process.
* Recommended techniques include:
    * Correlation analysis.
    * Feature importance from tree-based models.
    * Principal Component Analysis (PCA).
    * Variance thresholding.
    * Recursive Feature Elimination (RFE).
    * L1 Regularization.
    * Chi-squared test for categorical variables.
* Feature selection should be treated as a hyperparameter and tuned using cross-validation.