#pipeline_notebook.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

#Define a function for data preprocessing
def create_preprocessor(categorical_cols, numerical_cols):
    """
    Creates a preprocessor for both categorical and numerical features.
    :param categorical_cols: List of categorical column names.
    :param numerical_cols: List of numerical column names.
    :return: ColumnTransformer with the appropriate preprocessing steps.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ]
    )
    return preprocessor

#Define functions to create pipelines for each model

def create_logistic_pipeline(preprocessor):
    """
    Creates a pipeline for logistic regression with preprocessing.
    :param preprocessor: Preprocessing steps to be included in the pipeline.
    :return: Pipeline object for logistic regression.
    """
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])

def create_random_forest_pipeline(preprocessor):
    """
    Creates a pipeline for random forest with preprocessing.
    :param preprocessor: Preprocessing steps to be included in the pipeline.
    :return: Pipeline object for random forest classifier.
    """
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100))])

def create_knn_pipeline(preprocessor, n_neighbors=5):
    """
    Creates a pipeline for K-Nearest Neighbors with preprocessing.
    :param preprocessor: Preprocessing steps to be included in the pipeline.
    :param n_neighbors: Number of neighbors for KNN.
    :return: Pipeline object for K-Nearest Neighbors classifier.
    """
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors))])

def create_xgboost_pipeline(preprocessor):
    """
    Creates a pipeline for XGBoost with preprocessing.
    :param preprocessor: Preprocessing steps to be included in the pipeline.
    :return: Pipeline object for XGBoost classifier.
    """
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier())])