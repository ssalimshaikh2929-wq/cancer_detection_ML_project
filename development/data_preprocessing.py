import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler, OrdinalEncoder


def create_preprocessing_pipeline():

    MinMaxScaler_columns = [
        "Patient_Age",
        "Salary",
        "BMI",
        "Exercise_Time",
    ]

    StandardScaler_columns = [
        "CheckUp_Year",
        "CheckUp_Month",
        "CheckUp_Day",
        "CheckUp_DayOfWeek"
    ]

    OneHot_columns = [
        "Hospital_ID",
        "Any_Existing_Disease",  
    ]

    Ordinal_columns = [
        "Alcohol"
    ]

    MinMaxScaler_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler())         
    ])

    StandardScaler_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    OneHot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))  
    ])

    Ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder())
    ])

    preprocessing_pipeline = ColumnTransformer([
        ("MinMax", MinMaxScaler_pipeline, MinMaxScaler_columns),
        ("Standard", StandardScaler_pipeline, StandardScaler_columns),
        ("OneHot", OneHot_pipeline, OneHot_columns),
        ("Label", Ordinal_pipeline, Ordinal_columns)
        
    ])

    return preprocessing_pipeline