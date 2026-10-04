import pandas as pd


def create_features(df):

    df["CheckUp_Date"] = pd.to_datetime(df["CheckUp_Date"])

    df["CheckUp_Year"] = df["CheckUp_Date"].dt.year
    df["CheckUp_Month"] = df["CheckUp_Date"].dt.month
    df["CheckUp_Day"] = df["CheckUp_Date"].dt.day
    df["CheckUp_DayOfWeek"] = df["CheckUp_Date"].dt.dayofweek


    df = df.drop("CheckUp_Date", axis=1)

    return df