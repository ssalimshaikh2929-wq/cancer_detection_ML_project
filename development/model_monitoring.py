import os
import pandas as pd

# same code for all projects 

def save_prediction(
    input_data,
    prediction,   # model result
    probability,
    model_version,
    log_file   # rediction saved on this file
):    

    log_data = input_data.copy() 

    log_data["Prediction"] = prediction
    log_data["Probability"] = probability
    log_data["Model_Version"] = model_version
    log_data["Actual_Label"] = None

    file_exists = os.path.exists(log_file)

    log_data.to_csv(
        log_file,
        mode="a",
        header=not file_exists,
        index=False
    )

def add_actual_labels(log_file, actual_labels):   #human lables

    df = pd.read_csv(log_file)

    df["Actual_Label"] = actual_labels

    df.to_csv(
        log_file,
        index=False
    )
