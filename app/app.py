# import streamlit as st
# import pandas as pd
# import joblib
# import sys
# import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# sys.path.append(BASE_DIR)


# from development.feature_engineering import create_features
# from development.feature_selection import select_features
# from development.model_monitoring import save_prediction

# model = joblib.load(
#     os.path.join(BASE_DIR, "model.pkl")
# )

# preprocessing_pipeline = joblib.load(
#     os.path.join(BASE_DIR, "preprocessing_pipeline.pkl")
# )


# # Streamlit Page Configuration
# st.set_page_config(
#     page_title="Cancer Detection",
#     page_icon="🏥"
# )


# st.title("🏥 Cancer Detection ML Application")

# st.write("Enter patient information to get a prediction.")


# # Patient Information
# st.subheader("Patient Information")


# patient_age = st.number_input(
#     "Patient Age",
#     min_value=1,
#     max_value=75,
#     value=30
# )


# hospital_id = st.selectbox(
#     "Hospital ID",
#     ["H001", "H002", "H003", "H004", "H005"]
# )


# salary = st.number_input(
#     "Salary",
#     min_value=10000.0,
#     value=50000.0
# )


# bmi = st.number_input(
#     "BMI",
#     min_value=0.0,
#     value=25.0
# )


# exercise_time = st.number_input(
#     "Exercise Time",
#     min_value=0.0,
#     value=30.0
# )


# existing_disease = st.selectbox(
#     "Any Existing Disease",
#     ["Yes", "No"]
# )


# alcohol = st.selectbox(
#     "Alcohol",
#     ["Yes", "No"]
# )


# checkup_date = st.date_input(
#     "Check-Up Date"
# )


# # Prediction
# if st.button("Predict"):
    
#     try:
            
#     # Create input DataFrame
#         input_data = pd.DataFrame({
#         "Patient_ID": ["NEW001"],
#         "Hospital_ID": [hospital_id],
#         "Patient_Age": [patient_age],
#         "Contact_Number": ["0000000000"],
#         "Salary": [salary],
#         "BMI": [bmi],
#         "Exercise_Time": [exercise_time],
#         "Any_Existing_Disease": [existing_disease],
#         "CheckUp_Date": [checkup_date],
#         "Alcohol": [alcohol]
#         })

#     # Feature Engineering
#         input_data = create_features(input_data)


#     # Feature Selection
#         input_data = select_features(input_data)


#     # Preprocessing
#         input_processed = preprocessing_pipeline.transform(
#             input_data
#         )


#     # Prediction
#         prediction = model.predict(input_processed)

#         probability = model.predict_proba(input_processed)

#         cancer_probability = probability[0][1] * 100


#     # Display Prediction
#         if prediction[0] == 1:

#              st.error("⚠️ Cancer Detected")

#         else:

#             st.success("✅ Cancer Not Detected")


#         st.write(
#             "Cancer Probability:",
#             round(cancer_probability, 2),
#             "%"
#         )


#     # Save Prediction Log
#         log_file = os.path.join(
#             BASE_DIR,
#             "prediction_logs.csv"
#         )

#         save_prediction(
#             input_data,
#             prediction[0],
#             cancer_probability,
#             "v1",
#             log_file
#         )

#         st.success("Prediction logged successfully.")

#     except Exception as e:

#         st.error(
#             "⚠️ Prediction could not be completed. "
#             "Please try again."
#         )

#         print("Prediction Error:", e)



#----------------------------------------------
import streamlit as st
import pandas as pd
import joblib
import sys
import os
import mlflow
import mlflow.sklearn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)


from development.feature_engineering import create_features
from development.feature_selection import select_features
from development.model_monitoring import save_prediction

#model = joblib.load(
#    os.path.join(BASE_DIR, "model.pkl")
#)

#preprocessing_pipeline = joblib.load(
#    os.path.join(BASE_DIR, "preprocessing_pipeline.pkl")
#)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_URI = "models:/CancerDetectionModel@production"

model = mlflow.sklearn.load_model(MODEL_URI)


# Streamlit Page Configuration
st.set_page_config(
    page_title="Cancer Detection",
    page_icon="🏥"
)


st.title("🏥 Cancer Detection ML Application")

st.write("Enter patient information to get a prediction.")


# Patient Information
st.subheader("Patient Information")


patient_age = st.number_input(
    "Patient Age",
    min_value=1,
    max_value=50,
    value=30
)


hospital_id = st.selectbox(
    "Hospital ID",
    ["H001", "H002", "H003", "H004", "H005"]
)


salary = st.number_input(
    "Salary",
    min_value=10000.0,
    value=50000.0
)


bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)


exercise_time = st.number_input(
    "Exercise Time",
    min_value=0.0,
    value=30.0
)


existing_disease = st.selectbox(
    "Any Existing Disease",
    ["Yes", "No"]
)


alcohol = st.selectbox(
    "Alcohol",
    ["Yes", "No"]
)


checkup_date = st.date_input(
    "Check-Up Date"
)


# Prediction
# Prediction
if st.button("Predict"):

    try:

        # Create input DataFrame
        input_data = pd.DataFrame({
            "Hospital_ID": [hospital_id],
            "Patient_Age": [patient_age],
            "Salary": [salary],
            "BMI": [bmi],
            "Exercise_Time": [exercise_time],
            "Any_Existing_Disease": [existing_disease],
            "CheckUp_Date": [checkup_date],
            "Alcohol": [alcohol]
        })

        # Feature Engineering
        input_data = create_features(input_data)

        # Feature Selection
        #input_data = select_features(input_data)

        # Preprocessing
        #input_processed = preprocessing_pipeline.transform(
        #    input_data
        #)

        # Prediction
        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)

        cancer_probability = probability[0][1] * 100

        # Display Prediction
        if prediction[0] == 1:

            st.error("⚠️ Cancer Detected")

        else:

            st.success("✅ Cancer Not Detected")

        st.write(
            "Cancer Probability:",
            round(cancer_probability, 2),
            "%"
        )

        # Save Prediction Log
        log_file = os.path.join(
            BASE_DIR,
            "prediction_logs.csv"
        )

        save_prediction(
            input_data,
            prediction[0],
            cancer_probability,
            "v1",
            log_file
        )

        st.success("Prediction logged successfully.")

    except Exception as e:

        st.error(
            "⚠️ Prediction could not be completed. "
            "Please try again."
        )