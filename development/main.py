# from data_load import load_data
# from feature_engineering import create_features
# from feature_selection import select_features
# from train_test_split import split_data
# from data_preprocessing import create_preprocessing_pipeline
# from model_training import train_models
# from model_evaluation import evaluate_models
# import joblib
# from handle_imbalance import handle_imbalance

# # 1. Load Data
# file_path = "Data/cancer_detection_dataset.csv"

# df = load_data(file_path)


# # 2. Feature Engineering
# df = create_features(df)


# # 3. Feature Selection
# X, y = select_features(df)


# # 4. Train Test Split
# X_train, X_test, y_train, y_test = split_data(X, y)


# # 5. Create Preprocessing Pipeline
# preprocessing_pipeline = create_preprocessing_pipeline()


# # 6. Fit preprocessing only on training data
# X_train_processed = preprocessing_pipeline.fit_transform(X_train)
# X_test_processed = preprocessing_pipeline.transform(X_test)

# # handle imballancing
# X_train_resampled, y_train_resampled = handle_imbalance(
#     X_train_processed,
#     y_train,
#     method = "smote"
# )
# print(y_train.value_counts())
# print(y_train_resampled.value_counts())

# # 7. Train Model
# trained_models = train_models(X_train_resampled, y_train_resampled)

# print("Training Data Shape:", X_train_processed.shape)
# print("Testing Data Shape:", X_test_processed.shape)

# # 8. Model Evaluation
# results_df = evaluate_models(trained_models, X_test_processed, y_test)
# print(results_df)

# # 9. Save Model
# joblib.dump(trained_models, "model.pkl")

# # 10. Save Preprocessing Pipeline
# joblib.dump(preprocessing_pipeline, "preprocessing_pipeline.pkl")

# print("Model and preprocessing pipeline saved successfully")




#----------------------------------------------
from data_load import load_data
from feature_engineering import create_features
from feature_selection import select_features
from train_test_split import split_data
from data_preprocessing import create_preprocessing_pipeline
from model_training import train_models
from model_evaluation import evaluate_models
from handle_imbalance import handle_imbalance
from mlflow_tracking import (start_experiment, start_run, log_parameters, log_metrics, log_tags, log_model)
import mlflow

# Start MLflow experiment
start_experiment("Cancer Detection")

# 1. Load Data
file_path = "Data/cancer_detection_dataset.csv"

df = load_data(file_path)


# 2. Feature Engineering
df = create_features(df)


# 3. Feature Selection
X, y = select_features(df)


# 4. Train Test Split
X_train, X_test, y_train, y_test = split_data(
    X,
    y
)


# 5. Create Preprocessing Pipeline
preprocessing_pipeline = create_preprocessing_pipeline()


# 6. Fit preprocessing only on training data
X_train_processed = preprocessing_pipeline.fit_transform(
    X_train
)

X_test_processed = preprocessing_pipeline.transform(
    X_test
)


# 7. Handle Class Imbalance
methods = ["smote", "random_over", "random_under"]


for method in methods:
    X_train_resampled, y_train_resampled = handle_imbalance(
        X_train_processed,
        y_train,
        method=method
    )

    print(f"\nClass Distribution Before Resampling:")
    print(y_train.value_counts())

    print(f"\nClass Distribution After {method.upper()}:")
    print(y_train_resampled.value_counts())

# 8. Train Multiple Models

    trained_models = train_models(
        X_train_resampled,
        y_train_resampled
    )


# 8. Evaluate Multiple Models
    results_df = evaluate_models(
        trained_models,
        X_test_processed,
        y_test
    )

    print("\nModel Evaluation Results:")
    print(results_df)

# 9. Start MLflow Run for Each Model

    for model_name, model in trained_models.items():

        start_run(
        f"{method} - {model_name}"
        )

        # Parameters
        params = model.get_params()

        #log parameters
        log_parameters(
        params
        )

        # Metrics
        metrics_row = results_df[
        results_df["Model"] == model_name
        ].iloc[0]

        # log metrics
        metrics = {
        "accuracy": float(metrics_row["Accuracy"]),
        "precision": float(metrics_row["Precision"]),
        "recall": float(metrics_row["Recall"]),
        "f1_score": float(metrics_row["F1 Score"])
        }

        print(
        f"\nLogging metrics for {model_name}:"
        )

        print(metrics)

        log_metrics(
        metrics
        )

        # Tags
        tags = {
        "model_name": model_name,
        "imbalance_method": method,
        "experiment_type": "imbalance_comparison"
        }

        #log tags
        log_tags(
        tags
        )

        # Model Artifact
        log_model(
        model,
        preprocessing_pipeline
        ) 

        mlflow.end_run()