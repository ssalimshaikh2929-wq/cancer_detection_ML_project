from model_monitoring import add_actual_labels


log_file = "prediction_logs.csv"


actual_labels = [
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    0,
    1
]


add_actual_labels(
    log_file,
    actual_labels
)

print("Actual labels added successfully.")