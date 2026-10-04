# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     classification_report
# )


# def evaluate_model(model, X_test, y_test):

#     y_pred = model.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     print("Accuracy:", accuracy)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 Score:", f1)

#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))

#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))


import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_models(trained_models, X_test, y_test):

    results = []

    for name, model in trained_models.items():

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred,)*100
        precision = precision_score(y_test, y_pred, zero_division=0)*100
        recall = recall_score(y_test, y_pred, zero_division=0)*100
        f1 = f1_score(y_test, y_pred, zero_division=0)*100

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    results_df = pd.DataFrame(results)

    return results_df