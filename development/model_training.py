# from sklearn.linear_model import LogisticRegression

# def train_model(X_train, y_train):

#    model = LogisticRegression()

#    model.fit(X_train, y_train)

#    return model



from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_models(X_train, y_train):

    models = {
        "Logistic Regression": LogisticRegression(),

        "Decision Tree": DecisionTreeClassifier(),

        "Random Forest": RandomForestClassifier(),

        "SVM": SVC()
    }

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models