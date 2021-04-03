from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import numpy as np


def classify(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    # train the model
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test) * 100
    print("Accuracy:", accuracy)

    # cross-validation
    score = cross_val_score(model, X, y, cv=5)
    mean_CV_score = np.mean(score)*100
    print("CV Score:", mean_CV_score)

    y_pred = model.predict(x_test)
    print(classification_report(y_pred, y_test))
    return accuracy, mean_CV_score


def drop_outliers(df, field):
    iqr = 1.5 * (np.percentile(df[field], 75) - np.percentile(df[field], 25))
    df.drop(df[df[field] > (iqr + np.percentile(df[field], 75))].index, inplace=True)
