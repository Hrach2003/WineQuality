from sklearn.model_selection import cross_val_score
import numpy as np


class Classifier:
    def __init__(self, *models) -> None:
        self._names = list(map(str, models))
        self._models = dict(zip(self._names, models))
        print(self._names)

    def fit(self, x_train, y_train):
        for model in self._models.values():
            model.fit(x_train, y_train)
        return self._models

    def predict(self, x_test):
        y_pred_arr = map(lambda m: m.predict(x_test), self._models.values())
        return zip(y_pred_arr, self._names)

    def score(self, x_test, y_test):
        for name, model in self._models.items():
            accuracy = model.score(x_test, y_test) * 100
            yield (name, accuracy)

    def get_models(self):
        return self._models

    def get_models_feature_importance(self):
        for name, model in self._models.items():
            yield (name, model.feature_importances_)

    def CV_score(self, X, y):
        for name, model in self._models.items():
            score = cross_val_score(model, X, y, cv=5)
            yield (name, np.mean(score)*100)

    def drop_models(self, *model_names):
        for name in model_names:
            if name in self._models:
                del self._models[name]
        self._names = self._models.keys()
        return self._names
