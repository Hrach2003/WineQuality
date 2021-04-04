from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pprint import pprint


def getRandomizedSearchParams():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 30, num=5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # overall 5 * 2 * 6 * 3 * 3 * 2 = 1080
    pprint(random_grid)
    return random_grid


def searchBestParamsRandomizedSearch(model, X, y, random_grid):
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid,
                                   n_iter=50, cv=3, verbose=2, random_state=42)
    # Fit the random search model
    rf_random.fit(X, y)
    return rf_random.best_params_


def getGridSearchParams():
    # based on random search results
    best_params_rf = {
        'n_estimators': [500, 550, 600, 650, 700],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1],
        'max_features': ['auto'],
        'max_depth': [None, 5, 8],
        'bootstrap': [False],
    }
    return best_params_rf


def gridSearch(rf_model, X, y, param_grid):
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                               cv=3)
    grid_search.fit(X, y)
    return grid_search.best_estimator_
