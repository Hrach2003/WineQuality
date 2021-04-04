from scipy.stats.stats import rankdata
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def randomizedSearch(model, X, y, random_grid, n_iter=50):
    random_search = RandomizedSearchCV(
        estimator=model, param_distributions=random_grid, n_iter=n_iter, cv=3, verbose=2, random_state=42)
    # Fit the random search model
    random_search.fit(X, y)
    return random_search.best_params_


def gridSearch(model, X, y, param_grid):
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=3)
    # Fit the grid search model
    grid_search.fit(X, y)
    return grid_search.best_estimator_
