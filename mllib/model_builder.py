import os
import importlib
from pathlib import Path
from sklearn import model_selection

from utils.file_utils import load_from_json
from utils.misc import get_display_time

MODELS_FILE_DIR = Path(__file__).resolve().parent
CLF_MODELS_JSON_FILE = os.path.join(MODELS_FILE_DIR, 'clf_models.json')
REG_MODELS_JSON_FILE = os.path.join(MODELS_FILE_DIR, 'reg_models.json')


class Model:
    def __init__(self, model_name, model_type, **model_kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.model_class = None
        self.model = None
        self.param_grid = None
        self.best_estimator = None
        self.metrics = None
        self._set_model(model_name, **model_kwargs)

    def _set_model(self, model_name, **model_kwargs):
        # Load JSON based on model type
        if self.model_type == 'classification':
            models = load_from_json(CLF_MODELS_JSON_FILE)
        else:
            models = load_from_json(REG_MODELS_JSON_FILE)

        # Check if model name is valid
        if model_name not in models.keys():
            raise Exception(f"Model name should have one of the value {models.keys()}")

        # Load the model package, class and params
        self.model_class = models[model_name]['model_name']
        self.param_grid = models[model_name]['param_grid']
        model_package = models[model_name]['model_package']
        print(f"{model_package}.{self.model_class}")
        module = importlib.import_module(model_package)
        model_class = getattr(module, self.model_class)
        # create the model
        self.model = model_class(**model_kwargs)


class ClassifierModel(Model):
    def __init__(self, model_name, **model_kwargs):
        super().__init__(model_name, 'classification', **model_kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def fine_tune(self, X, y, param_grid=None, cv=5, randomized=False, verbose=0):
        from time import perf_counter
        start_time = perf_counter()

        if param_grid is not None:
            self.param_grid = param_grid

        grid_search = None
        if randomized:
            print(f"Performing Randomized search for {type(self.model).__name__}...")
            grid_search = model_selection.RandomizedSearchCV(self.model, param_grid, cv=cv, verbose=verbose, n_jobs=-1)
        else:
            print(f"Performing Grid search for {type(self.model).__name__}...")
            grid_search = model_selection.GridSearchCV(self.model, param_grid, cv=cv, verbose=verbose, n_jobs=-1)

        # Start fine tuning of the model
        grid_search.fit(X, y)
        time_taken = round(perf_counter() - start_time, 2)
        print(f"Time elapsed : {get_display_time(time_taken)} | score : {grid_search.best_score_:.2}")
        print(f"Best parameters : {grid_search.best_params_} ")
        self.best_estimator = grid_search.best_estimator_
        return grid_search.best_estimator_


class RegressionModel(Model):
    def __init__(self, model_name, **model_kwargs):
        super().__init__(model_name, 'regression', **model_kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        predictions = self.model.predict(X)
        # calculate the metrics

        return predictions

    def fine_tune(self, X, y, param_grid=None, cv=5, randomized=False, verbose=0):
        """
        Fine tune the Regression model for the Param grid parameters
        :param X: X, numpy array
        :param y: y, numpy array of values
        :param param_grid: Custom parameter grid, if None default will be taken from JSON.
        :param cv: Cross validation value.
        :param randomized: If randomized search to be applied.
        :param verbose: verbose, default 0.
        :return: A fine tuned estimator.
        """
        from time import perf_counter
        start_time = perf_counter()

        if param_grid is not None:
            self.param_grid = param_grid

        grid_search = None
        if randomized:
            print(f"Performing Randomized search for {type(self.model).__name__}...")
            grid_search = model_selection.RandomizedSearchCV(self.model, param_grid, cv=cv, verbose=verbose, n_jobs=-1)
        else:
            print(f"Performing Grid search for {type(self.model).__name__}...")
            grid_search = model_selection.GridSearchCV(self.model, param_grid, cv=cv, verbose=verbose, n_jobs=-1)

        # Start fine tuning of the model
        grid_search.fit(X, y)
        time_taken = round(perf_counter() - start_time, 2)
        print(f"Time elapsed : {get_display_time(time_taken)} | score : {grid_search.best_score_:.2}")
        print(f"Best parameters : {grid_search.best_params_} ")
        self.best_estimator = grid_search.best_estimator_
        return grid_search.best_estimator_


if __name__ == "__main__":
    clf_model = ClassifierModel('logistic')
    print(clf_model.model_class, clf_model.model_type)
