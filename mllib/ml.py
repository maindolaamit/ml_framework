import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.model_selection as model_selection
import sklearn.svm as svm
import sklearn.tree as tree
import xgboost as xgboost

from utils.misc import get_display_time

# Keep randomness same
np.random.seed(2210)


class EstimatorSelectHelper:
    # Code derived and changed accordingly from below
    # https://github.com/davidsbatista/machine-learning-notebooks/blob/master/hyperparameter-across-models.ipynb

    def __init__(self, models):
        self.models = models
        self.keys = models.keys()
        self.search_grid = {}
        self.df_val_score = None

    def fit(self, X, y, **grid_kwargs):
        for model_key in self.keys:
            # Check the model and param_grid
            model = self.models[model_key][0]
            param_grid = self.models[model_key][1]
            # Call GridSearchCV on the model and param_grid
            print(f"Running GridSearchCV for {model_key}")
            grid = model_selection.GridSearchCV(model, param_grid, **grid_kwargs)
            grid.fit(X, y)
            self.search_grid[model_key] = grid
        return self

    def val_score(self, sort_by='mean_val_score'):
        frames = []
        for name, grid in self.search_grid.items():
            frame = pd.DataFrame(grid.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame) * [name]
            frames.append(frame)
        df_val_score = pd.concat(frames)

        df_val_score = df_val_score.reset_index()
        df_val_score = df_val_score.drop(['rank_test_score', 'index'], 1)

        # columns = ['estimator'] + df.columns.tolist().remove('estimator')
        # Keep required columns
        df_val_score.rename(columns={'mean_test_score': 'mean_val_score', 'std_test_score': 'std_val_score'},
                            inplace=True)
        keep_columns = [
            "estimator",
            "mean_val_score",
            "std_val_score",
            "mean_fit_time",
            "mean_score_time",
            "params",
        ]
        df_val_score = df_val_score[keep_columns].sort_values([sort_by], ascending=False)
        self.df_val_score = df_val_score
        return self.df_val_score


class RegressionSelectHelper(EstimatorSelectHelper):

    def __init__(self, models):
        super().__init__(models)
        self.df_test_score = None

    def fit(self, X, y, **grid_kwargs):
        super().fit(X, y, **grid_kwargs)

    def val_score(self, sort_by='mean_val_score'):
        return super().val_score(sort_by)

    def test_score(self, X_test, y_test, sort_by=['mean_squared_error']):
        test_scores = []
        for key, model in self.search_grid.items():
            y_pred = model.predict(X_test)
            import sklearn.metrics as sm
            mse = sm.mean_squared_error(y_test, y_pred)
            mae = sm.mean_absolute_error(y_test, y_pred)
            r2 = sm.r2_score(y_test, y_pred)
            test_scores.append([key, model.best_params_, mse, mae, r2])

        test_score_columns = ['estimator', 'params', 'mean_squared_error', 'mean_absolute_error', 'r2_score',
                              'jacobian_score']
        self.df_test_score = pd.DataFrame(test_scores, columns=test_score_columns).reset_index(drop=True)
        return self.df_test_score


class ClassifierSelectHelper(EstimatorSelectHelper):

    def __init__(self, models):
        super().__init__(models)
        self.df_test_score = None

    def fit(self, x, y, **grid_kwargs):
        super().fit(x, y, **grid_kwargs)

    def val_score(self, sort_by='mean_val_score'):
        return super().val_score(sort_by)

    def test_score(self, x_test, y_test, sort_by=None):
        if sort_by is None:
            sort_by = ['precision']
        test_scores = []
        for key, model in self.search_grid.items():
            y_pred = model.predict(x_test)
            import sklearn.metrics as sm
            accuracy = sm.accuracy_score(y_test, y_pred)
            precision = sm.precision_score(y_test, y_pred)
            recall = sm.recall_score(y_test, y_pred)
            f1_score = sm.f1_score(y_test, y_pred)
            roc_auc = sm.roc_auc_score(y_test, y_pred)
            log_loss = sm.log_loss(y_test, y_pred)
            test_scores.append([key, model.best_params_, accuracy, precision, recall, f1_score, roc_auc, log_loss])

        test_score_columns = ['estimator', 'params', 'accuracy', 'precision', 'recall',
                              'f1-score', 'roc_auc', 'log_loss']
        self.df_test_score = pd.DataFrame(test_scores, columns=test_score_columns)
        self.df_test_score = self.df_test_score.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
        return self.df_test_score


def evaluate_classifiers(X_train, y_train, X_test, y_test, is_binary=False, cv=5, sort_by=['f1-score']):
    """
    Perform raw evaluation of the Classifer Models on the given data and return the Validation and Test Score results
    """
    models = {
        'DecisionTreeClassifier': (tree.DecisionTreeClassifier(), {}),
        'SVM': (svm.SVC(), {}),
        'RandomForestClassifier': (ensemble.RandomForestClassifier(), {}),
        'LightGBMClassifier': (lgb.LGBMClassifier(), {}),
        'AdaBoostClassifier': (ensemble.AdaBoostClassifier(), {}),
        'GradinetBoostingClassifier': (ensemble.GradientBoostingClassifier(), {}),
        'XGBClassifier': (xgboost.XGBClassifier(verbose=0, silent=True), {}),
    }

    # LogisticRegression
    if is_binary:
        models.update({'LogisticRegression': (linear_model.LogisticRegression(), {})})

    if len(X_train) > 10000:
        models.update({'SGDClassifier': (linear_model.SGDClassifier(), {})})

    select = ClassifierSelectHelper(models)
    select.fit(X_train, y_train, cv=cv, verbose=0)
    df_val_score = select.val_score(sort_by='mean_val_score')
    df_test_score = select.test_score(X_test, y_test, sort_by=sort_by)
    search_grid = select.search_grid
    return df_val_score, df_test_score, search_grid


# TODO : will be depricated with fine_tune in model_builder
def fine_tune_classifier(model_name, x_train, y_train, cv=5, verbose=0, randomized=False):
    model, param_grid = None, None
    if model_name == 'xgb':
        model = xgboost.XGBClassifier(verbose=verbose)
        param_grid = {
            "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
            "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
        }
    elif model_name == 'rf':
        model = ensemble.RandomForestClassifier()
        param_grid = {'n_estimators': [10, 25], 'max_features': [5, 10],
                      'max_depth': [10, 50, None], 'bootstrap': [True, False]}
    elif model_name == 'lr':
        model = linear_model.LogisticRegression()
        param_grid = {
            "solver": ["newton-cg", "lbfgs", "liblinear"],
            "penalty": ['l1', 'l2'],
            "C": [100, 10, 1, 0.1, 0.01],
        }
    elif model_name == 'ada':
        model = ensemble.AdaBoostClassifier()
        param_grid = {
            "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "n_estimators": [0, 50, 100, 500]
        }
    elif model_name == 'gb':
        model = ensemble.GradientBoostingClassifier
        param_grid = {}
    elif model_name == 'lgb':
        model = lgb.LGBMClassifier()
        param_grid = {}
    elif model_name == 'svm':
        model = svm.SVC()
        param_grid = {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
            "kernel": ["rbf", 'linear', 'sigmoid'],
        }
    elif model_name == 'dt':
        model = tree.DecisionTreeClassifier()
        param_grid = {}
    elif model_name == 'sgd':
        model = linear_model.SGDClassifier()
        param_grid = {}

    return fine_tune_model(model, param_grid, x_train, y_train, cv, verbose, randomized)
    # from time import perf_counter
    #
    # start_time = perf_counter()
    #
    # grid_search = None
    # if randomized:
    #     print(f"Performing Randomized search for {type(model).__name__}...")
    #     grid_search = model_selection.RandomizedSearchCV(model, param_grid, cv=cv, verbose=verbose, n_jobs=-1)
    # else:
    #     print(f"Performing Grid search for {type(model).__name__}...")
    #     grid_search = model_selection.GridSearchCV(model, param_grid, cv=cv, verbose=verbose, n_jobs=-1)
    #
    # # Start fine tuning of the model
    # grid_search.fit(x_train, y_train)
    # time_taken = round(perf_counter() - start_time, 2)
    # print(f"Time elapsed(s) : {get_display_time(time_taken)} | score : {grid_search.best_score_:.2}")
    # print(f"Best parameters : {grid_search.best_params_} ")
    # return grid_search.best_estimator_


# TODO : will be depricated with fine_tune in model_builder
def fine_tune_model(model, param_grid, x_train, y_train, cv=5, verbose=0, randomized=False):
    """
    Fine Tune a given Model by using GridSearchCV/RandomizedSearchCV with the Passed parameter grid
    :param model: Estimator Model
    :param param_grid: Parameters grid
    :param x_train: Train dataset
    :param y_train: Train target
    :param cv: No. of cross validations, default 5
    :param verbose: verbose, default 0
    :param randomized: default False, if True, randomized search to be used
    :return:
    """
    from time import perf_counter

    start_time = perf_counter()

    grid_search = None
    if randomized:
        print(f"Performing Randomized search for {type(model).__name__}...")
        grid_search = model_selection.RandomizedSearchCV(model, param_grid, cv=cv, verbose=verbose, n_jobs=-1)
    else:
        print(f"Performing Grid search for {type(model).__name__}...")
        grid_search = model_selection.GridSearchCV(model, param_grid, cv=cv, verbose=verbose, n_jobs=-1)

    # Start fine tuning of the model
    grid_search.fit(x_train, y_train)
    time_taken = round(perf_counter() - start_time, 2)
    print(f"Time elapsed : {get_display_time(time_taken)} | score : {grid_search.best_score_:.2}")
    print(f"Best parameters : {grid_search.best_params_} ")
    return grid_search.best_estimator_


# def train_models(X, y, cv, models, problem_type='classification', scoring=['accuracy', 'precision']):
#     for name, model in models.items():
#         model.fit(X, y)
#         np.random.seed(2210)
#         scores = model_selection.cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1, verbose=0)
#         print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
#
#         # Export the model
#         joblib.dump(model, f"models/{name}.pkl")
#         joblib.dump(X.columns, f"models/{name}_columns.pkl")
#
#         if problem_type == 'classification':
#             select = ClassifierSelectHelper(models)
#             select.fit(X_train, y_train, cv=cv, verbose=0)
#             df_score, search_grid = select.score_summary(X_test, y_test, sort_by=scoring)
#             return df_score, search_grid
#         else:
#             select = RegressionSelectHelper()
#             select.fit(X_train, y_train, cv=cv, verbose=0)
#             df_score, search_grid = select.test_score(X_test, y_test, sort_by=scoring)
#             return df_score, search_grid


FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

# def train_models_with_folds(fold, df, target_col, drop_columns, models,
#                             problem_type='classification', score='accuracy'):
#     """
#     Train the model on the given fold. Dataframe has a column having fold number
#     :param fold: Fold number raning from 0 to 5
#     :param df: DataFrame
#     :param target_col: Target column
#     :param drop_columns: Columns to drop
#     :param models: Model to train on
#     :param problem_type: Problem type
#     :param score: score used for evaluation
#     """
#     import dispatcher
#
#     train_df = df[df.kfold.isin(FOLD_MAPPPING.get(fold))].reset_index(drop=True)
#     valid_df = df[df.kfold == fold].reset_index(drop=True)
#
#     train_df = train_df.drop(drop_columns + target_col, axis=1)
#     valid_df = valid_df.drop(drop_columns + target_col, axis=1)
#
#     y_train = train_df[target_col].values
#     y_valid = valid_df[target_col].values
#
#     for name, model in models.items():
#         model.fit(train_df)
#
#         if problem_type == 'classification':
#             from metrics import ClassificationMetrics
#             dispatcher.MODELS[model]
#             preds = model.predict_proba(valid_df)[:, 1]
#             metric = ClassificationMetrics()
#             print(metric(score, y_valid, preds))
#         else:
#             from metrics import RegressionMetrics
#             preds = model.predict(valid_df)
#             metric = RegressionMetrics()
#             print(metric(score, y_valid, preds))
#
#         # Export the model
#         joblib.dump(model, f"models/{model}_{fold}.pkl")
#         joblib.dump(train_df.columns, f"models/{model}_{fold}_columns.pkl")


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    # Return the dataframe
    le.fit(df["diagnosis"])
    df["diagnosis"] = le.transform(df["diagnosis"])
    X, y = df.drop("diagnosis", axis=1), df["diagnosis"].values
    columns = X.columns.to_list()
    num_cols = len(columns)
    print(X.shape, y.shape)
