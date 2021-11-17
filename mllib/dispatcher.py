from sklearn import ensemble, linear_model
from sklearn import tree
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
}

CLF_MODELS = {
    "logistic": {
        "model_name": "LogisticRegression",
        "model_package": "sklearn.linear_model",
        "model": linear_model.LogisticRegression(n_estimators=200, n_jobs=-1, verbose=2),
        "param_grid": {
            "solver": ["newton-cg", "lbfgs", "liblinear"],
            "penalty": ["l1", "l2"],
            "C": [100, 10, 1, 0.1, 0.01]
        }
    },
    "sgd": {
        "model_name": "SGD",
        "model_package": "sklearn.linear_model",
        "model": linear_model.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
        "param_grid": {}
    },
    "liner_svc": {
        "model_name": "LinearSVC",
        "model_package": "sklearn.svm",
        "model": svm.LinearSVC(),
        "param_grid": {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [5, 1, 0.1, 0.01, 0.001, 0.0001],
            "kernel": ["rbf", 'poly', 'linear', 'sigmoid'],
        }
    },
    "svc": {
        "model_name": "SVM",
        "model_package": "sklearn.svm",
        "model": svm.SVC(),
        "param_grid": {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [5, 1, 0.1, 0.01, 0.001, 0.0001],
            "kernel": ["rbf", 'poly', 'linear', 'sigmoid'],
        }
    },
    "decision_tree": {
        "model_name": "DecisionTreeClassifier",
        "model_package": "sklearn.tree.DecisionTreeClassifier",
        "model": tree.DecisionTreeClassifier(),
        "param_grid": {
            "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
            "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
        }
    },
    "extra_tree": {
        "model_name": "ExtraTreesClassifier",
        "model_package": "sklearn.ensemble.ExtraTreesClassifier",
        "model": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
        "param_grid": {"n_estimators": [10, 25], "max_features": [5, 10],
                       "max_depth": [10, 50, None], "bootstrap": [True, False]}
    },
    "random_forest": {
        "model_name": "RandomForestClassifier",
        "model_package": "sklearn.ensemble.RandomForestClassifier",
        "model": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
        "param_grid": {"n_estimators": [10, 25], "max_features": [5, 10],
                       "max_depth": [10, 50, None], "bootstrap": [True, False]}
    },
    "lightgbm": {
        "model_name": "LightGBMClassifier",
        "model_package": "lgb.LGBMClassifier",
        "model": ensemble.LightGBMClassifier()(n_estimators=200, n_jobs=-1, verbose=2),
        "param_grid": {}
    },
    "gradient_boost": {
        "model_name": "RandomForestClassifier",
        "model_package": "sklearn.ensemble.GradientBoostingClassifier",
        "model": ensemble.GradientBoostingClassifier(warm_start=True),
        "param_grid": {
            "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "n_estimators": [0, 50, 100, 200, 500],
            "max_depth": [1, 2, 3, 4]
        }
    },
    "adaboost": {
        "model_name": "AdaBoostClassifier",
        "model_package": "sklearn.ensemble.AdaBoostClassifier",
        "model": ensemble.AdaBoostClassifier(n_estimators=200, n_jobs=-1, verbose=2),
        "param_grid": {
            "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "n_estimators": [0, 50, 100, 200, 500],
            "algorithm": ['SAMME.R', 'SAMME'],
            "max_depth": [1, 2, 3, 4]
        }
    },
    "xgb": {
        "model_name": "XGBClassifier",
        "model_package": "xgboost.XGBClassifier",
        "model": xgb.XGBClassifier(verbose=0, silent=True),
        "param_grid": {
            "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
            "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
        }
    }
}

REG_MODELS = {
    "svm": {
        "model_name": "SVM",
        "model_package": "sklearn.svm",
        "param_grid": {
            "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "n_estimators": [0, 50, 100, 500]
        }
    },
}
