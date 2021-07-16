import numpy as np
from sklearn import metrics as skmetrics
from utils import charts


class ClassificationMetrics:
    def __init__(self):
        """
        Class to return the various classification metrics score.
        clf_metric = ClassificationMetrics()
        clf_metric('accuracy', y_true, y_pred)
        clf_metric('auc', y_true, y_pred, y_proba)
        """
        self.metrics = {'accuracy': self._accuracy,
                        'precision': self._precision,
                        'recall': self._recall,
                        'f1': self._f1,
                        'auc': self._auc,
                        'logloss': self._log_loss
                        }

    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception(f'Invalid metric, should be {self.metrics.keys()}')
        if metric in ['auc', 'logloss']:
            if y_proba is not None:
                # return self._auc(y_true=y_true, y_pred=y_proba)
                return self.metrics[metric](y_true, y_proba)
            else:
                raise Exception(f'y_proba can not be None for AUC and Logloss.')
        # elif metric == 'logloss':
        #     if y_proba is not None:
        #         return self._log_loss(y_true=y_true, y_pred=y_proba)
        #     else:
        #         raise Exception(f'y_proba can not be None for logloss.')
        else:
            return self.metrics[metric](y_true, y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true, y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true, y_pred)

    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true, y_pred)

    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true, y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true, y_pred)

    @staticmethod
    def _log_loss(y_true, y_pred):
        return skmetrics.log_loss(y_true, y_pred)


class RegressionMetrics:
    def __init__(self):
        """
        Class to return the various classification metrics score.
        reg_metric = ClassificationMetrics()
        reg_metric('mae', y_true, y_pred)
        reg_metric('rmse', y_true, y_pred)
        """
        self.metrics = {'mae': self._mae,
                        'mse': self._mse,
                        'r2': self._r2,
                        'rmse': self._rmse,
                        'msle': self._msle,
                        'rmsle': self._rmsle
                        }

    def __call__(self, metric, y_true, y_pred):
        if metric not in self.metrics:
            raise Exception(f'Invalid metric, should be {self.metrics.keys()}')
        else:
            return self.metrics[metric](y_true, y_pred)

    @staticmethod
    def _mae(y_true, y_pred):
        return skmetrics.mean_absolute_error(y_true, y_pred)

    @staticmethod
    def _mse(y_true, y_pred):
        return skmetrics.mean_squared_error(y_true, y_pred)

    @staticmethod
    def _r2(y_true, y_pred):
        return skmetrics.r2_score(y_true, y_pred)

    @staticmethod
    def _rmse(y_true, y_pred):
        return np.sqrt(skmetrics.mean_squared_error(y_true, y_pred))

    @staticmethod
    def _msle(y_true, y_pred):
        return skmetrics.mean_squared_log_error(y_true, y_pred)

    @staticmethod
    def _rmsle(y_true, y_pred):
        return np.sqrt(skmetrics.mean_squared_log_error(y_true, y_pred))


def print_classification_score(model, X_test, y_test, target_names=None,
                               title="Confusion Matrix", multi_class=False):
    """ Method to print and plot Confusion matrix, F1 Score """
    from sklearn.metrics import classification_report
    if multi_class:
        y_pred_classes = np.argmax(model.predict(X_test), axis=1)  # Multiclass classification
        y_true_classes = np.argmax(y_test, axis=1)  # Multiclass classification
    else:
        y_pred_classes = model.predict(X_test)
        y_true_classes = y_test
    print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))
    from sklearn.metrics import confusion_matrix
    cnf = confusion_matrix(y_true_classes, y_pred_classes)
    # cnf = get_enhanced_confusion_matrix(y_true_classes, y_pred_classes, target_names)
    # Import confusion_matrix plotting from the library
    charts.plot_confusion_matrix(cnf, target_names, title)


def get_enhanced_confusion_matrix(y, y_pred, labels=None):
    """"enhances confusion_matrix by adding sensitivity and specificity metrics"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, y_pred, labels=labels)
    sensitivity = float(cm[1][1]) / float(cm[1][0] + cm[1][1])
    specificity = float(cm[0][0]) / float(cm[0][0] + cm[0][1])
    weighted_accuracy = (sensitivity * 0.9) + (specificity * 0.1)
    return cm, sensitivity, specificity, weighted_accuracy
