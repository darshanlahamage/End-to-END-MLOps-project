import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    report = {}
    best_model_params = {}

    try:
        for model_name, model in models.items():
            param_grid = params[model_name]

            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(x_train, y_train)

            best_model_params[model_name] = gs.best_params_

            # Use the best model and retrain on full training set
            best_model = gs.best_estimator_
            best_model.fit(x_train, y_train)

            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = {
                "train_r2_score": train_model_score,
                "test_r2_score": test_model_score
            }

        return report, best_model_params

    except Exception as e:
        raise CustomException(e, sys)
