import os
import sys
import dill
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logging_config import logging
from sklearn.model_selection import GridSearchCV



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        logging.error(f"Error loading object: {e}")
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, parameters: dict):
    report = {}

    try:
        for name, model in models.items():

        # Train on full training data
            #model.fit(X_train, y_train)
            params = parameters.get(name, {})
            
            gcv = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring="r2", n_jobs=-1)
            gcv.fit(X_train, y_train)
            
            cv_mean = gcv.best_score_
            
            model.set_params(**gcv.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            train_score = r2_score(y_train, y_train_pred)
            
            y_test_pred = model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[name] = {
                "cv_score": cv_mean,
                "train_score": train_score,
                "test_score": test_score
            }

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise CustomException(e, sys)

    return report