import sys
import os

from charset_normalizer import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.exception import CustomException
from src.logging_config import logging

from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from catboost import CatBoostRegressor
from xgboost import XGBRegressor    
from sklearn.ensemble import (RandomForestRegressor, 
GradientBoostingRegressor, 
AdaBoostRegressor, 
BaggingRegressor)
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object, evaluate_models
from sklearn.svm import SVR


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer: 
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Bagging": BaggingRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "SVR": SVR()
            }
            
            parameters = {
                
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                    "bootstrap": [True, False]
                    },
                
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                    },
                "AdaBoost": {
                    "n_estimators": [50, 100],      
                    "learning_rate": [0.01, 0.1, 1],
                    "loss": ["linear", "square", "exponential"]
                    },
                "Bagging": {
                    "n_estimators": [10, 50, 100],
                    "max_samples": [0.5, 0.75, 1.0],
                    "max_features": [0.5, 0.75, 1.0]
                    },
                "Decision Tree": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"]
                    },
                "KNN": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"]
                    },
                "XGBRegressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "gamma": [0, 0.1, 0.3],
                    "reg_alpha": [0, 0.1, 1],
                    "reg_lambda": [1, 1.5, 2]
                    },  
                "Linear Regression": {
                    "fit_intercept": [True, False], 
                    },
                "Ridge": {
                    "alpha": [0.1, 1.0, 10.0],  
                    "fit_intercept": [True, False],
                    "max_iter": [1000, 5000, 10000]
                    },
                "Lasso": {
                    "alpha": [0.1, 1.0, 10.0],
                    "fit_intercept": [True, False],
                    "max_iter": [10000],
                    },
                "ElasticNet": {
                    "alpha": [0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.5, 0.9],    
                    "fit_intercept": [True, False],
                    "max_iter": [10000]
                    },
                "CatBoost": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "depth": [3, 5, 7],
                    "l2_leaf_reg": [1, 3, 5],
                    "border_count": [32, 64]
                    },
                "SVR": {
                    "kernel": ["linear", "rbf"],
                    "C": [0.1, 1, 10],
                    "gamma": ["scale", "auto"]
                    }
            }
            
            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models, parameters
            )

            #best_model_score = max(model_report.values())
            #best_model_name = max(model_report, key=model_report.get)
            #best_model = models[best_model_name]
            
            best_model_name = max(model_report, key=lambda name: model_report[name]["test_score"])
            best_model_score = model_report[best_model_name]["test_score"]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name}, R2: {best_model_score}")
            
            if best_model_score < 0.6:
                raise CustomException("No model met the minimum R² threshold of 0.6", sys)



            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2
                    
            
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise CustomException(e, sys)