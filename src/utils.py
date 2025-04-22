import os
import pandas as pd
import numpy as np
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException

def save_object(file_path,obj):
    "save the object to a file"
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
        logging.info('Object saved successfully')

    except Exception as e:
        raise CustomException(e,sys) from e
    

def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    """Evaluate the model and return the best model score"""
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Starting grid search for {model_name}")
            param_grid = params[model_name]

            gs = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            logging.info(f"{model_name} model score:{test_model_score}")

        return report

    except Exception as e:
        raise CustomException(e, sys) from e
    
    
def load_object(file_path):
    "load the file path"
    try:

        with open(file_path,'rb') as file_obj:
             return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys) 


