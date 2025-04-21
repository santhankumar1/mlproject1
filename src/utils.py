import os
import pandas as pd
import numpy as np
import sys
import dill
from sklearn.metrics import r2_score

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
    


def evaluate_model(x_train,y_train,x_test,y_test,models):
    """Evaluate the model and return the best model score"""
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]

            model.fit(x_train,y_train)

            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)
            
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score
            logging.info(f"{model} model score:{test_model_score}")

        return report


    except Exception as e:
        raise CustomException(e,sys) from e


