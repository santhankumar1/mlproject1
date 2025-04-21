import os
import sys
from dataclasses import dataclass
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTranierConfig:
    trained_model_file_path=os.path.join('artificats','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTranierConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'RadomForest':RandomForestRegressor(),
                'GradientBoosting':GradientBoostingRegressor(),
                'AdaBoost':AdaBoostRegressor(),
                'LinearRegression':LinearRegression(),
                'KNeighbors':KNeighborsRegressor(),
                'DecisionTree':DecisionTreeRegressor(),
                'XGBRegressor':XGBRegressor(),
                'CatBoost':CatBoostRegressor(verbose=0),

            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            ##To get best model score from the dict
            best_model_score=max(sorted(model_report.values()))


            ##To get best name of the model from the dict
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            prediction_model=best_model.predict(x_test)
            r2_score_value=r2_score(y_test,prediction_model)
            return r2_score_value
        

        except Exception as e:
            raise CustomException(e,sys) from e
        




