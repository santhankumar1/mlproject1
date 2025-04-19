import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

class DataIngestionConfig:
    """DataIngestionConfig is  a class that contains the configuration for data Ingestion."""
    train_data_path: str =os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.Ingestion_config=DataIngestionConfig()

    def initate_data_ingestion(self):
        logging.info("Enter the Data Ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\data.csv')
            logging.info("Read the datasets as dataframe")

            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.Ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.Ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.Ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion completed sucessfully')
            return(
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initate_data_ingestion()



    
