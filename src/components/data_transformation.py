import sys
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_object_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns=["reading score","writing score"]
            categorical_columns=[
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch","test preparation course"
            ]

            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

            ])

            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehotencoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))

            ])

            logging.info("Numerical and Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                transformers=[
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ],remainder="passthrough"
            )

            return preprocessor
           
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            preprocessing_obj=self.get_data_transformer_object()

            target_colum_name="math score"
            numerical_columns=["reading score","writing score"]


            input_feature_train_df=train_df.drop(columns=target_colum_name,axis=1)
            target_colum_name_train=train_df[target_colum_name]

            input_feature_test_df=test_df.drop(columns=target_colum_name,axis=1)
            target_colum_name_test=test_df[target_colum_name]

            logging.info("Applying preprocessing object on training and testing data")

            input_features_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_features_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_features_train_arr,np.array(target_colum_name_train)]
            test_arr=np.c_[input_features_test_arr,np.array(target_colum_name_test)]

            logging.info("Train and test data transformation completed")

            #saving the preprocessing object

            save_object(
            file_path=self.data_tranformation_config.preprocess_object_file_path,
            obj=preprocessing_obj)


            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocess_object_file_path
            )



        except Exception as e:
            raise CustomException(e,sys)










