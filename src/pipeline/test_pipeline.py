import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'  # Corrected path
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            print("Model Path:", model_path)  # Debugging
            print("Preprocessor Path:", preprocessor_path)  # Debugging
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Error in Predict pipeline")
            raise CustomException(e, sys) from e
        

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str,
                 lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],  # Match the expected column name
                "parental level of education": [self.parental_level_of_education],  # Match the expected column name
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],  # Match the expected column name
                "reading score": [self.reading_score],  # Match the expected column name
                "writing score": [self.writing_score]  # Match the expected column name
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Custom data to dataframe conversion is successful")
            return df
        except Exception as e:
            logging.info("Custom data to dataframe conversion failed")
            raise CustomException(e, sys) from e


