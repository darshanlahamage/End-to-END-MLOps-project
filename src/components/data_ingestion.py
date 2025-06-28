import os 
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
 

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts', "train.csv")
    test_data_path : str = os.path.join('artifacts', "test.csv")
    raw_data_path : str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Read the dataset as dataframe")
            df = pd.read_csv("notebook/data/stud.csv")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
  
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__== "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_= data_transformation.intiate_data_transformation(train_data, test_data)
    model_trainer = ModelTrainer()
    r2_score, best_params = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(r2_score)
    logging.info(f"R2_score of the selected model is {r2_score} with best params - {best_params}")

    

        




