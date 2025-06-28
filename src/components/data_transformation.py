import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        """
           This function does catagorical and numerical transformation
      
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            catagorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_tranformer_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scalar", StandardScaler())
                ]
            )

            cat_transformer_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scalar", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_tranformer_pipeline, numerical_columns),
                    ("cat_pipeline", cat_transformer_pipeline, catagorical_columns)
                ]
            )
            logging.info(f"Numerical columns {numerical_columns}")
            logging.info(f"catagorical columns {catagorical_columns}")

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def intiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("obtaining preprocessing order")
            preprocessing_obj = self.get_data_transformer_obj()

            target_column = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df= train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing on train and test df")

            inp_feature_train_preprocessed = preprocessing_obj.fit_transform(input_feature_train_df)
            inp_feature_test_preprocessed = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                inp_feature_train_preprocessed, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                inp_feature_test_preprocessed, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)



