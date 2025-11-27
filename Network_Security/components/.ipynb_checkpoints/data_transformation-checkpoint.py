from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.constants.training_pipeline import TARGET_COLUMN
from network_security.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
import os,sys
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from network_security.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from network_security.entity.config_entity import DataTransformationConfig
from network_security.utils.main_utils.utils import save_numpy_array_data,save_object




class DataTransformation:
    def __init__(self , artifact : DataValidationArtifact , config: DataTransformationConfig ):
        try:
            self.artifact = artifact
            self.config = config
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    @staticmethod
    def read_data(filepath) -> pd.DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def get_data_transformer_object(cls) -> Pipeline:
        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor: Pipeline = Pipeline([("imputer",imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_dataframe = DataTransformation.read_data(self.artifact.valid_train_file_path)
            test_dataframe = DataTransformation.read_data(self.artifact.valid_test_file_path)
    
            #train data frame
            input_feature_train_df = train_dataframe.drop(columns = [TARGET_COLUMN] , axis = 1)
            target_feature_train_df = train_dataframe[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)
    
            #test data frame
            input_feature_test_df = test_dataframe.drop(columns = [TARGET_COLUMN] , axis =1)
            target_feature_test_df = test_dataframe[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1,0)
    
            preprocessor = self.get_data_transformer_object()
            preprocessor_object=preprocessor.fit(input_feature_train_df)  #Stores which neighbors to use for imputing missing values later
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df) #uses the info to replace missing values
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)
    
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df) ]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]
    
            #save numpy array data
            save_numpy_array_data( self.config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.config.transformed_test_file_path,array=test_arr,)
            save_object( self.config.transformed_object_file_path, preprocessor_object,)
    
            save_object( "final_model/preprocessor.pkl", preprocessor_object,)
    
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.config.transformed_object_file_path,
                transformed_train_file_path=self.config.transformed_train_file_path,
                transformed_test_file_path=self.config.transformed_test_file_path
    
            )
            return data_transformation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)