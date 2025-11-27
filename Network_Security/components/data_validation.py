from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.entity.config_entity import DataValidationConfig
from network_security.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
import pandas as pd
import numpy as np
import os,sys
from network_security.utils.main_utils.utils import read_yaml_file,write_yaml_file
from network_security.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self , data_ingestion_artifact:DataIngestionArtifact , config: DataValidationConfig ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.config = config
            self._schema_config = SCHEMA_FILE_PATH
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    @staticmethod
    def read_data(filepath) -> pd.DataFrame:
        try:
            logging.info("Reading csv file")
            return pd.read_csv(filepath)
        except Exception as e:
            raise NetworkSecurityException(e,sys)


    def validate_number_of_columns(self , dataframe: pd.DataFrame) -> bool:
        try:
            logging.info("Validating columns length")
            schema_columns_len = len(self._schema_config)
            df_columns_len = len(dataframe.columns)
            if schema_columns_len==df_columns_len:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)


    def detect_dataset_drift(self , base_dataset , target_dataset , threshold =0.05 ) -> bool:
        try:
            drift_detection_status = False
            report = {}
            for column in base_dataset.columns:
                d1 = base_dataset[column]
                d2 = target_dataset[column]
                result = ks_2samp(d1,d2)
                if threshold<= result.pvalue:
                    is_found = False
                else:
                    drift_detection_status = True
                    is_found = True

                report.update({
                    column : {
                        "p_value": float(result.pvalue),
                        "is_found": is_found
                    }
                })

            drift_report_file_path = self.config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path , exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)
            return drift_detection_status
       
        except Exception as e:
            raise NetworkSecurityException(e,sys)
                    

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
    
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
    
            validate_column_status = self.validate_number_of_columns(train_dataframe)
            if not validate_column_status:
                error_message = f"Train data frame does not contain all the columns"
    
            validate_column_status = self.validate_number_of_columns(test_dataframe)
            if not validate_column_status:
                error_message= f"Test data frame does not contain all the columns"
    
            data_drift_status = self.detect_dataset_drift(train_dataframe , test_dataframe)
            dir_path = os.path.dirname(self.config.valid_train_file_path)
            os.makedirs(dir_path , exist_ok=True)
    
            train_dataframe.to_csv(self.config.valid_train_file_path , index=False, header=True)
            test_dataframe.to_csv(self.config.valid_test_file_path , index=False , header=True)
    
            data_validation_artifact = DataValidationArtifact(
                validation_status = data_drift_status,
                valid_train_file_path = train_file_path,
                valid_test_file_path=test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.config.drift_report_file_path,
                
            )
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)


