
import os
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer

from network_security.entity.config_entity import (TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from network_security.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)


import os,sys

from network_security.constants.training_pipeline import TRAINING_BUCKET_NAME


# class S3Sync:
#     def sync_folder_to_s3(self,folder,aws_bucket_url):
#         command = f"aws s3 sync {folder} {aws_bucket_url} "
#         os.system(command)

#     def sync_folder_from_s3(self,folder,aws_bucket_url):
#         command = f"aws s3 sync  {aws_bucket_url} {folder} "
#         os.system(command)

def sync_artifact_to_s3():
        try:
            aws_bucket_url = f"s3://bucket-networksecurity/artifact/artifact"
            #self.s3_sync.sync_folder_to_s3(folder= self.training_pipeline_config.artifact_dir , aws_bucket_url = aws_bucket_url)
            command = f"aws s3 sync Artifacts {aws_bucket_url} "
            os.system(command)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

def sync_saved_model_to_dir():
        try:
            aws_bucket_url = f"s3://bucket-networksecurity/final_model/final_model"
            #self.s3_sync.sync_folder_to_s3(folder= self.training_pipeline_config.model_dir , aws_bucket_url = aws_bucket_url)
            command = f"aws s3 sync final_model {aws_bucket_url} "
            os.system(command)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

if __name__ == "__main__":
    try:
        sync_artifact_to_s3()
        sync_saved_model_to_dir()
        print("Sync completed successfully!")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
