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

from network_security.cloud.s3_syncer import S3Sync
import os,sys

from network_security.constants.training_pipeline import TRAINING_BUCKET_NAME

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(config = self.data_ingestion_config )
            logging.info("Data ingestion stage started")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion stage completed")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def start_data_validation(self , data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validation = DataValidation(config=self.data_validation_config , data_ingestion_artifact= data_ingestion_artifact )
            logging.info("Data validation stage started")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation stage completed")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys) 

    def start_data_transformation(self , data_validation_artifact : DataValidationArtifact):
        try:
            self.data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformation = DataTransformation(artifact = data_validation_artifact ,  config = self.data_transformation_config)
            logging.info("Data transformation stage started")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation stage completed")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def start_model_trainer(self , data_transformation_artifact : DataTransformationArtifact):
        try:
            self.model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer = ModelTrainer(config = self.model_trainer_config , artifact =data_transformation_artifact )
            logging.info("Model training stage started")
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model training stage completed")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def sync_artifact_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder= self.training_pipeline_config.artifact_dir , aws_bucket_url = aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def sync_saved_model_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder= self.training_pipeline_config.model_dir , aws_bucket_url = aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact = data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact = data_transformation_artifact)

            self.sync_artifact_to_s3()
            self.sync_saved_model_to_s3()

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
        