from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer
from network_security.entity.config_entity import DataIngestionConfig , DataValidationConfig , DataTransformationConfig , ModelTrainerConfig
from network_security.entity.config_entity import TrainingPipelineConfig
from network_security.entity.artifact_entity import DataIngestionArtifact
from network_security.entity.artifact_entity import DataValidationArtifact
from network_security.entity.artifact_entity import DataTransformationArtifact

import sys

if __name__=="__main__":
    try:
        training_data_pipeline = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_data_pipeline)
        logging.info("Initiating data ingestion")
        data_ingestion = DataIngestion(data_ingestion_config)
        dataingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed")
        
        data_validation_config = DataValidationConfig(training_data_pipeline)
        data_validation = DataValidation(dataingestion_artifact , data_validation_config)
        logging.info("Initiating data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("data Validation Completed")

        data_transformation_config = DataTransformationConfig(training_data_pipeline)
        data_transformation = DataTransformation(data_validation_artifact ,data_transformation_config )
        logging.info("Initiating data transformation")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed")

        model_trainer_config = ModelTrainerConfig(training_data_pipeline)
        model_trainer = ModelTrainer(model_trainer_config ,data_transformation_artifact )
        logging.info("Initiating model training")
        model_training_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed")
        
        
        print(model_training_artifact)
    except Exception as e:
        raise NetworkSecurityException(e,sys)