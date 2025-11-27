from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.entity.config_entity import DataIngestionConfig
from network_security.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

class DataIngestion:
    def __init__(self , config : DataIngestionConfig):
        try:
            self.config = config
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def collection_to_df(self):
        try:
            logging.info("creating Dataframe ")
            database_name = self.config.database_name
            collection_name = self.config.collection_name
            logging.info(f"Trying to connect to db with name {database_name} witn collection {collection_name}")
            self.mongo_client = pymongo.MongoClient(MONGO_URI)
            logging.info(f"Connected to MongoDB: {MONGO_URI}")
            collection = self.mongo_client[database_name][collection_name]
            count = collection.count_documents({})
            logging.info(f"Found {count} documents in the collection.")

            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Loaded data into DataFrame. Shape: {df.shape}")
            logging.info(f"Data preview:\n{df.head()}")
            if "_id" in df.columns.to_list():
                df = df.drop(columns=['_id'] , axis=1)

            df.replace({"na":np.nan},inplace=True)
            logging.info("Dataframe created")
            return df
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def data_to_feature_store(self , dataframe : pd.DataFrame ):
        try:
            logging.info("Creating feature store")
            feature_store_file_path = self.config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path , exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            logging.info("feature store created")
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e,sys)
            
    def split_train_test(self , dataframe: pd.DataFrame):
        try:
            logging.info("splitting data")
            train_test_split_ratio = self.config.train_test_split_ratio
            train_set , test_set = train_test_split(dataframe , test_size = train_test_split_ratio)
            dir_path = os.path.dirname(self.config.training_file_path)
            os.makedirs(dir_path , exist_ok=True)

            logging.info("saving training and testing sets")
            train_set.to_csv(self.config.training_file_path , index=False, header=True )
            test_set.to_csv(self.config.testing_file_path , index=False , header=True)
            logging.info("split completed")
        except Exception as e:
            raise NetworkSecurityException(e,sys)


    def initiate_data_ingestion(self):
        try:
            dataframe = self.collection_to_df()
            dataframe = self.data_to_feature_store(dataframe)
            self.split_train_test(dataframe)
            dataingestionartifact = DataIngestionArtifact(trained_file_path = self.config.training_file_path  , test_file_path = self.config.testing_file_path)
            return dataingestionartifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
            