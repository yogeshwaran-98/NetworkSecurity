import os
import json
import sys
from dotenv import load_dotenv
import certifi

ca=certifi.where()
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

import pandas as pd
import numpy as np
import pymongo
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def csv_to_json_converter(self , file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop = True , inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def insert_data_into_mongodb(self , records , database , collection):
        try:
            self.records = records
            self.mongo_client = pymongo.MongoClient(MONGO_URI)
            self.database = self.mongo_client[database]
            self.collection = self.database[collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

if __name__=="__main__":
    file_path = "Network_Data/phishing.csv"
    database = "DB"
    collection = "NetworkData"
    extract = NetworkDataExtract()
    records = extract.csv_to_json_converter(file_path)
    len_records = extract.insert_data_into_mongodb(records , database , collection)
    print(len_records)