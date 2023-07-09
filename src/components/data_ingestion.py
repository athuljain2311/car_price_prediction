import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# USEFUL JUST FOR DEFINING CLASS VARIABLES
@dataclass
class DataIngestionConfig:
    # CLASS USED TO PROVIDE INPUTS
    raw_data_path = os.path.join('artifacts','data.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered The Data Ingestion Component')
        try:
            df = pd.read_csv('data\car_data.csv')
            logging.info('Successfully Read The Data')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Initiating Train Test Split')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=23)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion Completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)
