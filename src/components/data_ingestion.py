import os
import sys
from src.exception import CustomException
from src.logger import logging
from data_transformation import DataTransformation
import pandas as pd
from model_trainer import ModelTrainer

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#sys.path.append("D:\Projects\FullStack_Projects\car_price_prediction\src\components")

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
        
if __name__=='__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    transformer_obj = DataTransformation()
    train_arr,test_arr,_ = transformer_obj.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    logging.info(f"R2 Score : {model_trainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr)}")