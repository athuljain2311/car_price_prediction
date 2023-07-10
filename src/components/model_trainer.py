import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting Training And Test Input Data')
            x_train,y_train,x_test,y_test = train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]

            models = {
                "Linear Regression" : LinearRegression(),
                "Lasso Regression" : Lasso(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Ada Boost" : AdaBoostRegressor(),
                "Gradient Boost" : GradientBoostingRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "XGBRegressor" : XGBRegressor()
            }

            model_report = evaluate_models(x_train,y_train,x_test,y_test,models)
            best_model_name = sorted(model_report.items(),key=lambda m:m[1],reverse=True)[0][0]
            best_model = models[best_model_name]

            if model_report[best_model_name]<0.6:
                raise CustomException("No Best Model Found",sys)
            logging.info('Best Model Found On Train And Test Data')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            return r2_score(y_test,predicted)

        except Exception as e:
            raise CustomException(e,sys)