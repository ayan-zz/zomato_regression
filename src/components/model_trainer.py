import os,sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.utils import evaluate_models,save_object
from src.logging import logging
from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class modeltrainingconfig:
    model_path=os.path.join('artifacts','model.pkl')

class modeltrainer:
    def __init__(self):
        self.model_trainer_obj=modeltrainingconfig()
    def initiate_model_training(self, train_array,test_array):
        try:
            logging.info('model training inititaiated')
            X_train,y_train,X_test,y_test=(train_array[:,:-1],
                                           train_array[:,-1],
                                           test_array[:,:-1],
                                           test_array[:,-1])
            models={
                'linear_regression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elastic-Net': ElasticNet(),
                'Decision-Tree': DecisionTreeRegressor(),
                'Random-forest': RandomForestRegressor()
            }

            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)

            best_score=max(sorted(model_report.values()))
            best_algo=list(model_report.keys())[list(model_report.values()).index(best_score)]

            best_model = models[best_algo]
            print('\n')
            print(f'The best ML alorithm for this regression problem is {best_algo} having R2-score of: {best_score}')

            logging.info('model training ended and saving the model path')

            save_object(self.model_trainer_obj.model_path,obj=best_model)

        except Exception as e:
            logging.info('error in model training')
            raise CustomException(e,sys)
           