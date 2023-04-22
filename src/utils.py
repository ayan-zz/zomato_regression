import os
import sys
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logging import logging
   
def time_to_float(time_ordered):
    try:
        ddt=time_ordered
        #dst=[]
        #for i in range(0,len(ddt)):
        dsst=str(ddt)
        spt = dsst.split(':')
        if len(spt) == 2:
            sst = spt[0] + '.' + spt[1]
        elif len(spt) == 3:
            sst = spt[0] + '.' + spt[1]
        else:
            sst=spt[0]
        #dst.append(sst)
        dst=float(sst)
        return dst
    
    except Exception as e:
        raise CustomException (e,sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        logging.info('report generation for model score initiated')
        report={}
        for i in range(len(models)):
            #select model
            model=list(models.values())[i]
            #model training
            model.fit(X_train,y_train)

            #predict data
            y_pred=model.predict(X_test)

            #find model performance
            model_score=r2_score(y_test,y_pred)
            report[list(models.keys())[i]]=model_score

        return report
    except Exception as e:
        logging.info('error occured in finding model score')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        logging.info('load object file path initiated')
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('error in loading file path')
        raise CustomException(e,sys)


        