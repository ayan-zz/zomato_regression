from src.components.data_ingestion import dataingestion
from src.components.data_transformation import datatransformation
from src.components.model_trainer import modeltrainer

if '__name__'=='__main__':
    obj=dataingestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_trans=datatransformation()
    train_arr,test_arr,_=data_trans.initiate_data_transformation(train_data,test_data)
    model_train=modeltrainer()
    model_train.initiate_model_training(train_arr,test_arr)

