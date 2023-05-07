import pandas as pd
import numpy as np
import os,sys
from src.logging import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from src.utils import save_object,time_to_float
from dataclasses import dataclass

@dataclass
class datatransformationconfig:
    preprocesser_path=os.path.join('artifacts','preprocessor.pkl')

class datatransformation:
    def __init__(self):
        self.transformation_config=datatransformationconfig()

    def get_transformation_obj(self):
        try:
            logging.info('data transformation initiated')
            num_col=['Delivery_person_Age','Delivery_person_Ratings',
                     'Restaurant_latitude','Restaurant_longitude',
                     'Delivery_location_latitude','Delivery_location_longitude']
            time_col=['Time_Orderd','Time_Order_picked']         
            cat_col=['Weather_conditions','Road_traffic_density','Vehicle_condition','Type_of_order',
                      'Type_of_vehicle','multiple_deliveries','Festival','City']

            target_col=['Time_taken_min']

            logging.info('pipeline formation initiated')

            numerical_pipeline=Pipeline(
                steps=[('imputer',SimpleImputer(missing_values=np.nan,strategy='median')),
                       ('scaling',StandardScaler())])
            
            time_pipeline=Pipeline(steps=[('imputer',SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
                                          ('time_transformer', TimeTransformer()),
                                          ('scaling',StandardScaler())])
            
            categorical_pipeline=Pipeline(
                steps=[('imputer',SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
                       ('encoder',OneHotEncoder(handle_unknown='ignore')),
                       ('scaling',StandardScaler(with_mean=False))])
            
            preprocessor=ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,num_col),
                ('categorical_pipeline',categorical_pipeline,cat_col),
                ('time_pipeline',time_pipeline,time_col)])
            
            return preprocessor
        
        except Exception as e:
            logging.info('error in data transformation initiation')
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            df_train=pd.read_csv(train_path)
            df_test=pd.read_csv(test_path)

            df_train['Time_Orderd'] = df_train.apply(lambda row:time_to_float(row['Time_Orderd']),axis=1)
            df_test['Time_Orderd'] = df_test.apply(lambda row:time_to_float(row['Time_Orderd']),axis=1)
            df_train['Time_Order_picked'] = df_train.apply(lambda row:time_to_float(row['Time_Order_picked']),axis=1)
            df_test['Time_Order_picked'] = df_train.apply(lambda row:time_to_float(row['Time_Order_picked']),axis=1)

            logging.info("Read Train ans test data completed")
            logging.info("obtaining preprocesser object")

            preprocessor_obj=self.get_transformation_obj()

            
            drop_col=['ID','Order_Date','Time_taken_min']
            target_col=['Time_taken_min']

        
            print(df_train.head(2))

            target_feature_train_df=df_train[target_col]
            input_feature_train_df=df_train.drop(columns=drop_col,axis=1)

            target_feature_test_df=df_test[target_col]
            input_feature_test_df=df_test.drop(columns=drop_col,axis=1)

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,target_feature_train_df]
            test_arr=np.c_[input_feature_test_arr,target_feature_test_df]

            logging.info('data transformation ended and saving the preprocessor obj')

            save_object(
                file_path=self.transformation_config.preprocesser_path,
                obj=preprocessor_obj
            )

            return(train_arr,test_arr,self.transformation_config.preprocesser_path)
        
        except Exception as e:
            logging.info('error in data transformation')
            raise CustomException(e,sys)
        
class TimeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def fit(self, X, y=None):

        return self

    def transform(self, X):
        def time_to_float(time_ordered):
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
        
        time_strings = [x[0] for x in X]

        time_floats = [time_to_float(time_str) for time_str in time_strings]
        return [[time_float] for time_float in time_floats]




