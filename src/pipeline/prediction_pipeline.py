import os,sys
import pandas as pd
import numpy as np
from src.utils import load_object
from src.logging import logging
from src.exception import CustomException


class Predictionpipeline:
    def __init__(self):
        pass
        
    def prediction(self,features):
        try:
            logging.info('prediction of test data initiated')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            scaled=preprocessor.transform(features)
            
            predict=model.predict(scaled)
            logging.info('prediction ended')
            return predict
        
        except Exception as e:
            logging.info('error in path joining or cant predict')
            raise CustomException (e,sys)
        
class Customdata:
    def __init__(self,
                 ID:str,
                 Delivery_person_ID:str,
                 Delivery_person_Age:int,
                 Delivery_person_Ratings:float,
                 Restaurant_latitude:float,
                 Restaurant_longitude:float,
                 Delivery_location_latitude:float,
                 Delivery_location_longitude:float,
                 Order_Date:str,
                 Time_Orderd:str,
                 Time_Order_picked:str,
                 Weather_conditions:str,
                 Road_traffic_density:str,
                 Vehicle_condition:int,
                 Type_of_order:str,
                 Type_of_vehicle:str,
                 multiple_deliveries:int,
                 Festival:str,
                 City:str):
        self.ID=ID
        self.Delivery_person_ID=Delivery_person_ID
        self.Delivery_person_Age=Delivery_person_Age
        self.Delivery_person_Ratings=Delivery_person_Ratings
        self.Restaurant_latitude=Restaurant_latitude
        self.Restaurant_longitude=Restaurant_longitude
        self.Delivery_location_latitude=Delivery_location_latitude
        self.Delivery_location_longitude=Delivery_location_longitude
        self.Order_Date=Order_Date
        self.Time_Orderd=Time_Orderd
        self.Time_Order_picked=Time_Order_picked
        self.Weather_conditions=Weather_conditions
        self.Road_traffic_density=Road_traffic_density
        self.Vehicle_condition=Vehicle_condition
        self.Type_of_order=Type_of_order
        self.Type_of_vehicle=Type_of_vehicle
        self.multiple_deliveries=multiple_deliveries
        self.Festival=Festival
        self.City=City

    def get_dataframe(self):
        try:
            custom_input_dict={
                'ID':[self.ID],
                'Delivery_person_ID':[self.Delivery_person_ID],
                'Delivery_person_Age':[self.Delivery_person_Age],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Restaurant_latitude':[self.Restaurant_latitude],
                'Restaurant_longitude':[self.Restaurant_longitude],
                'Delivery_location_latitude':[self.Delivery_location_latitude],
                'Delivery_location_longitude':[self.Delivery_location_longitude],
                'Order_Date':[self.Order_Date],
                'Time_Orderd':[self.Time_Orderd],
                'Time_Order_picked':[self.Time_Order_picked],
                'Weather_conditions':[self.Weather_conditions],
                'Road_traffic_density':[self.Road_traffic_density],
                'Vehicle_condition':[self.Vehicle_condition],
                'Type_of_order':[self.Type_of_order],
                'Type_of_vehicle':[self.Type_of_vehicle],
                'multiple_deliveries':[self.multiple_deliveries],
                'Festival':[self.Festival],
                'City':[self.City]
                }
            df=pd.DataFrame(custom_input_dict)
            logging.info('dataframe created')
            return df
        except Exception as e:
            logging.info('error in the dataframe formation')
            raise CustomException(e,sys)
     





    
