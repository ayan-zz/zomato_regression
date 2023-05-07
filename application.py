from flask import Flask,request,render_template
from src.pipeline.prediction_pipeline import Predictionpipeline,Customdata

application=Flask(__name__)

app=application
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=Customdata(
            ID=request.form.get('ID'),
            Delivery_person_ID=request.form.get('Delivery_person_ID'),
            Delivery_person_Age=int(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings=float(request.form.get('Delivery_person_Ratings')),
            Restaurant_latitude=float(request.form.get('Restaurant_latitude')),
            Restaurant_longitude=float(request.form.get('Restaurant_longitude')),
            Delivery_location_latitude=float(request.form.get('Delivery_location_latitude')),
            Delivery_location_longitude=float(request.form.get('Delivery_location_longitude')),
            Order_Date=str(request.form.get('Order_Date')),
            Time_Orderd=str(request.form.get('Time_Orderd')),
            Time_Order_picked=str(request.form.get('Time_Order_picked')),
            Weather_conditions=str(request.form.get('Weather_conditions')),
            Road_traffic_density=str(request.form.get('Road_traffic_density')),
            Vehicle_condition=int(request.form.get('Vehicle_condition')),
            Type_of_order=str(request.form.get('Type_of_order')),
            Type_of_vehicle=str(request.form.get('Type_of_vehicle')),
            multiple_deliveries=str(request.form.get('multiple_deliveries')),
            Festival=str(request.form.get('Festival')),
            City=str(request.form.get('City')))
        

        
        final_data=data.get_dataframe()
        print(final_data)
        predict_pipeline=Predictionpipeline()
        pred=predict_pipeline.prediction(final_data)

        results=pred[0]

        return render_template('home.html',results=results)
    
if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)


        
