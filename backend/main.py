from fastapi import FastAPI 
from fastapi .middleware .cors import CORSMiddleware 
from pydantic import BaseModel 
import joblib 
import numpy as np 
import pandas as pd 
import os 
import tensorflow as tf 
from tensorflow .keras .models import load_model 
from tensorflow .keras .preprocessing .image import ImageDataGenerator 
from sklearn .preprocessing import LabelEncoder 
from dotenv import load_dotenv 
import requests 
import json 
from sklearn .preprocessing import MinMaxScaler 
import os 
import numpy as np 
import tensorflow as tf 
from PIL import Image 
import io 
import base64 
import uuid 

app =FastAPI ()

origins =[
"http://localhost:3000",
"http://127.0.0.1:3000",
]

app .add_middleware (
CORSMiddleware ,
allow_origins =["*"],
allow_credentials =True ,
allow_methods =["*"],
allow_headers =["*"],
)

print("Starting server...")

CROP_YIELD_MODEL = None
SOIL_CROP_RECOMMENDATION_MODEL = None
DISEASE_DETECTION_MODEL = None
DISEASE_CLASSES = None
FERTILIZER_RECOMMENDATION_MODEL = None

try:
    print("Loading Crop Yield Model...")
    CROP_YIELD_MODEL =joblib .load ('Trained_models/CROP_YIELD_MODEL.joblib')
    print("Crop Yield Model loaded successfully")
except Exception as e:
    print("ERROR LOADING CROP YIELD MODEL:", e)

try:
    print("Loading Soil Crop Recommendation Model...")
    SOIL_CROP_RECOMMENDATION_MODEL =joblib .load ('Trained_models/Soil_crop_recom.joblib')
    print("Soil Crop Recommendation Model loaded successfully")
except Exception as e:
    print("ERROR LOADING SOIL CROP RECOMMENDATION MODEL:", e)

try:
    print("Loading Disease Detection Model...")
    DISEASE_DETECTION_MODEL =load_model ('Trained_models/CNN/Disease_Detection_model[CNN].h5')
    print("Disease Detection Model loaded successfully")
except Exception as e:
    print("ERROR LOADING DISEASE DETECTION MODEL:", e)

try:
    print("Loading Disease Classes...")
    DISEASE_CLASSES =np .load ('Trained_models/CNN/disease_classes.npy',allow_pickle =True )
    print("Disease Classes loaded successfully")
except Exception as e:
    print("ERROR LOADING DISEASE CLASSES:", e)

try:
    print("Loading Fertilizer Recommendation Model...")
    FERTILIZER_RECOMMENDATION_MODEL =joblib .load ('Trained_models/fertilizer_recommendation_model.joblib')
    print("Fertilizer Recommendation Model loaded successfully")
except Exception as e:
    print("ERROR LOADING FERTILIZER RECOMMENDATION MODEL:", e)

load_dotenv ()
WEATHER_API_KEY =os .getenv ('WEATHER_API_KEY')
MARKET_MODEL_PATH ='Trained_models/lstm_model.keras'
MARKET_DATA_PATH ='Datasets/synthetic_crop_prices_2years.csv'

@app .get ("/")
async def root ():
    return {"message":"Welcome to the CROPIX API!"}

class CropYieldInput (BaseModel ):
    N :float 
    P :float 
    K :float 
    temperature :float 
    humidity :float 
    ph :float 
    rainfall :float 
    Crop :str 
    Season :str 
    Area :float 
    Fertilizer :float 
    Crop_Year :int 
    Pesticide :float 
    Annual_Rainfall :float 
    Production :float =None 

@app .post ("/predict_crop_yield/")
async def predict_crop_yield (input :CropYieldInput ):
    if CROP_YIELD_MODEL is None:
        return {"error": "Crop yield model is currently unavailable."}

    production =input .Production if input .Production is not None else input .Area *1000 

    model_input =pd .DataFrame ([{
    'Area':input .Area ,
    'Production':production ,
    'Annual_Rainfall':input .Annual_Rainfall ,
    'Fertilizer':input .Fertilizer ,
    'Pesticide':input .Pesticide 
    }])

    prediction =CROP_YIELD_MODEL .predict (model_input )[0 ]
    return {"predicted_yield":float (prediction )}

class SoilCropRecommendationInput (BaseModel ):
    N :float 
    P :float 
    K :float 
    temperature :float 
    humidity :float 
    ph :float 
    rainfall :float 

@app .post ("/recommend_soil_crop/")
async def recommend_soil_crop (input :SoilCropRecommendationInput ):
    if SOIL_CROP_RECOMMENDATION_MODEL is None:
        return {"error": "Soil crop recommendation model is currently unavailable."}

    df_input =pd .DataFrame ([input .dict ()])
    prediction_label =SOIL_CROP_RECOMMENDATION_MODEL .predict (df_input )[0 ]
    return {"recommended_crop":prediction_label }

class DiseaseDetectionInput (BaseModel ):
    image_base64 :str 

@app .post ("/detect_disease/")
async def detect_disease (input :DiseaseDetectionInput ):
    if DISEASE_DETECTION_MODEL is None or DISEASE_CLASSES is None:
        return {"error": "Disease detection model is currently unavailable."}

    from PIL import Image 
    import io 
    import base64 
    import uuid 

    try :
        image_data =base64 .b64decode (input .image_base64 )
        image =Image .open (io .BytesIO (image_data ))

        temp_filename =f"temp_image_{uuid.uuid4()}.png"
        temp_filepath =os .path .join ("temp_images",temp_filename )
        os .makedirs ("temp_images",exist_ok =True )
        image .save (temp_filepath )

        img =tf .keras .preprocessing .image .load_img (temp_filepath ,target_size =(256 ,256 ))
        img_array =tf .keras .preprocessing .image .img_to_array (img )
        img_array =np .expand_dims (img_array ,axis =0 )
        img_array /=255.0 

        predictions =DISEASE_DETECTION_MODEL .predict (img_array )
        predicted_class_index =np .argmax (predictions )
        predicted_class =DISEASE_CLASSES [predicted_class_index ]
        confidence =float (np .max (predictions ))

        os .remove (temp_filepath )
        return {"predicted_disease":predicted_class ,"confidence":confidence }
    except Exception as e :
        return {"error":str (e )}

class LSTMWeatherForecastInput (BaseModel ):
    city :str 
    days :int 

@app .post ("/weather_forecast_lstm/")
async def weather_forecast_lstm (input :LSTMWeatherForecastInput ):
    if not WEATHER_API_KEY :
        return {"error":"Weather API key not found."}

    url =f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={input.city}&days={input.days}"
    response =requests .get (url )
    response .raise_for_status ()
    data =response .json ()['forecast']['forecastday']

    forecast_summary =[]
    for day_data in data :
        date =day_data ['date']
        day_info =day_data ['day']
        forecast_summary .append ({
        "date":date ,
        "min_temp_c":day_info ['mintemp_c'],
        "max_temp_c":day_info ['maxtemp_c'],
        "avg_temp_c":day_info ['avgtemp_c'],
        "avg_humidity":day_info ['avghumidity'],
        "chance_of_rain":day_info ['daily_chance_of_rain'],
        "condition":day_info ['condition']['text']
        })
    return {"forecast":forecast_summary }

class WeatherForecastInput (BaseModel ):
    city :str 
    days :int 

@app .post ("/weather_forecast/")
async def weather_forecast (input :WeatherForecastInput ):
    if not WEATHER_API_KEY :
        return {"error":"Weather API key not found."}

    url =f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={input.city}&days={input.days}"
    response =requests .get (url )
    response .raise_for_status ()
    data =response .json ()['forecast']['forecastday']

    forecast_summary =[]
    for day_data in data :
        date =day_data ['date']
        day_info =day_data ['day']
        forecast_summary .append ({
        "date":date ,
        "min_temp_c":day_info ['mintemp_c'],
        "max_temp_c":day_info ['maxtemp_c'],
        "avg_temp_c":day_info ['avgtemp_c'],
        "avg_humidity":day_info ['avghumidity'],
        "chance_of_rain":day_info ['daily_chance_of_rain'],
        "condition":day_info ['condition']['text']
        })
    return {"forecast":forecast_summary }

class FertilizerRecommendationInput (BaseModel ):
    Crop :str 
    Current_N :float 
    Current_P :float 
    Current_K :float 

@app .post ("/recommend_fertilizer/")
async def recommend_fertilizer (input :FertilizerRecommendationInput ):
    if FERTILIZER_RECOMMENDATION_MODEL is None:
        return {"error": "Fertilizer recommendation model is currently unavailable."}

    df_input =pd .DataFrame ([input .dict ()])
    prediction =FERTILIZER_RECOMMENDATION_MODEL .predict (df_input )[0 ]
    return {"recommended_N":prediction [0 ].item (),"recommended_P":prediction [1 ].item (),"recommended_K":prediction [2 ].item ()}

class MarketPriceForecastInput (BaseModel ):
    crop_name :str 
    weeks_to_forecast :int 

@app .post ("/forecast_market_prices/")
async def forecast_market_prices (input :MarketPriceForecastInput ):
    try:
        model =load_model (MARKET_MODEL_PATH )
    except Exception as e:
        print("ERROR LOADING MARKET MODEL:", e)
        return {"error": "Market forecast model is currently unavailable."}

    try:
        df_historical =pd .read_csv (MARKET_DATA_PATH )
    except Exception as e:
        print("ERROR LOADING MARKET DATA:", e)
        return {"error": "Market historical data is currently unavailable."}

    date_columns =['Month','Month Name']
    for col in date_columns :
        if col in df_historical .columns :
            df_historical =df_historical .drop (columns =[col ])

    crop_name_mapping ={
    'Wheat':'Wheat (INR/Quintal)',
    'Rice':'Paddy (Rice) (INR/Quintal)',
    'Maize':'Maize (INR/Quintal)',
    'Soybean':'Soybean (INR/Quintal)',
    'Gram':'Gram (Chana) (INR/Quintal)',
    'Mustard':'Mustard (INR/Quintal)'
    }

    mapped_crop_name =crop_name_mapping .get (input .crop_name ,input .crop_name )

    if mapped_crop_name not in df_historical .columns :
        return {"error":f"Crop '{input.crop_name}' not found in historical data. Available crops: {list(crop_name_mapping.keys())}"}

    crop_data =df_historical [[mapped_crop_name ]].values 

    scaler =MinMaxScaler (feature_range =(0 ,1 ))
    scaled_data =scaler .fit_transform (crop_data )

    n_steps =12 
    n_features =1 

    last_known_data =scaled_data [-n_steps :]
    current_batch =last_known_data .reshape ((1 ,n_steps ,n_features ))
    forecast =[]

    for _ in range (input .weeks_to_forecast ):
        current_pred =model .predict (current_batch ,verbose =0 )[0 ]
        forecast .append (current_pred )
        current_batch =np .append (current_batch [:,1 :,:],[[current_pred ]],axis =1 )

    forecast_prices =scaler .inverse_transform (np .array (forecast ).reshape (-1 ,1 ))

    base_date =pd .to_datetime ('today')
    future_dates =[base_date +pd .Timedelta (weeks =i )for i in range (1 ,input .weeks_to_forecast +1 )]

    forecast_dict ={
    input .crop_name :{
    date .strftime ('%Y-%m-%d'):float (price [0 ])
    for date ,price in zip (future_dates ,forecast_prices )
    }
    }

    return {"forecast":forecast_dict }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)