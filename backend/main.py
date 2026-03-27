from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from dotenv import load_dotenv
import requests
import json
from PIL import Image
import io
import base64
import uuid
import traceback

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Starting server...")

# ─── Global model references ─────────────────────────────────────────────────
CROP_YIELD_MODEL = None
SOIL_CROP_RECOMMENDATION_MODEL = None
DISEASE_DETECTION_MODEL = None
DISEASE_CLASSES = None
FERTILIZER_RECOMMENDATION_MODEL = None
FERTILIZER_LABEL_ENCODER = None

WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
MARKET_MODEL_PATH = 'Trained_models/lstm_model.keras'
MARKET_DATA_PATH = 'Datasets/synthetic_crop_prices_2years.csv'


# ─── Helper: check if a file is a valid model or just a Git LFS pointer ──────
def is_lfs_pointer(filepath):
    """Check if a file is a Git LFS pointer (text file starting with 'version https://git-lfs')"""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(48)
            return header.startswith(b'version https://git-lfs')
    except Exception:
        return False


def is_valid_model_file(filepath):
    """Check if a model file exists and is not an LFS pointer"""
    if not os.path.exists(filepath):
        return False
    if is_lfs_pointer(filepath):
        print(f"  WARNING: {filepath} is a Git LFS pointer, not actual model data")
        return False
    # Check minimum size (LFS pointers are ~130 bytes)
    if os.path.getsize(filepath) < 1000:
        print(f"  WARNING: {filepath} is too small ({os.path.getsize(filepath)} bytes), likely corrupted")
        return False
    return True


# ─── Train models if missing (handles Render where LFS files aren't pulled) ──
def train_crop_yield_model():
    """Train crop yield model from dataset"""
    print("  Training Crop Yield Model from dataset...")
    df = pd.read_csv('Datasets/Crop_yield.csv')
    X = df[['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
    y = df['Yield']
    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    os.makedirs('Trained_models', exist_ok=True)
    joblib.dump(model, 'Trained_models/CROP_YIELD_MODEL.joblib')
    print(f"  ✓ Crop Yield Model trained on {len(X)} samples")
    return model


def train_soil_crop_model():
    """Train soil crop recommendation model from dataset"""
    print("  Training Soil Crop Recommendation Model from dataset...")
    df = pd.read_csv('Datasets/Soil_Crop_recommendation.csv')
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    os.makedirs('Trained_models', exist_ok=True)
    joblib.dump(model, 'Trained_models/Soil_crop_recom.joblib')
    print(f"  ✓ Soil Crop Recommendation Model trained on {len(X)} samples")
    return model


def train_fertilizer_model():
    """Train fertilizer recommendation model from dataset.
    Returns (model, label_encoder)
    The model predicts recommended N, P, K values given Crop + Current N, P, K.
    """
    print("  Training Fertilizer Recommendation Model from dataset...")
    df = pd.read_csv('Datasets/fertilizer_training_data.csv')

    le_crop = LabelEncoder()
    df['Crop_Encoded'] = le_crop.fit_transform(df['Crop'])

    X = df[['Crop_Encoded', 'Current_N', 'Current_P', 'Current_K']]
    y = df[['Required_N', 'Required_P', 'Required_K']]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs('Trained_models', exist_ok=True)
    joblib.dump(model, 'Trained_models/fertilizer_recommendation_model.joblib')
    joblib.dump(le_crop, 'Trained_models/crop_label_encoder.joblib')
    print(f"  ✓ Fertilizer Model trained on {len(X)} samples")
    return model, le_crop


def train_lstm_market_model():
    """Train LSTM market price forecast model from dataset"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    print("  Training Market Price Forecast (LSTM) Model from dataset...")
    df = pd.read_csv(MARKET_DATA_PATH)

    wheat_col = 'Wheat (INR/Quintal)'
    if wheat_col not in df.columns:
        print("  ✗ Wheat column not found in market dataset")
        return None

    prices = df[wheat_col].values.astype(float)
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

    sequence_length = 4
    X, y = [], []
    for i in range(sequence_length, len(prices_scaled)):
        X.append(prices_scaled[i - sequence_length:i, 0])
        y.append(prices_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X, y, epochs=30, batch_size=4, verbose=0)
    model.save(MARKET_MODEL_PATH)
    print(f"  ✓ LSTM Market Model trained on {len(X)} sequences")
    return model


# ─── Load or retrain all models ──────────────────────────────────────────────
def load_all_models():
    global CROP_YIELD_MODEL, SOIL_CROP_RECOMMENDATION_MODEL
    global DISEASE_DETECTION_MODEL, DISEASE_CLASSES
    global FERTILIZER_RECOMMENDATION_MODEL, FERTILIZER_LABEL_ENCODER

    # 1. Crop Yield Model
    try:
        if is_valid_model_file('Trained_models/CROP_YIELD_MODEL.joblib'):
            print("Loading Crop Yield Model...")
            CROP_YIELD_MODEL = joblib.load('Trained_models/CROP_YIELD_MODEL.joblib')
            print("  ✓ Crop Yield Model loaded from file")
        elif os.path.exists('Datasets/Crop_yield.csv'):
            CROP_YIELD_MODEL = train_crop_yield_model()
        else:
            print("  ✗ Cannot load Crop Yield Model: no model file or dataset found")
    except Exception as e:
        print(f"  ✗ Error with Crop Yield Model: {e}")
        # Try retraining as fallback
        try:
            if os.path.exists('Datasets/Crop_yield.csv'):
                CROP_YIELD_MODEL = train_crop_yield_model()
        except Exception as e2:
            print(f"  ✗ Retraining also failed: {e2}")

    # 2. Soil Crop Recommendation Model
    try:
        if is_valid_model_file('Trained_models/Soil_crop_recom.joblib'):
            print("Loading Soil Crop Recommendation Model...")
            SOIL_CROP_RECOMMENDATION_MODEL = joblib.load('Trained_models/Soil_crop_recom.joblib')
            print("  ✓ Soil Crop Recommendation Model loaded from file")
        elif os.path.exists('Datasets/Soil_Crop_recommendation.csv'):
            SOIL_CROP_RECOMMENDATION_MODEL = train_soil_crop_model()
        else:
            print("  ✗ Cannot load Soil Crop Model: no model file or dataset found")
    except Exception as e:
        print(f"  ✗ Error with Soil Crop Model: {e}")
        try:
            if os.path.exists('Datasets/Soil_Crop_recommendation.csv'):
                SOIL_CROP_RECOMMENDATION_MODEL = train_soil_crop_model()
        except Exception as e2:
            print(f"  ✗ Retraining also failed: {e2}")

    # 3. Disease Detection Model (CNN - can't easily retrain, just try loading)
    try:
        if is_valid_model_file('Trained_models/CNN/Disease_Detection_model[CNN].h5'):
            print("Loading Disease Detection Model...")
            DISEASE_DETECTION_MODEL = load_model('Trained_models/CNN/Disease_Detection_model[CNN].h5')
            print("  ✓ Disease Detection Model loaded")
    except Exception as e:
        print(f"  ✗ Error with Disease Detection Model: {e}")

    # 4. Disease Classes
    try:
        if is_valid_model_file('Trained_models/CNN/disease_classes.npy'):
            print("Loading Disease Classes...")
            DISEASE_CLASSES = np.load('Trained_models/CNN/disease_classes.npy', allow_pickle=True)
            print(f"  ✓ Disease Classes loaded ({len(DISEASE_CLASSES)} classes)")
        elif os.path.exists('Datasets/CNN_labels.csv'):
            print("  Extracting disease classes from dataset...")
            df_cnn = pd.read_csv('Datasets/CNN_labels.csv')
            DISEASE_CLASSES = df_cnn['label'].unique()
            os.makedirs('Trained_models/CNN', exist_ok=True)
            np.save('Trained_models/CNN/disease_classes.npy', DISEASE_CLASSES)
            print(f"  ✓ Disease Classes extracted ({len(DISEASE_CLASSES)} classes)")
    except Exception as e:
        print(f"  ✗ Error with Disease Classes: {e}")

    # 5. Fertilizer Recommendation Model
    try:
        fert_model_valid = is_valid_model_file('Trained_models/fertilizer_recommendation_model.joblib')
        fert_encoder_valid = is_valid_model_file('Trained_models/crop_label_encoder.joblib')

        if fert_model_valid and fert_encoder_valid:
            print("Loading Fertilizer Recommendation Model...")
            FERTILIZER_RECOMMENDATION_MODEL = joblib.load('Trained_models/fertilizer_recommendation_model.joblib')
            FERTILIZER_LABEL_ENCODER = joblib.load('Trained_models/crop_label_encoder.joblib')
            print("  ✓ Fertilizer Model loaded from file")
        elif os.path.exists('Datasets/fertilizer_training_data.csv'):
            FERTILIZER_RECOMMENDATION_MODEL, FERTILIZER_LABEL_ENCODER = train_fertilizer_model()
        else:
            print("  ✗ Cannot load Fertilizer Model: no model file or dataset found")
    except Exception as e:
        print(f"  ✗ Error with Fertilizer Model: {e}")
        try:
            if os.path.exists('Datasets/fertilizer_training_data.csv'):
                FERTILIZER_RECOMMENDATION_MODEL, FERTILIZER_LABEL_ENCODER = train_fertilizer_model()
        except Exception as e2:
            print(f"  ✗ Retraining also failed: {e2}")

    # 6. LSTM Market Model - train if missing
    try:
        if not is_valid_model_file(MARKET_MODEL_PATH):
            if os.path.exists(MARKET_DATA_PATH):
                train_lstm_market_model()
            else:
                print("  ✗ Cannot train LSTM model: no dataset found")
        else:
            print("  ✓ LSTM Market Model file exists")
    except Exception as e:
        print(f"  ✗ Error with LSTM Market Model: {e}")


# Load all models at startup
load_all_models()

print("\n" + "=" * 60)
print("Server startup complete. Model status:")
print(f"  Crop Yield:       {'✓ loaded' if CROP_YIELD_MODEL else '✗ unavailable'}")
print(f"  Soil Crop:        {'✓ loaded' if SOIL_CROP_RECOMMENDATION_MODEL else '✗ unavailable'}")
print(f"  Disease Detection:{'✓ loaded' if DISEASE_DETECTION_MODEL else '✗ unavailable'}")
print(f"  Disease Classes:  {'✓ loaded' if DISEASE_CLASSES is not None else '✗ unavailable'}")
print(f"  Fertilizer:       {'✓ loaded' if FERTILIZER_RECOMMENDATION_MODEL else '✗ unavailable'}")
print(f"  Weather API Key:  {'✓ set' if WEATHER_API_KEY else '✗ not set'}")
print(f"  LSTM Market Model:{'✓ exists' if os.path.exists(MARKET_MODEL_PATH) else '✗ missing'}")
print("=" * 60)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "Welcome to the Farmora API!"}


# ── Crop Yield Prediction ────────────────────────────────────────────────────

class CropYieldInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    Crop: str
    Season: str
    Area: float
    Fertilizer: float
    Crop_Year: int
    Pesticide: float
    Annual_Rainfall: float
    Production: float = None


@app.post("/predict_crop_yield/")
async def predict_crop_yield(input: CropYieldInput):
    if CROP_YIELD_MODEL is None:
        return {"error": "Crop yield model is currently unavailable."}

    try:
        production = input.Production if input.Production is not None else input.Area * 1000

        model_input = pd.DataFrame([{
            'Area': input.Area,
            'Production': production,
            'Annual_Rainfall': input.Annual_Rainfall,
            'Fertilizer': input.Fertilizer,
            'Pesticide': input.Pesticide
        }])

        prediction = CROP_YIELD_MODEL.predict(model_input)[0]
        return {"predicted_yield": float(prediction)}
    except Exception as e:
        print(f"Error in predict_crop_yield: {traceback.format_exc()}")
        return {"error": f"Prediction failed: {str(e)}"}


# ── Soil Crop Recommendation ─────────────────────────────────────────────────

class SoilCropRecommendationInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.post("/recommend_soil_crop/")
async def recommend_soil_crop(input: SoilCropRecommendationInput):
    if SOIL_CROP_RECOMMENDATION_MODEL is None:
        return {"error": "Soil crop recommendation model is currently unavailable."}

    try:
        df_input = pd.DataFrame([input.dict()])
        prediction_label = SOIL_CROP_RECOMMENDATION_MODEL.predict(df_input)[0]
        return {"recommended_crop": prediction_label}
    except Exception as e:
        print(f"Error in recommend_soil_crop: {traceback.format_exc()}")
        return {"error": f"Prediction failed: {str(e)}"}


# ── Disease Detection ─────────────────────────────────────────────────────────

class DiseaseDetectionInput(BaseModel):
    image_base64: str


@app.post("/detect_disease/")
async def detect_disease(input: DiseaseDetectionInput):
    if DISEASE_DETECTION_MODEL is None or DISEASE_CLASSES is None:
        return {"error": "Disease detection model is currently unavailable."}

    try:
        image_data = base64.b64decode(input.image_base64)
        image = Image.open(io.BytesIO(image_data))

        temp_filename = f"temp_image_{uuid.uuid4()}.png"
        temp_filepath = os.path.join("temp_images", temp_filename)
        os.makedirs("temp_images", exist_ok=True)
        image.save(temp_filepath)

        img = tf.keras.preprocessing.image.load_img(temp_filepath, target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = DISEASE_DETECTION_MODEL.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = DISEASE_CLASSES[predicted_class_index]
        confidence = float(np.max(predictions))

        os.remove(temp_filepath)
        return {"predicted_disease": predicted_class, "confidence": confidence}
    except Exception as e:
        print(f"Error in detect_disease: {traceback.format_exc()}")
        return {"error": str(e)}


# ── Weather Forecast ──────────────────────────────────────────────────────────

class WeatherForecastInput(BaseModel):
    city: str
    days: int


@app.post("/weather_forecast/")
async def weather_forecast(input: WeatherForecastInput):
    if not WEATHER_API_KEY:
        return {"error": "Weather API key not found. Please set WEATHER_API_KEY environment variable."}

    try:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={input.city}&days={input.days}"
        response = requests.get(url)
        response.raise_for_status()
        full_data = response.json()
        data = full_data['forecast']['forecastday']
        current_data = full_data.get('current', {})
        location_data = full_data.get('location', {})

        forecast_summary = []
        for day_data in data:
            date = day_data['date']
            day_info = day_data['day']
            forecast_summary.append({
                "date": date,
                "min_temp_c": day_info['mintemp_c'],
                "max_temp_c": day_info['maxtemp_c'],
                "avg_temp_c": day_info['avgtemp_c'],
                "avg_humidity": day_info['avghumidity'],
                "chance_of_rain": day_info['daily_chance_of_rain'],
                "condition": day_info['condition']['text'],
                "max_wind_kph": day_info.get('maxwind_kph', current_data.get('wind_kph', 0)),
                "pressure_mb": current_data.get('pressure_mb', 0)
            })
        return {
            "forecast": forecast_summary,
            "city_name": location_data.get('name', input.city)
        }
    except requests.exceptions.HTTPError as e:
        print(f"Weather API HTTP error: {e}")
        return {"error": f"Weather API error: {str(e)}"}
    except Exception as e:
        print(f"Error in weather_forecast: {traceback.format_exc()}")
        return {"error": f"Weather forecast failed: {str(e)}"}


# ── Fertilizer Recommendation ─────────────────────────────────────────────────

class FertilizerRecommendationInput(BaseModel):
    Crop: str
    Current_N: float
    Current_P: float
    Current_K: float


@app.post("/recommend_fertilizer/")
async def recommend_fertilizer(input: FertilizerRecommendationInput):
    if FERTILIZER_RECOMMENDATION_MODEL is None or FERTILIZER_LABEL_ENCODER is None:
        return {"error": "Fertilizer recommendation model is currently unavailable."}

    try:
        # Encode crop name using the saved label encoder (case-insensitive)
        crop_name = input.Crop.strip()
        known_crops = list(FERTILIZER_LABEL_ENCODER.classes_)

        # Try exact match first
        matched_crop = None
        for kc in known_crops:
            if kc.lower() == crop_name.lower():
                matched_crop = kc
                break

        # Common aliases
        if not matched_crop:
            aliases = {"rice": "Paddy", "tur": "Arhar/Tur", "chana": "Gram"}
            matched_crop = aliases.get(crop_name.lower())

        if not matched_crop:
            return {
                "error": f"Unknown crop '{crop_name}'. Supported crops: {known_crops}"
            }

        try:
            crop_encoded = FERTILIZER_LABEL_ENCODER.transform([matched_crop])[0]
        except ValueError:
            return {
                "error": f"Unknown crop '{crop_name}'. Supported crops: {known_crops}"
            }

        df_input = pd.DataFrame([{
            'Crop_Encoded': crop_encoded,
            'Current_N': input.Current_N,
            'Current_P': input.Current_P,
            'Current_K': input.Current_K,
        }])

        prediction = FERTILIZER_RECOMMENDATION_MODEL.predict(df_input)[0]

        # The model is a multi-output regressor predicting [Required_N, Required_P, Required_K]
        recommended_N = float(prediction[0])
        recommended_P = float(prediction[1])
        recommended_K = float(prediction[2])

        return {
            "recommended_N": round(recommended_N, 2),
            "recommended_P": round(recommended_P, 2),
            "recommended_K": round(recommended_K, 2)
        }
    except Exception as e:
        print(f"Error in recommend_fertilizer: {traceback.format_exc()}")
        return {"error": f"Fertilizer recommendation failed: {str(e)}"}


# ── Market Price Forecast ─────────────────────────────────────────────────────

class MarketPriceForecastInput(BaseModel):
    crop_name: str
    weeks_to_forecast: int


@app.post("/forecast_market_prices/")
async def forecast_market_prices(input: MarketPriceForecastInput):
    try:
        model = load_model(MARKET_MODEL_PATH)
    except Exception as e:
        print(f"ERROR LOADING MARKET MODEL: {e}")
        # Try retraining
        try:
            if os.path.exists(MARKET_DATA_PATH):
                model_retrained = train_lstm_market_model()
                if model_retrained:
                    model = model_retrained
                else:
                    return {"error": "Market forecast model is currently unavailable."}
            else:
                return {"error": "Market forecast model is currently unavailable."}
        except Exception as e2:
            return {"error": "Market forecast model is currently unavailable."}

    try:
        df_historical = pd.read_csv(MARKET_DATA_PATH)
    except Exception as e:
        print(f"ERROR LOADING MARKET DATA: {e}")
        return {"error": "Market historical data is currently unavailable."}

    date_columns = ['Month', 'Month Name']
    for col in date_columns:
        if col in df_historical.columns:
            df_historical = df_historical.drop(columns=[col])

    crop_name_mapping = {
        'Wheat': 'Wheat (INR/Quintal)',
        'Rice': 'Paddy (Rice) (INR/Quintal)',
        'Maize': 'Maize (INR/Quintal)',
        'Soybean': 'Soybean (INR/Quintal)',
        'Gram': 'Gram (Chana) (INR/Quintal)',
        'Mustard': 'Mustard (INR/Quintal)'
    }

    mapped_crop_name = crop_name_mapping.get(input.crop_name, input.crop_name)

    if mapped_crop_name not in df_historical.columns:
        return {"error": f"Crop '{input.crop_name}' not found in historical data. Available crops: {list(crop_name_mapping.keys())}"}

    crop_data = df_historical[[mapped_crop_name]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(crop_data)

    n_steps = 12
    n_features = 1

    # Handle case where we have less data than n_steps
    if len(scaled_data) < n_steps:
        n_steps = len(scaled_data)

    last_known_data = scaled_data[-n_steps:]
    current_batch = last_known_data.reshape((1, n_steps, n_features))
    forecast = []

    for _ in range(input.weeks_to_forecast):
        current_pred = model.predict(current_batch, verbose=0)[0]
        forecast.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    base_date = pd.to_datetime('today')
    future_dates = [base_date + pd.Timedelta(weeks=i) for i in range(1, input.weeks_to_forecast + 1)]

    forecast_dict = {
        input.crop_name: {
            date.strftime('%Y-%m-%d'): float(price[0])
            for date, price in zip(future_dates, forecast_prices)
        }
    }

    return {"forecast": forecast_dict}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)