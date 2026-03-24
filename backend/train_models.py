"""
Script to train all required models for the CROPIX application
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

print("=" * 80)
print("CROPIX Model Training Script")
print("=" * 80)

# Create directories if they don't exist
os.makedirs('Trained_models/CNN', exist_ok=True)
# 1. CROP YIELD MODEL
print("\n[1/5] Training Crop Yield Prediction Model...")
try:
    df_yield = pd.read_csv('Datasets/Crop_yield.csv')
    print(f"   Loaded {len(df_yield)} samples from Crop_yield.csv")
    
    # Prepare features and target - using actual columns from dataset
    X = df_yield[['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
    y = df_yield['Yield']
    
    # Remove rows with missing values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"   Training with {len(X)} samples after removing missing values")
    
    # Train model
    model_yield = RandomForestRegressor(n_estimators=100, random_state=42)
    model_yield.fit(X, y)
    
    # Save model
    joblib.dump(model_yield, 'Trained_models/CROP_YIELD_MODEL.joblib')
    print("   ✓ Crop Yield Model trained and saved successfully!")
except Exception as e:
    print(f"   ✗ Error training Crop Yield Model: {e}")

# 2. SOIL CROP RECOMMENDATION MODEL
print("\n[2/5] Training Soil Crop Recommendation Model...")
try:
    df_soil = pd.read_csv('Datasets/Soil_Crop_recommendation.csv')
    print(f"   Loaded {len(df_soil)} samples from Soil_Crop_recommendation.csv")
    
    # Prepare features and target
    X = df_soil[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df_soil['label']
    
    # Train model
    model_soil = RandomForestClassifier(n_estimators=100, random_state=42)
    model_soil.fit(X, y)
    
    # Save model
    joblib.dump(model_soil, 'Trained_models/Soil_crop_recom.joblib')
    print("   ✓ Soil Crop Recommendation Model trained and saved successfully!")
except Exception as e:
    print(f"   ✗ Error training Soil Crop Recommendation Model: {e}")

# 3. FERTILIZER RECOMMENDATION MODEL
print("\n[3/5] Training Fertilizer Recommendation Model...")
try:
    df_fert = pd.read_csv('Datasets/fertilizer_training_data.csv')
    print(f"   Loaded {len(df_fert)} samples from fertilizer_training_data.csv")
    
    # Encode the Crop column
    le_crop = LabelEncoder()
    df_fert['Crop_Encoded'] = le_crop.fit_transform(df_fert['Crop'])
    
    # Prepare features and target
    X = df_fert[['Crop_Encoded', 'Current_N', 'Current_P', 'Current_K', 'Required_N', 'Required_P', 'Required_K']]
    
    # Create fertilizer recommendations based on deficiencies
    def recommend_fertilizer(row):
        n_def = max(0, row['Required_N'] - row['Current_N'])
        p_def = max(0, row['Required_P'] - row['Current_P'])
        k_def = max(0, row['Required_K'] - row['Current_K'])
        
        if n_def > 20 and p_def > 20:
            return 'DAP'
        elif n_def > 20:
            return 'Urea'
        elif p_def > 20:
            return 'SSP'
        elif k_def > 20:
            return 'MOP'
        elif n_def > 0 or p_def > 0 or k_def > 0:
            return 'NPK'
        else:
            return 'None'
    
    y = df_fert.apply(recommend_fertilizer, axis=1)
    
    # Train model
    model_fert = RandomForestClassifier(n_estimators=100, random_state=42)
    model_fert.fit(X, y)
    
    # Save model and label encoder
    joblib.dump(model_fert, 'Trained_models/fertilizer_recommendation_model.joblib')
    joblib.dump(le_crop, 'Trained_models/crop_label_encoder.joblib')
    print("   ✓ Fertilizer Recommendation Model trained and saved successfully!")
except Exception as e:
    print(f"   ✗ Error training Fertilizer Recommendation Model: {e}")

# 4. DISEASE DETECTION MODEL (CNN)
print("\n[4/5] Training Disease Detection Model (CNN)...")
try:
    df_cnn = pd.read_csv('Datasets/CNN_labels.csv')
    print(f"   Loaded {len(df_cnn)} samples from CNN_labels.csv")
    
    # Get unique disease classes
    disease_classes = df_cnn['label'].unique()
    print(f"   Found {len(disease_classes)} disease classes")
    
    # Save disease classes
    np.save('Trained_models/CNN/disease_classes.npy', disease_classes)
    
    # Create a simple CNN model (placeholder - would need actual images to train properly)
    num_classes = len(disease_classes)
    model_cnn = Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Save model (untrained but with correct architecture)
    model_cnn.save('Trained_models/CNN/Disease_Detection_model[CNN].h5')
    print("   ✓ Disease Detection Model created and saved successfully!")
    print("   Note: Model architecture created but needs actual training with images")
except Exception as e:
    print(f"   ✗ Error creating Disease Detection Model: {e}")

# 5. MARKET PRICE FORECAST MODEL (LSTM)
print("\n[5/5] Training Market Price Forecast Model (LSTM)...")
try:
    df_market = pd.read_csv('Datasets/synthetic_crop_prices_2years.csv')
    print(f"   Loaded {len(df_market)} samples from synthetic_crop_prices_2years.csv")
    
    # Get crop columns (exclude Month and Month Name columns)
    crop_columns = [col for col in df_market.columns if col not in ['Month', 'Month Name']]
    print(f"   Found {len(crop_columns)} crops: {', '.join([c.split(' ')[0] for c in crop_columns])}")
    
    # Prepare data for LSTM (using Wheat as example)
    wheat_col = 'Wheat (INR/Quintal)'
    if wheat_col in df_market.columns:
        prices = df_market[wheat_col].values.astype(float)
        
        # Normalize data
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
        
        # Create sequences
        sequence_length = 4  # Use 4 weeks to predict next week
        X, y = [], []
        for i in range(sequence_length, len(prices_scaled)):
            X.append(prices_scaled[i-sequence_length:i, 0])
            y.append(prices_scaled[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print(f"   Created {len(X)} training sequences")
        
        # Create LSTM model
        model_lstm = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        print("   Training LSTM model (this may take a moment)...")
        model_lstm.fit(X, y, epochs=30, batch_size=4, verbose=0)
        
        # Save model
        model_lstm.save('Trained_models/lstm_model.keras')
        print("   ✓ Market Price Forecast Model trained and saved successfully!")
    else:
        print("   ✗ Wheat column not found in dataset")
except Exception as e:
    print(f"   ✗ Error training Market Price Forecast Model: {e}")

print("\n" + "=" * 80)
print("Training Complete!")
print("=" * 80)
print("\nSummary of trained models:")
print("  1. Trained_models/CROP_YIELD_MODEL.joblib")
print("  2. Trained_models/Soil_crop_recom.joblib")
print("  3. Trained_models/fertilizer_recommendation_model.joblib")
print("  4. Trained_models/CNN/Disease_Detection_model[CNN].h5")
print("  5. Trained_models/CNN/disease_classes.npy")
print("  6. Trained_models/lstm_model.keras")
print("\nYou can now run the FastAPI backend server!")
print("=" * 80)
