import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # For saving the scaler

# Load dataset
df = pd.read_csv('car_purchasing.csv', encoding='latin-1')

# Drop unnecessary columns
df = df.drop(['customer name', 'customer e-mail', 'country'], axis=1)

# Separate features and target
X = df.drop('car purchase amount', axis=1)
y = df['car purchase amount']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
joblib.dump(scaler, 'car_scaler.pkl')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build ANN Model
model = keras.Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=300, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

# Save the trained model
model.save('car_sales_model.h5')
print("\nModel saved as 'car_sales_model.h5'.")

# Evaluate model performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nOptimized ANN Model R² Score: {r2:.4f}")
print(f"Optimized ANN Model MSE: {mse:.2f}")
