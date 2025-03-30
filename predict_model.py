import numpy as np
import tensorflow as tf
import joblib  # Load the scaler

# Load trained model
model2 = tf.keras.models.load_model('car_sales_model.h5')
print("Model loaded successfully.")

# Load saved scaler
scaler2 = joblib.load('car_scaler.pkl')

# Function to predict car purchase amount
def predict_car_purchase(gender, age, annual_salary, credit_card_debt, net_worth):
    input_data = np.array([[gender, age, annual_salary, credit_card_debt, net_worth]])
    input_scaled = scaler2.transform(input_data)  # Scale input data
    prediction = model2.predict(input_scaled)
    return prediction[0][0]

# Example prediction
predicted_purchase = predict_car_purchase(1, 55, 70000, 10000, 500000)
print(f"\nPredicted Car Purchase Amount: {predicted_purchase:.2f}")
