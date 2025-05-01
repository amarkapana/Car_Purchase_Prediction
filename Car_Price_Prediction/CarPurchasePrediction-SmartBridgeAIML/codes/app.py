from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained KNN model and scaler
model = joblib.load(r'C:\Users\HP\AI & ML Projects\Car_Price_Prediction\car_purchase_predictor.pkl')
scaler = joblib.load(r'C:\Users\HP\AI & ML Projects\Car_Price_Prediction\scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        salary = float(request.form['salary'])
    except ValueError:
        return "Invalid input. Please enter valid numerical values."
    
    if  age < 18 or age > 100:
        return "Please enter a valid age. It should be greater than or equal to 18 and less than or equal to 100."
    elif salary < 0:
        return "Please enter a valid salary. It cannot be a negative number."

    # Prepare the data (scaling using the loaded scaler)
    input_data = np.array([[age, salary]])
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        ans = "Customer is interested to purchase the car😊"
    else:
        ans = "Customer is not interested to purchase a car😔"
    
    return render_template('result.html', ans=ans)

if __name__ == "__main__":
    app.run(debug=True)

