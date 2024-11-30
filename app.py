from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model (assuming you have a model for calorie prediction)
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    total_steps = float(request.form['totalSteps'])
    very_active = float(request.form['veryActive'])
    fairly_active = float(request.form['fairlyActive'])
    lightly_active = float(request.form['lightlyActive'])
    sedentary = float(request.form['sedentary'])
    
    # Prepare features for prediction (modify this as per your model)
    features = np.array([[total_steps, very_active, fairly_active, lightly_active, sedentary]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Display the prediction (calories burned)
    output = f"Estimated Calories Burned: {prediction[0]:.2f} calories"

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
