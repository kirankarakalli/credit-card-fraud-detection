from flask import Flask, render_template, request
import pickle
import numpy as np
import logging

app = Flask(__name__)

# Load the trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Setup logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all input values from form
        features = [float(x) for x in request.form.values()]
        
        # Convert to numpy array (1 row, 30 columns)
        final_features = np.array(features).reshape(1, -1)
        
        # Predict using model
        prediction = model.predict(final_features)[0]

        # Output message
        result = "Fraud Detected ⚠️" if prediction == 1 else "Normal Transaction ✅"
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text="Error: Invalid Input or Shape Issue")

if __name__ == "__main__":
    app.run(debug=True)
