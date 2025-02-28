from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model and encoders
rf_model = joblib.load("healthcare_ai_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "AI Healthcare Diagnosis API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get JSON data from request
        features = np.array([data["features"]])  # Convert to NumPy array

        # Make prediction
        prediction = rf_model.predict(features)

        # Decode prediction
        predicted_disease = label_encoders["Disease"].inverse_transform(prediction)

        return jsonify({"predicted_disease": predicted_disease[0]})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask API locally
if __name__ == "__main__":
    app.run(debug=True)
