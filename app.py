from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model and encoders
rf_model = joblib.load("healthcare_ai_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
disease_info = joblib.load("disease_info.pkl")  # A dictionary with explanations for each illness

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Renders the homepage

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get JSON data from request
        symptoms = data.get("features", [])  # Extract symptoms
        
        if not symptoms:
            return jsonify({"error": "Please provide at least one symptom."})
        
        # Convert symptoms into numerical features (handling unknowns)
        features = np.array([symptoms])
        prediction = rf_model.predict(features)
        predicted_disease = label_encoders["Disease"].inverse_transform(prediction)
        disease_name = predicted_disease[0]
        
        # Fetch disease explanation
        explanation = disease_info.get(disease_name, "No description available.")
        
        return jsonify({
            "predicted_disease": disease_name,
            "explanation": explanation,
            "recommendation": "Consult a doctor if symptoms persist."
        })
    except Exception as e:
        return jsonify({"error": "Invalid input. Please enter valid symptoms."})

if __name__ == "__main__":
    app.run(debug=True)
