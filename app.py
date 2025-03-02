from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Load the trained model and encoders
rf_model = joblib.load("healthcare_ai_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Display the form page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the form data from the frontend
        symptoms = request.form["symptoms"]  

        # Convert symptoms to a numerical format (this needs proper preprocessing)
        # For now, let's assume we use dummy input
        features = np.array([[0] * len(label_encoders)])  # Dummy placeholder

        # Make prediction
        prediction = rf_model.predict(features)

        # Decode prediction
        predicted_disease = label_encoders["Disease"].inverse_transform(prediction)

        return render_template("result.html", disease=predicted_disease[0])  # Show result page

    except Exception as e:
        return render_template("result.html", error=str(e))  # Show error page

if __name__ == "__main__":
    app.run(debug=True)
