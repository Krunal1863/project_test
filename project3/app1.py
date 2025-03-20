import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app1 = Flask(__name__, static_folder='static')

# Load the model and scaler
# Load the scaler and model
scaler = pickle.load(open("land_scaler.pkl", "rb"))  # Load the scaler first
model = pickle.load(open("land_random_forest_model.pkl", "rb"))  # Load the model second

# Define a mapping for categorical values (e.g., encoding cities)
city_mapping = {
    "Rajkot": 1,
    "Vadodara": 2,
    "Surat": 3,
    "Gandhinagar": 4,
    "Ahmedabad": 5
}

@flask_app1.route("/")
def home():
    return render_template("index1.html")

@flask_app1.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form values
        form_values = request.form.to_dict()

        # Prepare features for the model
        processed_features = []
        city_name = None  # Variable to store the city name
        for key, value in form_values.items():
            if key.lower() == "city":  # Handle the "City" field
                city_name = value  # Store the city name
                if value in city_mapping:
                    processed_features.append(city_mapping[value])  # Encode the city
                else:
                    return render_template("index1.html", prediction_text="Error: Unknown city '{}'".format(value))
            else:
                try:
                    processed_features.append(float(value))  # Convert numeric fields to float
                except ValueError:
                    return render_template("index1.html", prediction_text="Error: Invalid input for '{}'".format(key))

        # Scale the features if necessary
        scaled_features = scaler.transform([processed_features])

        # Make prediction
        prediction = model.predict(scaled_features)

        # Include the city name in the output
        return render_template(
            "index1.html",
            prediction_text="The Predicted Price of Land in {} is {}".format(city_name, prediction[0])
        )
    except Exception as e:
        return render_template("index1.html", prediction_text="Error: {}".format(str(e)))

if __name__ == "__main__":
    flask_app1.run(debug=True,port=5001)