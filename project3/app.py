# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle

# # Create flask app
# flask_app = Flask(__name__, static_folder='static')
# model = pickle.load(open("new_scaler.pkl", "rb"))
# model = pickle.load(open("new_random_forest_model.pkl", "rb"))

# @flask_app.route("/")
# def Home():
#     return render_template("index.html")

# @flask_app.route("/predict", methods = ["POST"])
# def predict():
#     float_features = [float(x) for x in request.form.values()]
#     features = [np.array(float_features)]
#     prediction = model.predict(features)
#     return render_template("index.html", prediction_text = "The Predicted Crop is {}".format(prediction))

# if __name__ == "__main__":
#     flask_app.run(debug=True)

# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle

# # Create flask app
# flask_app = Flask(__name__, static_folder='static')

# # Load the model and scaler
# scaler = pickle.load(open("new_scaler.pkl", "rb"))
# model = pickle.load(open("new_random_forest_model.pkl", "rb"))

# # Define a mapping for categorical values (e.g., encoding cities)
# city_mapping = {
#     "Mumbai": 1,
#     "Delhi": 2,
#     "Bangalore": 3,
#     "Chennai": 4,
#     "Kolkata": 5
# }

# @flask_app.route("/")
# def home():
#     return render_template("index.html")

# @flask_app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Extract form values
#         form_values = request.form.to_dict()

#         # Prepare features for the model
#         processed_features = []
#         for key, value in form_values.items():
#             if key.lower() == "city":  # Handle the "City" field
#                 if value in city_mapping:
#                     processed_features.append(city_mapping[value])  # Encode the city
#                 else:
#                     return render_template("index.html", prediction_text="Error: Unknown city '{}'".format(value))
#             else:
#                 try:
#                     processed_features.append(float(value))  # Convert numeric fields to float
#                 except ValueError:
#                     return render_template("index.html", prediction_text="Error: Invalid input for '{}'".format(key))

#         # Scale the features if necessary
#         scaled_features = scaler.transform([processed_features])

#         # Make prediction
#         prediction = model.predict(scaled_features)

#         return render_template("index.html", prediction_text="The Predicted Price is {}".format(prediction[0]))
#     except Exception as e:
#         return render_template("index.html", prediction_text="Error: {}".format(str(e)))

# if __name__ == "__main__":
#     flask_app.run(debug=True)

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__, static_folder='static')

# Load the model and scaler
scaler = pickle.load(open("new_scaler.pkl", "rb"))
model = pickle.load(open("new_random_forest_model.pkl", "rb"))

# Define a mapping for categorical values (e.g., encoding cities)
city_mapping = {
    "Mumbai": 1,
    "Delhi": 2,
    "Bangalore": 3,
    "Chennai": 4,
    "Kolkata": 5
}

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
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
                    return render_template("index.html", prediction_text="Error: Unknown city '{}'".format(value))
            else:
                try:
                    processed_features.append(float(value))  # Convert numeric fields to float
                except ValueError:
                    return render_template("index.html", prediction_text="Error: Invalid input for '{}'".format(key))

        # Scale the features if necessary
        scaled_features = scaler.transform([processed_features])

        # Make prediction
        prediction = model.predict(scaled_features)

        # Include the city name in the output
        return render_template(
            "index.html",
            prediction_text="The Predicted Price for a house in {} is {}".format(city_name, prediction[0])
        )
    except Exception as e:
        return render_template("index.html", prediction_text="Error: {}".format(str(e)))

if __name__ == "__main__":
    flask_app.run(debug=True)