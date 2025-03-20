from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('house_price.html')  # Render the house_price.html file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the request
        city = request.form['city']
        location = request.form['location']
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])

        # Combine the features into a single array (adjust based on your model's input)
        features = np.array([area, bedrooms, bathrooms]).reshape(1, -1)

        # Scale the features
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Render the result on the same page
        return render_template('house_price.html', prediction_text=f'Predicted Price: ${prediction[0]:,.2f}')
    except Exception as e:
        return render_template('house_price.html', error_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)