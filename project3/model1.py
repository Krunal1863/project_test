import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset
file_path = "D:\Github_Project\project_test\project3\Land_prediction_model\LAND_PRICE_PREDICTION.csv"
df = pd.read_csv(file_path)


df1= df.drop(['ID','State','Locality', 'Land Type','Proximity to Amenities', 'Property Use','Market Trend (%)', 'Latitude', 'Longitude'],axis=1)


X= df1.drop('Price (INR)',axis=1)


y=df['Price (INR)']

le = LabelEncoder()
X['City'] = le.fit_transform(X['City/District'])

# Drop City/ District column
X= df1.drop('City/District',axis=1)



# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Model evaluation
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Random Forest Model Performance:\nR² Score: {r2:.4f}\nRMSE: {rmse:.2f}")



import pickle

# Save the trained model to a new pickle file
pickle.dump(rf_model, open('land_random_forest_model.pkl', 'wb'))

# Save the scaler to a new pickle file
pickle.dump(scaler, open('land_scaler.pkl', 'wb'))