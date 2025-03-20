# import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle

# Load the dataset
file_path = "/content/drive/MyDrive/DATASET/HOUSE_PRICE_PREDICTION.csv"
df = pd.read_csv(file_path)

# Selecting the target variable
target = "Price"

# Dropping non-numeric and identifier columns
drop_columns = ["ID", "City/District", "State", "Locality"]
df_clean = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

# Encoding categorical variables
for col in df_clean.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])

# Splitting data into features and target variable
X = df_clean.drop(columns=[target])
y = df_clean[target]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Model evaluation
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Random Forest Model Performance:\nR² Score: {r2:.4f}\nRMSE: {rmse:.2f}")

# Save the model and scaler
pickle.dump(rf_model, open('models/random_forest_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))


# joblib.dump(rf_model, "random_forest_model.pkl")
# joblib.dump(scaler, "scaler.pkl")