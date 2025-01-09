import streamlit as st
import pandas as pd
import numpy as np

# Title and description
st.title("Prediksi Harga Tawar Properti di Malaysia")
st.write("Aplikasi ini membantu memprediksi harga properti berdasarkan parameter yang Anda masukkan.")

# Load and preprocess data
def load_data():
    dataset = pd.read_csv('blob/master/malaysia_property_for_sale.csv')  # Adjust path if necessary
    # Data cleaning
    dataset['list_price'] = dataset['list_price'].str.replace('RM', '').str.replace(',', '').astype(float)
    dataset['unit_price'] = dataset['unit_price'].str.replace('RM', '').str.replace(',', '').str.replace('/ m2', '').str.replace('(', '').str.replace(')', '').astype(float)
    dataset['area'] = dataset['area'].str.replace(' m2', '').str.replace(',', '').astype(float)
 
    # Encoding
    dataset['location_encoded'] = dataset['location'].astype('category').cat.codes
    dataset['type_encoded'] = dataset['type'].astype('category').cat.codes
 
    return dataset

data = load_data()

# Display dataset preview
if st.checkbox("Tampilkan dataset", False):
    st.write(data.head())

# Define inputs
st.sidebar.header("Input Parameter")
number_bedroom = st.sidebar.number_input("Jumlah Kamar Tidur", min_value=0, value=3)
number_bathroom = st.sidebar.number_input("Jumlah Kamar Mandi", min_value=0, value=2)
location = st.sidebar.selectbox("Lokasi", data['location'].unique())
area_m2 = st.sidebar.number_input("Area (m2)", min_value=0, value=100)
type_property = st.sidebar.selectbox("Tipe Properti", data['type'].unique())
unit_price_rm_m2 = st.sidebar.number_input("Harga per m2 (RM)", min_value=0, value=5000)

# Map inputs to encoded values
location_encoded = data.loc[data['location'] == location, 'location_encoded'].iloc[0]
type_encoded = data.loc[data['type'] == type_property, 'type_encoded'].iloc[0]

# Prepare model data
X = data[['number_bedroom', 'number_bathroom', 'location_encoded', 'area', 'type_encoded', 'unit_price']].values
y = data['list_price'].values

# Handle missing or infinite values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

# Manually scale features
mean = X.mean(axis=0)
std = X.std(axis=0)
X_scaled = (X - mean) / std

# Split data into training and test sets (80/20 split)
split_index = int(0.8 * len(X))
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train Ridge Regression model manually
# We need to compute the Ridge regression coefficients using the normal equation
lambda_ = 1.0  # Regularization parameter (alpha in Ridge regression)
X_train_with_intercept = np.c_[np.ones(X_train.shape[0]), X_train]  # Add intercept term (bias)
theta = np.linalg.inv(X_train_with_intercept.T @ X_train_with_intercept + lambda_ * np.eye(X_train_with_intercept.shape[1])) @ X_train_with_intercept.T @ y_train

# Predict on both training and test data
X_test_with_intercept = np.c_[np.ones(X_test.shape[0]), X_test]
y_train_pred = X_train_with_intercept @ theta
y_test_pred = X_test_with_intercept @ theta

# Calculate evaluation metrics
train_r2 = 1 - ((y_train - y_train_pred) ** 2).sum() / ((y_train - y_train.mean()) ** 2).sum()
test_r2 = 1 - ((y_test - y_test_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
train_mse = ((y_train - y_train_pred) ** 2).mean()
test_mse = ((y_test - y_test_pred) ** 2).mean()

# Display evaluation results
# st.subheader("Akurasi Model")
# st.write(f"**R-squared (R²) pada Data Training:** {train_r2:.2f}")
# st.write(f"**R-squared (R²) pada Data Testing:** {test_r2:.2f}")
# st.write(f"**Mean Squared Error (MSE) pada Data Training:** {train_mse:,.2f}")
# st.write(f"**Mean Squared Error (MSE) pada Data Testing:** {test_mse:,.2f}")

# Prediction
user_input = np.array([
    number_bedroom, number_bathroom, location_encoded, area_m2, type_encoded, unit_price_rm_m2
]).reshape(1, -1)
user_input_scaled = (user_input - mean) / std  # Scale user input
user_input_with_intercept = np.c_[np.ones(user_input_scaled.shape[0]), user_input_scaled]

estimated_price = user_input_with_intercept @ theta

# Display result
st.subheader("Estimasi Harga")
st.write(f"Harga properti yang diperkirakan adalah RM {estimated_price[0]:,.2f}")
