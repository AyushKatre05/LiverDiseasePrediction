# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
data = pd.read_csv("liver.csv")

# Data preprocessing
# Handle missing values
data.dropna(inplace=True)

# Encode categorical variables
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Split features and target variable
X = data.drop(columns=['Dataset'])
y = data['Dataset']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App
st.title("Liver Disease Prediction App")

# Sidebar for user input
st.sidebar.header("User Input")
age = st.sidebar.slider("Age", min_value=0, max_value=100, value=30)
gender = st.sidebar.radio("Gender", options=['Male', 'Female'])

total_bilirubin = st.sidebar.number_input("Total Bilirubin", value=0.5)
direct_bilirubin = st.sidebar.number_input("Direct Bilirubin", value=0.1)
alkaline_phosphotase = st.sidebar.number_input("Alkaline Phosphotase", value=150)
alamine_aminotransferase = st.sidebar.number_input("Alamine Aminotransferase", value=20)
aspartate_aminotransferase = st.sidebar.number_input("Aspartate Aminotransferase", value=25)
total_proteins = st.sidebar.number_input("Total Proteins", value=6.6)
albumin = st.sidebar.number_input("Albumin", value=3.2)
albumin_globulin_ratio = st.sidebar.number_input("Albumin and Globulin Ratio", value=0.8)

# Make prediction
gender_code = 0 if gender == 'Male' else 1
input_data = [[age, gender_code, total_bilirubin, direct_bilirubin, alkaline_phosphotase, alamine_aminotransferase, 
               aspartate_aminotransferase, total_proteins, albumin, albumin_globulin_ratio]]
prediction = model.predict(input_data)[0]

st.subheader("Prediction")
if prediction == 1:
    st.write("The model predicts that the patient may have liver disease.")
else:
    st.write("The model predicts that the patient may not have liver disease.")

    st.markdown("""
        **Disclaimer:** This is a machine learning-based model for disease prediction. While it can provide insights, it may not always be accurate. For accurate medical advice, please consult a qualified healthcare professional.
        """)
