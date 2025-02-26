import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Set page layout
st.set_page_config(layout="wide")

# Load the trained model
with open('model.pkl', 'rb') as file:
    rf = pickle.load(file)

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Clean dataset
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_with_zeros:
    df[col] = df[col].replace(0, df[col].median())

# Streamlit UI
st.title('Diabetes Prediction System')
st.sidebar.header('Patient Data')

# Training Data Stats
st.subheader('Training Data Stats')
st.write(df.describe())

# User data input function
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(user_report_data, index=[0])

# Collect user data
user_data = user_report()

# Display patient data
st.subheader('Patient Data')
st.write(user_data)

# Make predictions
user_result = rf.predict(user_data)

# Show prediction result
st.subheader('Prediction Result:')
output = 'You are Diabetic' if user_result[0] == 1 else 'You are not Diabetic'
st.title(output)

# Visualizations
st.subheader('Visualized Patient Report')

sns.set_theme(style="whitegrid")

# Plotting
fig_preg = plt.figure(figsize=(12, 6))
sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Greens', alpha=0.6)
sns.scatterplot(x=user_data['Age'], y=user_data['Pregnancies'], s=200, color='red')
plt.title('Age vs Pregnancies (0: Healthy, 1: Diabetic)')
st.pyplot(fig_preg)

fig_glucose = plt.figure(figsize=(12, 6))
sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='magma', alpha=0.6)
sns.scatterplot(x=user_data['Age'], y=user_data['Glucose'], s=200, color='red')
plt.title('Age vs Glucose Level')
st.pyplot(fig_glucose)

fig_bp = plt.figure(figsize=(12, 6))
sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='Blues', alpha=0.6)
sns.scatterplot(x=user_data['Age'], y=user_data['BloodPressure'], s=200, color='red')
plt.title('Age vs Blood Pressure')
st.pyplot(fig_bp)

# Wrap-up
st.success("App ready and improved! 🎯")

# Let me know if you want any adjustments! 🚀
