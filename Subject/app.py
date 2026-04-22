import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Load trained model and scaler
@st.cache_resource
def load_models():
    with open("best_heart_disease_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_models()

# Load dataset for EDA
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_cleaned.csv")
    return df

df = load_data()

st.title("❤️ Heart Disease Prediction & Analysis")

tab1, tab2 = st.tabs(["🩺 Predictive Model", "📊 Exploratory Data Analysis (EDA)"])

with tab1:
    st.header("Assess Patient Risk")
    st.markdown("Enter patient details below to predict the likelihood of heart disease in real-time.")
    
    # Create forms or layout
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            cp = st.selectbox("Chest Pain Type (CP)", options=[0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic")
            trestbps = st.number_input("Resting BP (mm Hg)", min_value=50.0, max_value=250.0, value=120.0)
            chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=50.0, max_value=600.0, value=200.0)
            
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 (FBS)", options=[0, 1], format_func=lambda x: "True (>120)" if x == 1 else "False (<120)")
            restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], help="0: Normal, 1: ST-T abnormality, 2: LV hypertrophy")
            thalach = st.number_input("Max Heart Rate Achieved", min_value=50.0, max_value=250.0, value=150.0)
            exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            
        with col3:
            slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
            ca = st.selectbox("Major Vessels Colored by Flourosopy (CA)", options=[0, 1, 2, 3, 4], help="Values 0 to 4")
            thal = st.selectbox("Thalassemia (THAL)", options=[0, 1, 2, 3], help="1: Fixed Defect, 2: Normal, 3: Reversable Defect")
            
            st.write("")
            st.write("")
            submit_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

    if submit_btn:
        # Create a dataframe for the input row
        input_data = pd.DataFrame({
            "age": [age], "sex": [sex], "cp": [cp], "trestbps": [trestbps],
            "chol": [chol], "fbs": [fbs], "restecg": [restecg], "thalach": [thalach],
            "exang": [exang], "oldpeak": [oldpeak], "slope": [slope], "ca": [ca], "thal": [thal]
        })
        
        # Scale the data using our loaded scaler
        scaled_input = scaler.transform(input_data)
        
        # Ensure our input shape aligns with how it was trained
        prediction = model.predict(scaled_input)[0]
        
        # Predict probability if supported
        try:
            probabilities = model.predict_proba(scaled_input)[0]
            prob_disease = probabilities[0] * 100
        except AttributeError:
            prob_disease = None
        
        # Determine the target mapping. In typical UCI, target=1 handles presence/absence or risk=1 depends on author choice
        if prediction == 1:
            st.error(f"### ⚠️ High Risk of Heart Disease")
            st.markdown("The model detected clinical indicators strongly correlating with heart disease.")
            if prob_disease is not None:
                st.write(f"Estimated Probability: **{prob_disease:.2f}%**")
        else:
            st.success(f"### ✅ Low Risk of Heart Disease")
            st.markdown("The patient's clinical profile suggests a low probability of heart disease.")
            if prob_disease is not None:
                st.write(f"Estimated Probability: **{prob_disease:.2f}%**")

with tab2:
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Understand the properties of the dataset and how individual features relate to heart disease.")
    
    st.subheader("Data Overview")
    st.dataframe(df.head(), use_container_width=True)
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        fig1 = px.histogram(df, x="age", color="target", barmode="overlay",
                            title="Age Distribution by Target",
                            labels={"target": "Heart Disease (Target)", "age": "Age in Years"},
                            color_discrete_map={0: '#3498db', 1: '#e74c3c'})
        st.plotly_chart(fig1, use_container_width=True)
        
        fig_cp = px.histogram(df, x="cp", color="target", barmode="group",
                            title="Chest Pain Type by Target",
                            labels={"target": "Heart Disease (Target)", "cp": "Chest Pain Type"},
                            color_discrete_map={0: '#3498db', 1: '#e74c3c'})
        st.plotly_chart(fig_cp, use_container_width=True)

    with col_v2:
        fig2 = px.box(df, x="target", y="thalach", color="target",
                      title="Max Heart Rate (Thalach) vs Target",
                      labels={"target": "Heart Disease (Target)", "thalach": "Max Heart Rate Achieved"},
                      color_discrete_map={0: '#3498db', 1: '#e74c3c'})
        st.plotly_chart(fig2, use_container_width=True)
        
        fig4 = px.scatter(df, x="age", y="thalach", color="target",
                          title="Age vs Max Heart Rate (Colored by Target)",
                          labels={"target": "Heart Disease", "age": "Age", "thalach": "Max Heart Rate"},
                          color_discrete_map={0: '#3498db', 1: '#e74c3c'}, opacity=0.7)
        st.plotly_chart(fig4, use_container_width=True)
        
    st.subheader("Feature Correlation")
    # Mapping the target into numeric isn't strictly necessary since it already is, but we compute correlation
    corr = df.corr()
    fig3 = px.imshow(corr, text_auto=".2f", aspect="auto",
                     title="Pairwise Correlation Matrix of Clinical Features", 
                     color_continuous_scale="RdBu_r")
    st.plotly_chart(fig3, use_container_width=True)
