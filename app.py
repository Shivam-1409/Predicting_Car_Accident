import streamlit as st
import numpy as np
import pickle
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    with open("model_6.pkl", "rb") as file:
        model = joblib.load(file)
    return model

model = load_model()

st.title("🚗 Car Accident Risk Prediction Game")

st.markdown("""
### Instructions
1. Enter the road and environment details below.  
2. **Predict the accident risk (0 or 1)** using the slider.  
3. The AI will also predict — if your guess matches, you score **+1 point**!
""")

# --- Initialize user score in session state ---
if "score" not in st.session_state:
    st.session_state.score = 0
if "rounds" not in st.session_state:
    st.session_state.rounds = 0

# --- Input fields ---
st.header("Enter Road and Traffic Details")

road_type = st.selectbox("Road Type", ['urban', 'rural', 'highway'])
num_lanes = st.number_input("Number of Lanes", min_value=1, max_value=10, value=2)
curvature = st.number_input("Curvature (in degrees)", min_value=0.0, max_value=90.0, value=5.0)
speed_limit = st.number_input("Speed Limit (km/h)", min_value=20, max_value=150, value=60)
lighting = st.selectbox("Lighting Condition", ['daylight', 'dim', 'night'])
weather = st.selectbox("Weather Condition", ['rainy', 'clear', 'foggy'])
road_signs_present = st.selectbox("Are Road Signs Present?", [True, False])
public_road = st.selectbox("Is it a Public Road?", [True, False])
time_of_day = st.selectbox("Time of Day", ['morning', 'afternoon', 'evening'])
holiday = st.selectbox("Is it a Holiday?", [True, False])
school_season = st.selectbox("Is it School Season?", [True, False])
num_reported_accidents = st.number_input("Number of Reported Accidents", min_value=0, value=1)

# --- User prediction ---
st.subheader("Your Prediction")
user_pred = st.slider("Predict accident risk (0 = No Risk, 1 = High Risk)", 0, 1, 0)

# --- AI prediction ---
columns = [
    'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
    'weather', 'road_signs_present', 'public_road', 'time_of_day',
    'holiday', 'school_season', 'num_reported_accidents'
]

# Convert user inputs into DataFrame
input_data = pd.DataFrame([[road_type, num_lanes, curvature, speed_limit,
                            lighting, weather, road_signs_present, public_road,
                            time_of_day, holiday, school_season, num_reported_accidents]],
                          columns=columns)

if st.button("Submit Prediction"):
    try:
        if model.predict(input_data)>0.5:
           ai_pred = 1
        else:
           ai_pred=0
        st.session_state.rounds += 1

        st.write(f"**AI Prediction:** {ai_pred}")
        st.write(f"**Your Prediction:** {user_pred}")

        # Check if user prediction matches
        if ai_pred == user_pred:
            st.session_state.score += 1
            st.success("Great! You guessed it right!")
        else:
            st.error(" Oops! AI disagrees with your prediction.")

        # Display score
        st.info(f"🏆 **Your Score:** {st.session_state.score}/{st.session_state.rounds}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")



