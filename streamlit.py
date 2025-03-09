import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import folium
import geopandas as gpd
import statsmodels.api as sm
from streamlit_folium import st_folium
import requests

def load_data():
    file_path = "cleaned_precipitation_wildfires_ca_or_wa.csv"
    df = pd.read_csv(file_path)
    return df

def compute_wildfire_risk(state, precipitation):
    df = load_data()
    
    precip_col = f"precipitation_{state.lower()}"
    wildfire_col = f"wildfires_{state.lower()}"
    
    wildfire_threshold = df[wildfire_col].median()
    df["wildfire_high"] = (df[wildfire_col] >= wildfire_threshold).astype(int)
    prior_prob_high = df["wildfire_high"].mean()
    
    high_wildfire_data = df[df["wildfire_high"] == 1][precip_col]
    low_wildfire_data = df[df["wildfire_high"] == 0][precip_col]

    mean_precip_high = high_wildfire_data.mean()
    std_precip_high = high_wildfire_data.std()
    mean_precip_low = low_wildfire_data.mean()
    std_precip_low = low_wildfire_data.std()

    std_precip_high = std_precip_high if std_precip_high > 0 else 1
    std_precip_low = std_precip_low if std_precip_low > 0 else 1

    likelihood_high = 1 - stats.norm.cdf(precipitation, mean_precip_high, std_precip_high)
    likelihood_low = stats.norm.cdf(precipitation, mean_precip_low, std_precip_low)

    likelihood_high = max(likelihood_high, 1e-6)
    likelihood_low = max(likelihood_low, 1e-6)

    prior_prob_low = 1 - prior_prob_high
    evidence = (likelihood_high * prior_prob_high) + (likelihood_low * prior_prob_low)

    posterior_prob_high = (likelihood_high * prior_prob_high) / evidence

    return max(0.0, min(1.0, posterior_prob_high))

def predict_wildfire_count(state, precipitation):
    df = load_data()
    
    precip_col = f"precipitation_{state.lower()}"
    wildfire_col = f"wildfires_{state.lower()}"
    
    X = df[[precip_col]]
    X = sm.add_constant(X)
    y = df[wildfire_col]
    
    poisson_model = sm.Poisson(y, X)
    poisson_result = poisson_model.fit()
    
    future_data = pd.DataFrame({
        "const": 1,
        precip_col: [precipitation]
    })
    predicted_wildfires = poisson_result.predict(future_data)[0]
    
    return int(round(predicted_wildfires))

def main():
    st.set_page_config(page_title="Wildfire Risk Prediction", layout="wide")
    
    st.title("🔥 Wildfire Risk & Prediction using Bayesian Inference & Poisson Regression")
    st.write("Enter the expected precipitation to estimate wildfire risk and number of wildfires.")
    
    state = st.selectbox("Select State", ["CA", "OR", "WA"])
    precipitation = st.number_input(f"🌧️ Enter expected precipitation for {state} (in inches):", min_value=0.0, step=0.1)
    
    if st.button("🚀 Predict Wildfire Risk & Count"):
        df = load_data()
        historical_avg = {s: df[f"precipitation_{s.lower()}"].mean() for s in ["CA", "OR", "WA"]}
        
        risk_dict = {s: compute_wildfire_risk(s, historical_avg[s]) for s in ["CA", "OR", "WA"]}
        wildfire_counts = {s: predict_wildfire_count(s, historical_avg[s]) for s in ["CA", "OR", "WA"]}
        
        risk_dict[state] = compute_wildfire_risk(state, precipitation)
        wildfire_counts[state] = predict_wildfire_count(state, precipitation)
        
        if risk_dict[state] < 0.2:
            bg_image = "https://i.giphy.com/media/l41lUjUgLlG4ri5X6/giphy.gif"  # Rain animation
            msg = "🌧️ Low wildfire risk! Enjoy the rain!"
        else:
            bg_image = "https://i.gifer.com/YZ5R.gif"  # Fire animation
            msg = "🔥 High wildfire risk! Stay prepared!"
        
        st.markdown(
            f"""
            <style>
                .bg {{
                    background-image: url("{bg_image}");
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                    height: 300px;
                    text-align: center;
                    padding: 20px;
                    color: white;
                    font-size: 24px;
                    border-radius: 10px;
                }}
            </style>
            <div class="bg">
                <h2>Wildfire Risk Level: {risk_dict[state]:.2%}</h2>
                <p>{msg}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(f"**The probability of a high wildfire year in {state} given {precipitation} inches of precipitation is: {risk_dict[state]:.2%}**")
        st.success(f"🌲 Predicted number of wildfires in {state}: **{wildfire_counts[state]}**")

if __name__ == "__main__":
    main()
