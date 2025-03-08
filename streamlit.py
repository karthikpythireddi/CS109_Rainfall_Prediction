import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import folium
import geopandas as gpd
from streamlit_folium import folium_static

def load_data():
    file_path = "cleaned_precipitation_wildfires_ca_or_wa.csv"
    df = pd.read_csv(file_path)
    return df

def compute_wildfire_risk(state, precipitation):
    df = load_data()
    
    # Select appropriate columns based on user input
    precip_col = f"precipitation_{state.lower()}"
    wildfire_col = f"wildfires_{state.lower()}"
    
    # Compute prior probability
    wildfire_threshold = df[wildfire_col].median()
    df["wildfire_high"] = (df[wildfire_col] >= wildfire_threshold).astype(int)
    prior_prob_high = df["wildfire_high"].mean()
    
    # Compute likelihoods using normal distributions
    high_wildfire_data = df[df["wildfire_high"] == 1][precip_col]
    mean_precip_high = high_wildfire_data.mean()
    std_precip_high = high_wildfire_data.std()
    likelihood_high = stats.norm.pdf(precipitation, mean_precip_high, std_precip_high)
    
    low_wildfire_data = df[df["wildfire_high"] == 0][precip_col]
    mean_precip_low = low_wildfire_data.mean()
    std_precip_low = low_wildfire_data.std()
    likelihood_low = stats.norm.pdf(precipitation, mean_precip_low, std_precip_low)
    
    # Compute total probability (evidence)
    prior_prob_low = 1 - prior_prob_high
    evidence = (likelihood_high * prior_prob_high) + (likelihood_low * prior_prob_low)
    
    # Compute posterior probability (Bayes' theorem)
    posterior_prob_high = (likelihood_high * prior_prob_high) / evidence
    
    return posterior_prob_high

def create_us_map(risk_dict):
    # Load US states GeoJSON
    geojson_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    us_map = folium.Map(location=[40, -120], zoom_start=5)
    
    # Define risk-based color coding
    def get_color(risk):
        return "green" if risk <= 0.2 else "yellow" if risk <= 0.5 else "red"
    
    # Load GeoJSON for state boundaries
    gdf = gpd.read_file(geojson_url)
    states = {"CA": "California", "OR": "Oregon", "WA": "Washington"}
    
    for state, full_name in states.items():
        risk = risk_dict.get(state, 0)  # Ensure correct state-specific risk value
        color = get_color(risk)
        
        # Filter GeoDataFrame for the specific state
        state_geom = gdf[gdf["name"] == full_name]
        folium.GeoJson(
            state_geom,
            style_function=lambda x: {"fillColor": color, "color": "black", "weight": 1, "fillOpacity": 0.5},
        ).add_to(us_map)
    
    return us_map

def main():
    st.title("Wildfire Risk Prediction using Bayesian Inference")
    st.write("Enter the expected precipitation to estimate wildfire risk.")
    
    if "risk_dict" not in st.session_state:
        st.session_state.risk_dict = {"CA": 0, "OR": 0, "WA": 0}  # Reset stored risk values
    
    state = st.selectbox("Select State", ["CA", "OR", "WA"])
    precipitation = st.number_input(f"Enter expected precipitation for {state} (in inches):", min_value=0.0, step=0.1)
    
    if st.button("Predict Wildfire Risk"):
        # Reset risk values before each new calculation
        st.session_state.risk_dict = {s: compute_wildfire_risk(s, 50) for s in ["CA", "OR", "WA"]}  # Default reference precipitation
        
        # Update only the selected state's risk
        st.session_state.risk_dict[state] = compute_wildfire_risk(state, precipitation)
        
        st.success(f"The probability of a high wildfire year in {state} given {precipitation} inches of precipitation is: {st.session_state.risk_dict[state]:.2%}")
        
        st.write("### Wildfire Risk Map for the Western US")
        folium_static(create_us_map(st.session_state.risk_dict))

if __name__ == "__main__":
    main()
