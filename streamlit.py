import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import folium
import geopandas as gpd
import statsmodels.api as sm
from streamlit_folium import folium_static

def load_data():
    file_path = "cleaned_precipitation_wildfires_ca_or_wa.csv"
    df = pd.read_csv(file_path)
    return df

def compute_wildfire_risk(state, precipitation):
    df = load_data()
    
    # Select appropriate columns
    precip_col = f"precipitation_{state.lower()}"
    wildfire_col = f"wildfires_{state.lower()}"
    
    # Compute prior probability
    wildfire_threshold = df[wildfire_col].median()
    df["wildfire_high"] = (df[wildfire_col] >= wildfire_threshold).astype(int)
    prior_prob_high = df["wildfire_high"].mean()
    
    # Compute likelihoods using normal distributions
    high_wildfire_data = df[df["wildfire_high"] == 1][precip_col]
    low_wildfire_data = df[df["wildfire_high"] == 0][precip_col]

    mean_precip_high = high_wildfire_data.mean()
    std_precip_high = high_wildfire_data.std()
    mean_precip_low = low_wildfire_data.mean()
    std_precip_low = low_wildfire_data.std()

    # Handle cases where std deviation is zero
    std_precip_high = std_precip_high if std_precip_high > 0 else 1
    std_precip_low = std_precip_low if std_precip_low > 0 else 1

    # Adjust likelihoods to correctly reflect that high precipitation reduces wildfire risk
    likelihood_high = 1 - stats.norm.cdf(precipitation, mean_precip_high, std_precip_high)
    likelihood_low = stats.norm.cdf(precipitation, mean_precip_low, std_precip_low)

    # Normalize likelihoods to prevent extreme probabilities
    likelihood_high = max(likelihood_high, 1e-6)
    likelihood_low = max(likelihood_low, 1e-6)

    # Compute total probability (evidence)
    prior_prob_low = 1 - prior_prob_high
    evidence = (likelihood_high * prior_prob_high) + (likelihood_low * prior_prob_low)

    # Compute posterior probability (Bayes' theorem)
    posterior_prob_high = (likelihood_high * prior_prob_high) / evidence

    return max(0.0, min(1.0, posterior_prob_high))  # Ensure valid probability range

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

def create_us_map(risk_dict, wildfire_counts):
    geojson_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    us_map = folium.Map(location=[40, -120], zoom_start=5)

    def get_color(risk):
        return "green" if risk < 0.2 else "yellow" if risk < 0.5 else "red"

    gdf = gpd.read_file(geojson_url)
    states = {"CA": "California", "OR": "Oregon", "WA": "Washington"}

    for state, full_name in states.items():
        risk = risk_dict[state]
        color = get_color(risk)
        
        state_geom = gdf[gdf["name"].str.lower() == full_name.lower()]
        if not state_geom.empty:
            folium.GeoJson(
                state_geom,
                style_function=lambda x, color=color: {"fillColor": color, "color": "black", "weight": 1, "fillOpacity": 0.5},
            ).add_to(us_map)

            folium.Marker(
                location=state_geom.geometry.centroid.iloc[0].coords[:][0][::-1],
                popup=f"{state}: {wildfire_counts[state]} predicted wildfires",
                icon=folium.Icon(color="blue")
            ).add_to(us_map)
    
    return us_map

def main():
    st.title("Wildfire Risk & Prediction using Bayesian Inference & Poisson Regression")
    st.write("Enter the expected precipitation to estimate wildfire risk and number of wildfires.")
    
    state = st.selectbox("Select State", ["CA", "OR", "WA"])
    precipitation = st.number_input(f"Enter expected precipitation for {state} (in inches):", min_value=0.0, step=0.1)
    
    if st.button("Predict Wildfire Risk & Count"):
        df = load_data()
        historical_avg = {s: df[f"precipitation_{s.lower()}"].mean() for s in ["CA", "OR", "WA"]}
        
        risk_dict = {s: compute_wildfire_risk(s, historical_avg[s]) for s in ["CA", "OR", "WA"]}
        wildfire_counts = {s: predict_wildfire_count(s, historical_avg[s]) for s in ["CA", "OR", "WA"]}
        
        # Update only the selected state with user-provided precipitation
        risk_dict[state] = compute_wildfire_risk(state, precipitation)
        wildfire_counts[state] = predict_wildfire_count(state, precipitation)
        
        st.success(f"The probability of a high wildfire year in {state} given {precipitation} inches of precipitation is: {risk_dict[state]:.2%}")
        st.success(f"Predicted number of wildfires in {state}: {wildfire_counts[state]}")
        
        st.write("### Wildfire Risk & Count Map for the Western US")
        folium_static(create_us_map(risk_dict, wildfire_counts))

if __name__ == "__main__":
    main()