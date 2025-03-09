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

    # ✅ Debugging: Show how precipitation affects wildfire risk
    st.write(f"🔍 Debug - {state}:")
    st.write(f"Precipitation Input: {precipitation}")
    st.write(f"Mean Precip (High Wildfire Years): {mean_precip_high}, Std Dev: {std_precip_high}")
    st.write(f"Mean Precip (Low Wildfire Years): {mean_precip_low}, Std Dev: {std_precip_low}")
    st.write(f"Likelihood High: {likelihood_high}, Likelihood Low: {likelihood_low}")
    st.write(f"Computed Wildfire Risk: {posterior_prob_high:.2%}")

    return max(0.0, min(1.0, posterior_prob_high))  # Ensure valid probability range


def create_us_map(risk_dict):
    # Load US states GeoJSON
    geojson_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    us_map = folium.Map(location=[40, -120], zoom_start=5)

    # Define risk-based color coding
    def get_color(risk):
        if risk < 0.2:
            return "green"
        elif risk < 0.5:
            return "yellow"
        else:
            return "red"

    # Load GeoJSON for state boundaries
    gdf = gpd.read_file(geojson_url)
    states = {"CA": "California", "OR": "Oregon", "WA": "Washington"}

    color_dict = {}

    for state, full_name in states.items():
        risk = risk_dict[state]
        assigned_color = get_color(risk)

        # ✅ Debug: Display state risk & assigned color
        st.write(f"**{state}: Risk = {risk:.2%}, Assigned Color = {assigned_color}**")

        color_dict[state] = {"Risk": f"{risk:.2%}", "Color": assigned_color}

        # Ensure valid GeoJSON filtering
        state_geom = gdf[gdf["name"].str.lower() == full_name.lower()]
        if not state_geom.empty:
            folium.GeoJson(
                state_geom,
                style_function=lambda x, color=assigned_color: {"fillColor": color, "color": "black", "weight": 1, "fillOpacity": 0.5},
            ).add_to(us_map)

    # ✅ Display the final risk-color mapping
    st.write("### 🎨 Debugging Info: State Colors")
    st.json(color_dict)

    return us_map



def main():
    st.title("Wildfire Risk Prediction using Bayesian Inference")
    st.write("Enter the expected precipitation to estimate wildfire risk.")
    
    state = st.selectbox("Select State", ["CA", "OR", "WA"])
    precipitation = st.number_input(f"Enter expected precipitation for {state} (in inches):", min_value=0.0, step=0.1)
    
    if st.button("Predict Wildfire Risk"):
        df = load_data()
        historical_avg = {s: df[f"precipitation_{s.lower()}"].mean() for s in ["CA", "OR", "WA"]}

        risk_dict = {s: compute_wildfire_risk(s, historical_avg[s]) for s in ["CA", "OR", "WA"]}

        # Update only the selected state with user-provided precipitation
        risk_dict[state] = compute_wildfire_risk(state, precipitation)

        # ✅ Debug: Display wildfire risk per state
        st.write("### 🔥 Wildfire Risk Debugging")
        st.json(risk_dict)

        st.success(f"The probability of a high wildfire year in {state} given {precipitation} inches of precipitation is: {risk_dict[state]:.2%}")

        st.write("### Wildfire Risk Map for the Western US")
        folium_static(create_us_map(risk_dict))




if __name__ == "__main__":
    main()
