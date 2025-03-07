import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy
st.info(f"SciPy version: {scipy.__version__}")

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

# Streamlit UI
def main():
    st.title("Wildfire Risk Prediction using Bayesian Inference")
    st.write("Enter the expected precipitation to estimate wildfire risk.")
    
    state = st.selectbox("Select State", ["CA", "OR", "WA"])
    precipitation = st.number_input(f"Enter expected precipitation for {state} (in inches):", min_value=0.0, step=0.1)
    
    if st.button("Predict Wildfire Risk"):
        risk = compute_wildfire_risk(state, precipitation)
        st.success(f"The probability of a high wildfire year in {state} given {precipitation} inches of precipitation is: {risk:.2%}")

if __name__ == "__main__":
    main()
