# Re-import necessary libraries
import pandas as pd
import numpy as np
import scipy.stats as stats

# Load the cleaned dataset
file_path = "cleaned_precipitation_wildfires_ca_or_wa.csv"
df = pd.read_csv(file_path)

# Create a binary wildfire classification (1 = High Wildfire Year, 0 = Low Wildfire Year)
wildfire_threshold = df["wildfires_ca"].median()  # Use median as threshold for high wildfire years
df["wildfire_high"] = (df["wildfires_ca"] >= wildfire_threshold).astype(int)

# Define prior probability of a high wildfire year
prior_prob_high = df["wildfire_high"].mean()

# Compute likelihood P(D | H) using a Gaussian distribution fit to past high wildfire years
high_wildfire_data = df[df["wildfire_high"] == 1]["precipitation_ca"]
mean_precip_high = high_wildfire_data.mean()
std_precip_high = high_wildfire_data.std()

# Compute likelihood P(D | L) using past low wildfire years
low_wildfire_data = df[df["wildfire_high"] == 0]["precipitation_ca"]
mean_precip_low = low_wildfire_data.mean()
std_precip_low = low_wildfire_data.std()

# Extract latest precipitation value (2023) to predict 2024
latest_precipitation = df["precipitation_ca"].iloc[-1]

# Compute likelihoods based on normal distribution
likelihood_high = stats.norm.pdf(latest_precipitation, mean_precip_high, std_precip_high)
likelihood_low = stats.norm.pdf(latest_precipitation, mean_precip_low, std_precip_low)

# Compute evidence P(D) using total probability
prior_prob_low = 1 - prior_prob_high
evidence = (likelihood_high * prior_prob_high) + (likelihood_low * prior_prob_low)

# Compute posterior probability P(H | D) using Bayes' Theorem
posterior_prob_high = (likelihood_high * prior_prob_high) / evidence

# Display results
prior_prob_high, likelihood_high, likelihood_low, posterior_prob_high


print(prior_prob_high)
print(likelihood_high)
print(likelihood_low)
print(posterior_prob_high)