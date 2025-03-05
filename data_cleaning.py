

import numpy as np
import pandas as pd


file_name = "precipitation_wildfires_CA_OR_WA.csv"
df = pd.read_csv(file_name)

# Convert column names to lowercase and replace spaces for easier handling
df.columns = df.columns.str.lower().str.replace(" ", "_")

#rename the columns to a simpler format without brackets
df.rename(columns={
    "precipitation_(ca)_(inches)":"precipitation_ca",
    "wildfires_(ca)":"wildfires_ca",
    "precipitation_(or)_(inches)":"precipitation_or",
    "wildfires_(or)":"wildfires_or",
    "precipitation_(wa)_(inches)":"precipitation_wa",
    "wildfires_(wa)":"wildfires_wa",
}, inplace=True) #inplace = True, changes are made directly to the dataframe without creating a copy


#we use median value as the threshold to classify the years into high and low wildfire categories
wildfire_threshold_ca = df["wildfires_ca"].median()
# print(wildfire_threshold_ca)
wildfire_threshold_or = df["wildfires_or"].median()
# print(wildfire_threshold_or)
wildfire_threshold_wa = df["wildfires_wa"].median()
# print(wildfire_threshold_wa)

#classify years as High and Low Wildfire years with the median threshold balue
df["wildfire_category_ca"] = np.where(df["wildfires_ca"] >= wildfire_threshold_ca, "High", "Low")
# print(df["wildfire_category_ca"])
df["wildfire_category_or"] = np.where(df["wildfires_or"] >= wildfire_threshold_or, "High", "Low")
# print(df["wildfire_category_or"])
df["wildfire_category_wa"] = np.where(df["wildfires_wa"] >= wildfire_threshold_wa, "High", "Low")
# print(df["wildfire_category_wa"])

cleaned_file = "cleaned_precipitation_wildfires_ca_or_wa.csv"
df.to_csv(cleaned_file, index=False)