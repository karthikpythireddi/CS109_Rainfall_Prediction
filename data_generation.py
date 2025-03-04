

import numpy as np
import pandas as pd


file_name = "precipitation_wildfires_CA_OR_WA.csv"
df = pd.read_csv(file_name)


#convert the column names to lowercase an replace the spaces for easier handling. 
df.columns = df.columns.str.lower().str.replace(" ", "_")

df["precipitation_ca_nomralized"] = (df["precipitation_(ca)_(inches)"] - df["precipitation_(ca)_(inches)"].min())/ (df["precipitation_(ca)_(inches)"].max() - df["precipitation_(ca)_(inches)"].min())


print(df["precipitation_ca_nomralized"])                                                                                                          
                                                                                                                
# print(df["precipitation_(ca)_(inches)"])
# print(df["precipitation_(ca)_(inches)"].min())
# print(df["precipitation_(ca)_(inches)"].max())