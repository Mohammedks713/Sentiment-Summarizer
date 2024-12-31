"""
    The purpose of this file is to provide the functions to do the following:
    - confirm appropriate data file type to read in
    - read in the file and convert to appropriate data type for manipulation
    - return characteristics of the ingested data
"""

# import libraries
import pandas as pd
import numpy as np
import os


# read in data
def read_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file, header = 0)
        elif uploaded_file.name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file, header = 0)
        else:
            raise ValueError(f"{uploaded_file} is an unsupported file type. Please upload a CSV or Excel file.")

    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

# Validate if created data frame is not empty
def validate_dataframe(df):
    if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
        return False
    return True

# check if all columns have headers
def check_for_headers(df):
    if df.columns.isnull().any():
        return False
    return True

# return the names of all columns except the last one (comment or text)
def return_columns(df):
    return df.columns[:-1].tolist()




