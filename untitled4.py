# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:04:07 2024

@author: kayvon.daie
"""

import pandas as pd

file_path = '//allen/aind/scratch/BCI/bpod_session_data/sessions/20240522-165652/20240522-165652.csv'

# Function to read the file and skip metadata lines dynamically
def read_csv_with_flexible_delimiters(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the first line with actual data (not starting with '__')
    data_start_line = 0
    for i, line in enumerate(lines):
        if not line.startswith('__'):
            data_start_line = i
            break

    # Attempt to read the file, skipping metadata lines
    try:
        # Read the file from the detected data start line
        df = pd.read_csv(file_path, skiprows=data_start_line, delimiter=';')
        return df
    except Exception as e:
        print(f"Failed to load the CSV file: {e}")
        return None

# Read the CSV file with flexible delimiters
df = read_csv_with_flexible_delimiters(file_path)

# Function to locate values associated with the variable "reward_L"
def find_values_for_variable(df, variable):
    if df is not None:
        # Print the first few rows to inspect the DataFrame
        print("DataFrame head:")
        print(df.head())
        
        # Print all column names
        print("Column names:")
        print(df.columns)
        
        # Filter rows where the 'MSG' column contains the variable (case-insensitive)
        variable_rows = df[df['MSG'].str.contains(variable, case=False, na=False)]
        return variable_rows
    else:
        return None

# Locate the values associated with the variable "reward_L"
reward_L_values = find_values_for_variable(df, "reward_L")

# Display the results
if reward_L_values is not None and not reward_L_values.empty:
    print("Values associated with 'reward_L':")
    print(reward_L_values)
else:
    print("No values found for 'reward_L' or the file could not be loaded.")
