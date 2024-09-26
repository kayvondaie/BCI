file_path = '//allen/aind/scratch/BCI/bpod_session_data/sessions/20240522-165652/20240522-165652.csv'

import pandas as pd

# Load the provided CSV file

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
        print("DataFrame head with semicolon delimiter:")
        print(df.head())

        print("\nDataFrame info with semicolon delimiter:")
        df.info()

        print("\nDataFrame description with semicolon delimiter:")
        print(df.describe())
    except Exception as e:
        print(f"Failed to load the CSV file with semicolon delimiter: {e}")
        return None
    return df

# Read the CSV file with flexible delimiters
df = read_csv_with_flexible_delimiters(file_path)
