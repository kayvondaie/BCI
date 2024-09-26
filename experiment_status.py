import os
import pandas as pd

# Define the root directory for raw data and processed data
raw_data_root = r'\\allen\aind\scratch\BCI\2p-raw'

# List of subjects of interest
subjects_of_interest = ["BCI85", "BCI75", "BCINM_017", "BCINM_018"]

# Initialize lists to store experiment information
subjects = []
dates = []
processed = []

# Walk through the directory structure for the subjects of interest
for subject in subjects_of_interest:
    subject_path = os.path.join(raw_data_root, subject)
    if os.path.isdir(subject_path):
        for date in os.listdir(subject_path):
            date_path = os.path.join(subject_path, date)
            if os.path.isdir(date_path):
                # Store subject and date information
                subjects.append(subject)
                dates.append(date)

                # Check if the processed data folder exists
                processed_folder = os.path.join(date_path, 'suite2p_BCI')
                if os.path.exists(processed_folder) and os.path.isdir(processed_folder):
                    processed.append('Yes')
                else:
                    processed.append('No')

# Create a DataFrame with the collected information
df = pd.DataFrame({
    'Subject': subjects,
    'Date': dates,
    'Processed': processed
})

# Define the output Excel file path
output_file = r'experiment_status.xlsx'

# Save the DataFrame to an Excel file
df.to_excel(output_file, index=False)

print(f'Experiment information has been saved to {output_file}')
