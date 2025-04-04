import pandas as pd
import subprocess
import os
from pathlib import Path

import session_counting
import data_dict_create_module_test as ddct

list_of_dirs = session_counting.counter2(
    mice=["BCI88", "BCI93", "BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI107", "BCI109"],
    cutoff_str="010525"
)


# Filter sessions that have 40+ neuron TIFFs and no suite2p folder yet
sessions_to_run = list_of_dirs[(list_of_dirs["Has 40+ neuron TIFs"]) & (~list_of_dirs["Has suite2p subdir"])]

print(f"Found {len(sessions_to_run)} sessions to process.")

script_to_run = "kd_suite2p_use_old_rois3.py"  # This script should create and save ind.npy in the folder

#%%
import pandas as pd
import requests
from io import BytesIO

# Public download link
url = "https://docs.google.com/spreadsheets/d/1dhKRnYQ7f5DlPyZ9UhKf9U2CueE7VWKRrQdS692yyZg/export?format=xlsx"

# Download the Excel file
response = requests.get(url)
xls = pd.ExcelFile(BytesIO(response.content))

# List available sheets (one per mouse)
print("Sheets found:", xls.sheet_names)

# Read one sheet (e.g., BCI104)
df = xls.parse("BCI104")

# Filter and extract FOVs
fov_per_session = df[["Date", "FOV"]].dropna()
print(fov_per_session)

#%%
for _, row in sessions_to_run[0:1].iterrows():
    folder = base_dir / row["Mouse"] / row["Session"] / "pophys"
    ind_path = session_folder / "ind.npy"
    
    if ind_path.exists():
        print(f"✓ Skipping {session_folder} (ind.npy already exists)")
        continue

    print(f"▶ Running generate_ind.py in {session_folder}")
    
    try:
        subprocess.run(["python", script_to_run], cwd=session_folder, check=True)
        runcell(1,r'C:/Users/kayvon.daie/Documents/GitHub/BCI/' + script_to_run)
    except subprocess.CalledProcessError as e:
        print(f"✗ Error processing {session_folder}: {e}")
