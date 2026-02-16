# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:23:40 2025

@author: kayvon.daie
"""

import os
import numpy as np
import session_counting
import data_dict_create_module_test as ddc
import bci_time_series as bts

mice = ["BCINM_017", "BCINM_027", "BCINM_021", "BCINM_024", "BCINM_031", "BCINM_034"]
sessions = session_counting.counter2(mice, '010112', has_pophys=False)

AXON_REW, AXON_TS = [], []
SESSION = []

save_dir = r'I:\My Drive\Learning rules\BCI_data\Neuromodulator imaging'

for i in range(len(sessions)):
    print(i)
    try:
        mouse = sessions['Mouse'][i]
        session = sessions['Session'][i]
        suffix = 'BCI'
        save_filename = f"data_main_{mouse}_{session}_{suffix}.npy"
        save_path = os.path.join(save_dir, save_filename)

        # Skip if file already exists
        if os.path.exists(save_path):
            print(f"Skipping {mouse} {session}: already saved.")
            continue

        # Try loading from pophys subfolder first
        try:
            folder = rf'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/pophys/'
            main_npy_filename = f"data_main_{mouse}_{session}_{suffix}.npy"
            main_npy_path = os.path.join(folder, main_npy_filename)
            data = np.load(main_npy_path, allow_pickle=True)
        except:
            # Fall back to session root folder
            folder = rf'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/'
            main_npy_filename = f"data_main_{mouse}_{session}_{suffix}.npy"
            main_npy_path = os.path.join(folder, main_npy_filename)
            data = np.load(main_npy_path, allow_pickle=True)

        # Save the data
        np.save(save_path, data)

    except Exception as e:
        print(f"Failed to process {mouse} {session}: {e}")
        continue
#%%
import pandas as pd
import os

# ---- List of mice you care about ----
mice = ["BCINM_017", "BCINM_027", "BCINM_021", "BCINM_024", "BCINM_031", "BCINM_034"]

# ---- Download genotype table from public Google Sheet ----
sheet_id = "15rnIwVV0hdLzp5gz0wOp2912v0HESNtZra2n8Cg2bys"
gid = "792846944"
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

# ---- Load sheet into DataFrame ----
df_genos = pd.read_csv(csv_url)

# ---- Mapping from genotype keywords to brain area and neuromodulator ----
mod_meta_map = {
    "Dbh-Cre":  {"Brain Area": "LC (brainstem)",  "Neuromodulator": "Norepinephrine"},
    "Chat-Cre": {"Brain Area": "BF (forebrain)",  "Neuromodulator": "Acetylcholine"},
    "Sert-Cre": {"Brain Area": "DRN (midbrain)", "Neuromodulator": "Serotonin"},
    # Add more lines here if you expand to e.g., Vglut2, Vgat, Adra1a, etc.
}

# ---- Build metadata ----
metadata = []
for mouse in mice:
    row = df_genos[df_genos["Name"].astype(str).str.strip() == mouse]
    if not row.empty:
        genotype = row.iloc[0]["Genotype"]
    else:
        genotype = "Unknown"

    # Infer brain area and neuromodulator
    brain_area = "Unknown"
    neuromod = "Unknown"
    for key in mod_meta_map:
        if key in genotype:
            brain_area = mod_meta_map[key]["Brain Area"]
            neuromod = mod_meta_map[key]["Neuromodulator"]
            break

    metadata.append({
        "Mouse": mouse,
        "Genotype": genotype,
        "Brain Area": brain_area,
        "Neuromodulator": neuromod
    })

# ---- Save metadata to CSV ----
save_dir = r'I:\My Drive\Learning rules\BCI_data\Neuromodulator imaging'
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "mouse_genotypes.csv")

pd.DataFrame(metadata).to_csv(csv_path, index=False)
print(f"Saved mouse genotype metadata to:\n{csv_path}")
