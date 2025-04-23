import pandas as pd
from pathlib import Path
from datetime import datetime
from io import BytesIO
import requests
import os
import numpy as np
from suite2p import default_ops as s2p_default_ops
import folder_props_fun
from collections import Counter

# --- Parameters ---
mice = ["BCI88", "BCI93", "BCI103", "BCI104", "BCI105", "BCI107"]
base_dir = Path("//allen/aind/scratch/BCI/2p-raw")
cutoff_str = "010525"

# --- Load Google Sheet ---
url = "https://docs.google.com/spreadsheets/d/1dhKRnYQ7f5DlPyZ9UhKf9U2CueE7VWKRrQdS692yyZg/export?format=xlsx"
xls = pd.ExcelFile(BytesIO(requests.get(url).content))

# --- Prep session list ---
cutoff_date = datetime.strptime(cutoff_str, "%m%d%y")
all_sessions = []

for mouse in mice:
    try:
        df = xls.parse(mouse)
    except ValueError:
        print(f"⚠️ No sheet for {mouse}")
        continue

    df = df[["Date", "FOV"]].dropna()
    df["Date_dt"] = pd.to_datetime(df["Date"], format="%m%d%y", errors='coerce')
    df = df.dropna(subset=["Date_dt"])
    df = df[df["Date_dt"] > cutoff_date]

    session_records = []

    for _, row in df.iterrows():
        session = row["Date"].strftime("%m%d%y") if isinstance(row["Date"], datetime) else str(row["Date"])
        fov = row["FOV"]
        date_dt = row["Date_dt"]
        session_path = base_dir / mouse / session / "pophys"
        has_suite2p = (
            (session_path / "suite2p_BCI/plane0/stat.npy").exists() or
            (session_path / "suite2p_spont/plane0/stat.npy").exists()
        )

        session_records.append({
            "Mouse": mouse,
            "Session": session,
            "FOV": fov,
            "Date": date_dt,
            "SessionPath": session_path,
            "Has suite2p": has_suite2p,
            "OldFolderPath": None
        })

    # Sort and find previous suite2p in same FOV
    session_records.sort(key=lambda x: x["Date"])
    for i, sess in enumerate(session_records):
        prev_with_suite2p = [
            r for j, r in enumerate(session_records)
            if r["FOV"] == sess["FOV"] and j < i and r["Has suite2p"]
        ]
        if prev_with_suite2p:
            sess["Has previous suite2p in FOV"] = True
            sess["OldFolderPath"] = str(prev_with_suite2p[-1]["SessionPath"])
        else:
            sess["Has previous suite2p in FOV"] = False

    all_sessions.extend(session_records)

# --- DataFrame of all sessions ---
sessions_df = pd.DataFrame(all_sessions).sort_values(by=["Mouse", "FOV", "Date"]).reset_index(drop=True)

# --- Select sessions to run ---
sessions_to_run = sessions_df[~sessions_df["Has suite2p"]]
print(f"Found {len(sessions_to_run)} sessions to process.")
print(sessions_to_run[["Mouse", "Session", "FOV", "Has previous suite2p in FOV", "OldFolderPath"]])
#%%
import numpy as np
import json
from collections import Counter
from pathlib import Path

for _, row in sessions_df.iterrows():
    folder = Path(row["SessionPath"])
    fov = row["FOV"]

    if not (folder / "manual_inds.npy").exists():
        continue  # Skip if not manually tagged

    if (folder / "manual_ind_basenames.npy").exists():
        print(f"✓ Already converted: {folder.name}")
        continue

    try:
        folder_props = folder_props_fun.folder_props_fun(str(folder))
        bases = folder_props['bases']
        tif_bases = [fname.rsplit('_', 1)[0] for fname in folder_props['siFiles']]
        base_counts = Counter(tif_bases)

        ind = np.load(folder / "manual_inds.npy")
        ind_basenames = [bases[i] for i in ind]

        # Determine savefolders
        if 'neuron' in ind_basenames[0].lower():
            savefolders = {
                0: 'BCI',
                1: 'photostim_single',
                2: 'photostim_single2',
                3: 'spont_pre',
                4: 'spont_post'
            }
        else:
            savefolders = {
                0: 'spont_pre',
                1: 'photostim_single',
                2: 'photostim_single2',
                3: 'spont_post'
            }

        # Save new files
        np.save(folder / "manual_ind_basenames.npy", np.array(ind_basenames))
        with open(folder / "savefolders.json", "w") as f:
            json.dump(savefolders, f, indent=2)
        with open(folder / "FOV.txt", "w") as f:
            f.write(fov)

        print(f"✓ Converted {folder.name}: {ind_basenames}")

    except Exception as e:
        print(f"⚠️ Failed to convert {folder.name}: {e}")

#%%
import numpy as np
import os
import copy
import shutil
import json
from pathlib import Path
import suite2p
from suite2p import default_ops as s2p_default_ops
from suite2p.extraction.masks import create_masks
from suite2p.extraction.extract import extract_traces_from_masks
import extract_scanimage_metadata
import data_dict_create_module_test as ddc


def run_suite2p_session(folder: Path, bases: list, folder_props: dict):
    folder = Path(folder)

    # Load manually selected base names (order matters)
    ind_basenames_path = folder / "manual_ind_basenames.npy"
    if not ind_basenames_path.exists():
        raise FileNotFoundError(f"Missing {ind_basenames_path}")

    ind_basenames = np.load(ind_basenames_path, allow_pickle=True)

    # Load savefolder mapping
    with open(folder / "savefolders.json", "r") as f:
        savefolders = json.load(f)

    for ei, base in enumerate(ind_basenames):
        files = os.listdir(folder)
        good = [fi for fi, f in enumerate(files) if f.endswith('.tif') and f.rsplit('_', 1)[0] == base]
        tiff_list = [files[i] for i in good]

        if len(tiff_list) == 0:
            print(f"⚠️ No TIFFs found for base {base} in {folder}, skipping")
            continue

        # --- Build ops ---
        ops = s2p_default_ops()
        ops['data_path'] = [str(folder)]
        ops['tiff_list'] = tiff_list
        ops['do_registration'] = True
        ops['save_mat'] = True
        ops['do_bidiphase'] = True
        ops['reg_tif'] = False
        ops['delete_bin'] = 0
        ops['keep_movie_raw'] = 0
        ops['fs'] = 20
        ops['nchannels'] = 1
        ops['tau'] = 1
        ops['nimg_init'] = 500
        ops['nonrigid'] = False
        ops['smooth_sigma'] = .5
        ops['threshold_scaling'] = .7
        ops['batch_size'] = 250
        ops['do_regmetrics'] = False
        ops['allow_overlap'] = False
        ops['roidetect'] = True
        ops['save_folder'] = 'suite2p_' + savefolders[str(ei)]

        # --- Run suite2p ---
        ops = suite2p.run_s2p(ops)

        # --- Save siHeader metadata ---
        file = str(folder / ops['tiff_list'][0])
        siHeader = extract_scanimage_metadata.extract_scanimage_metadata(file)
        siBase = {i: ind_basenames[i] if i < len(ind_basenames) else '' for i in range(3)}
        siHeader['siBase'] = siBase
        siHeader['savefolders'] = savefolders

        save_path = Path(ops['save_path0']) / ops['save_folder'] / 'plane0'
        np.save(save_path / 'siHeader.npy', siHeader)

    # --- Regenerate data dictionary
    data = ddc.main(str(folder))
    return data

#%%
import pandas as pd
from pathlib import Path
from datetime import datetime
from io import BytesIO
import requests
import os
import numpy as np
import csv
import json
import folder_props_fun
#from run_suite2p_session import run_suite2p_session  # make sure this is saved separately

# --- Parameters ---
mice = ["BCI88", "BCI93", "BCI103", "BCI104", "BCI105", "BCI107"]
base_dir = Path("//allen/aind/scratch/BCI/2p-raw")
cutoff_str = "010525"
progress_log = Path("suite2p_progress.csv")

# --- Load Google Sheet ---
url = "https://docs.google.com/spreadsheets/d/1dhKRnYQ7f5DlPyZ9UhKf9U2CueE7VWKRrQdS692yyZg/export?format=xlsx"
xls = pd.ExcelFile(BytesIO(requests.get(url).content))

# --- Collect sessions to process ---
cutoff_date = datetime.strptime(cutoff_str, "%m%d%y")
all_sessions = []

for mouse in mice:
    try:
        df = xls.parse(mouse)
    except ValueError:
        print(f"⚠️ No sheet for {mouse}")
        continue

    df = df[["Date", "FOV"]].dropna()
    df["Date_dt"] = pd.to_datetime(df["Date"], format="%m%d%y", errors='coerce')
    df = df.dropna(subset=["Date_dt"])
    df = df[df["Date_dt"] > cutoff_date]

    session_records = []

    for _, row in df.iterrows():
        session = row["Date"].strftime("%m%d%y") if isinstance(row["Date"], datetime) else str(row["Date"])
        fov = row["FOV"]
        date_dt = row["Date_dt"]
        session_path = base_dir / mouse / session / "pophys"
        has_suite2p = (
            (session_path / "suite2p_BCI/plane0/stat.npy").exists() or
            (session_path / "suite2p_spont/plane0/stat.npy").exists()
        )

        session_records.append({
            "Mouse": mouse,
            "Session": session,
            "FOV": fov,
            "Date": date_dt,
            "SessionPath": session_path,
            "Has suite2p": has_suite2p,
            "OldFolderPath": None
        })

    session_records.sort(key=lambda x: x["Date"])
    for i, sess in enumerate(session_records):
        prev = [
            r for j, r in enumerate(session_records)
            if r["FOV"] == sess["FOV"] and j < i and r["Has suite2p"]
        ]
        if prev:
            sess["Has previous suite2p in FOV"] = True
            sess["OldFolderPath"] = str(prev[-1]["SessionPath"])
        else:
            sess["Has previous suite2p in FOV"] = False

    all_sessions.extend(session_records)

# --- DataFrame of sessions to run ---
sessions_df = pd.DataFrame(all_sessions).sort_values(by=["Mouse", "FOV", "Date"]).reset_index(drop=True)
sessions_to_run = sessions_df[~sessions_df["Has suite2p"]]

print(f"\n🔎 Found {len(sessions_to_run)} sessions to run.")

# --- Start logging ---
log_exists = progress_log.exists()
with open(progress_log, mode='a', newline='') as log_file:
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["Mouse", "Session", "Status", "Message"])

    # --- Main processing loop ---
    for _, row in sessions_to_run.iterrows():
        folder = Path(row["SessionPath"])
        mouse = row["Mouse"]
        session = row["Session"]

        print(f"\n▶ {mouse} {session}")

        try:
            # Skip if already processed
            has_suite2p_outputs = (
                (folder / "suite2p_BCI/plane0/stat.npy").exists() or
                (folder / "suite2p_spont/plane0/stat.npy").exists()
            )
            
            if has_suite2p_outputs:
                print(f"✓ Skipping {folder} (Suite2p output already exists)")
                log_writer.writerow([mouse, session, "skipped", "suite2p output exists"])
                continue


            # Run session
            folder_props = folder_props_fun.folder_props_fun(str(folder))
            bases = folder_props['bases']
            if not (folder / "manual_inds.npy").exists() or not (folder / "savefolders.json").exists():
                print(f"⚠️ Warning: {folder} is missing manual_inds or savefolders.")

            run_suite2p_session(folder, bases, folder_props)

            log_writer.writerow([mouse, session, "success", ""])
        except Exception as e:
            print(f"⚠️ Failed on {folder}: {e}")
            log_writer.writerow([mouse, session, "error", str(e)])
            continue
