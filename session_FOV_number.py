import os
import numpy as np
import pandas as pd
from datetime import datetime

base_dir = '//allen/aind/scratch/BCI/2p-raw'
mice = ['BCI102', 'BCI109']
cutoff_date = datetime.strptime('2025-01-08', '%Y-%m-%d')

rows = []

for mouse in mice:
    mouse_dir = os.path.join(base_dir, mouse)
    if not os.path.isdir(mouse_dir):
        continue

    for session in sorted(os.listdir(mouse_dir)):
        try:
            session_date = datetime.strptime(session, '%m%d%y')
        except ValueError:
            continue  # skip folders that aren't in mmddyy format

        if session_date < cutoff_date:
            continue  # skip sessions before the cutoff

        session_dir = os.path.join(mouse_dir, session, 'pophys', 'suite2p_BCI', 'plane0')
        f_path = os.path.join(session_dir, 'iscell.npy')

        if os.path.exists(f_path):
            try:
                iscell = np.load(f_path, allow_pickle=True)
                num_rois = iscell.shape[0]
                rows.append({'mouse': mouse, 'session': session, 'num_rois': num_rois})
            except Exception as e:
                print(f"Error loading {f_path}: {e}")

# Convert to DataFrame
if rows:
    df = pd.DataFrame(rows).sort_values(by=['mouse', 'session'])

    # Assign FOVs
    fov_labels = []
    for mouse, group in df.groupby('mouse'):
        prev_n = None
        fov = 1
        for _, row in group.iterrows():
            if prev_n is None or row['num_rois'] != prev_n:
                prev_n = row['num_rois']
                fov_label = f'FOV{fov}'
                fov += 1
            fov_labels.append(fov_label)

    df['FOV'] = fov_labels
    print(df[['mouse', 'session', 'FOV']])
else:
    print("No valid sessions found after cutoff date.")
