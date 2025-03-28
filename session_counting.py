import os
from pathlib import Path
import datetime
import pandas as pd

def counter():    
    base_dir = Path(r'//allen/aind/scratch/BCI/2p-raw')
    mice = ["BCI88", "BCI93", "BCI102", "BCI103", "BCI104", "BCI105","BCI106","BCI107"]
    cutoff_str = "010525"
    cutoff_date = datetime.datetime.strptime(cutoff_str, "%m%d%y").date()
    
    session_counts = {mouse: 0 for mouse in mice}
    data = []  # Store data for DataFrame output
    
    for mouse in mice:
        mouse_dir = base_dir / mouse
        if not mouse_dir.is_dir():
            continue
        
        for item in mouse_dir.iterdir():
            if item.is_dir():
                try:
                    session_date = datetime.datetime.strptime(item.name, "%m%d%y").date()
                except ValueError:
                    continue
                
                if session_date > cutoff_date:
                    pophys_dir = item / "pophys"
                    if pophys_dir.is_dir():
                        session_counts[mouse] += 1
                        
                        # Check for existence of 'data_main.npy'
                        has_data_main = (pophys_dir / 'data_main.npy').is_file()
                        
                        # Find unique TIFF file stems
                        tiff_files = list(pophys_dir.glob("*.tif"))
                        file_stems = [f.stem.split('_')[0] for f in tiff_files]
                        
                        # Count occurrences of each stem
                        stem_counts = {stem: file_stems.count(stem) for stem in set(file_stems)}
                        
                        # Identify neuron stems with at least 40 files
                        has_neuron_sequence = any(stem.startswith("neuron") and count >= 40 
                                                  for stem, count in stem_counts.items())
                        
                        # Append data
                        data.append([mouse, item.name, has_data_main, has_neuron_sequence])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["Mouse", "Session", "Has data_main.npy", "Has 40+ neuron TIFs"])
    
    # Display results

    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_columns", None)  # Show all columns
    print(df)
    df.to_csv(r'//allen/aind/scratch/BCI/2p-raw'+"/session_data2.csv", index=False)


    return df

def counter2(mice, cutoff_str):
    base_dir = Path(r'//allen/aind/scratch/BCI/2p-raw')
    cutoff_date = datetime.datetime.strptime(cutoff_str, "%m%d%y").date()

    session_counts = {mouse: 0 for mouse in mice}
    data = []  # Store data for DataFrame output

    for mouse in mice:
        mouse_dir = base_dir / mouse
        if not mouse_dir.is_dir():
            continue

        for item in mouse_dir.iterdir():
            if item.is_dir():
                try:
                    session_date = datetime.datetime.strptime(item.name, "%m%d%y").date()
                except ValueError:
                    continue

                if session_date > cutoff_date:
                    pophys_dir = item / "pophys"
                    if pophys_dir.is_dir():
                        session_counts[mouse] += 1

                        # Check for existence of 'data_main.npy'
                        has_data_main = (pophys_dir / 'data_main.npy').is_file()

                        # Find unique TIFF file stems
                        tiff_files = list(pophys_dir.glob("*.tif"))
                        file_stems = [f.stem.split('_')[0] for f in tiff_files]

                        # Count occurrences of each stem
                        stem_counts = {stem: file_stems.count(stem) for stem in set(file_stems)}

                        # Identify neuron stems with at least 40 files
                        has_neuron_sequence = any(
                            stem.startswith("neuron") and count >= 40
                            for stem, count in stem_counts.items()
                        )

                        # Check if any subdirectory in pophys contains 'suite2p' in its name
                        has_suite2p = any(
                            subdir.is_dir() and 'suite2p' in subdir.name.lower()
                            for subdir in pophys_dir.iterdir()
                        )

                        # Append data
                        data.append([
                            mouse,
                            item.name,
                            has_data_main,
                            has_neuron_sequence,
                            has_suite2p
                        ])

    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        "Mouse", "Session", "Has data_main.npy", "Has 40+ neuron TIFs", "Has suite2p subdir"])

    # Display results
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_columns", None)  # Show all columns
    print(df)

    return df
