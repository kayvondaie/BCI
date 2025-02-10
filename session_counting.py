import os
from pathlib import Path
import datetime

def counter():    
    
    base_dir = Path(r'//allen/aind/scratch/BCI/2p-raw')
    mice = ["BCI88", "BCI93", "BCI102", "BCI103", "BCI104", "BCI105"]
    cutoff_str = "010525"
    cutoff_date = datetime.datetime.strptime(cutoff_str, "%m%d%y").date()
    
    session_counts = {mouse: 0 for mouse in mice}
    list_of_dirs = []
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
                        list_of_dirs.append(pophys_dir)
    
    # Print results in a simple comma-separated format
    print("Mouse,Number of Sessions After " + cutoff_str)
    for mouse in mice:
        print(f"{mouse},{session_counts[mouse]}")
    return list_of_dirs