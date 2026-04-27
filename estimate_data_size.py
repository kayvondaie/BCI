"""
Estimate the total size of data files across the ~46 analysis sessions.
Uses subprocess `du` calls instead of Python rglob for speed over network drives.
"""

import sys, os, csv, subprocess
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import numpy as np

RESULTS_DIR = os.path.join(_THIS_DIR, 'meta_analysis_results')
BASE_DIR = Path(r'//allen/aind/scratch/BCI/2p-raw')

# ── Identify sessions ───────────────────────────────────────────────────
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]
list_of_dirs = session_counting.counter()

_qc_csv = os.path.join(RESULTS_DIR, 'qc', 'qc_summary.csv')
_qc_fail = set()
if os.path.exists(_qc_csv):
    with open(_qc_csv) as f:
        for r in csv.DictReader(f):
            if r['pass_qc'] != 'True':
                _qc_fail.add((r['mouse'], r['session']))
    print(f"QC filter: {len(_qc_fail)} sessions excluded")

sessions = []
for mouse in mice:
    inds = np.where(
        (list_of_dirs['Mouse'] == mouse) &
        (list_of_dirs['Has data_main.npy'] == True)
    )[0]
    for idx in inds:
        m = list_of_dirs['Mouse'][idx]
        s = list_of_dirs['Session'][idx]
        if (m, s) not in _qc_fail:
            sessions.append((m, s))

print(f"\n{len(sessions)} sessions to scan\n")

# ── Helpers ──────────────────────────────────────────────────────────────
def du(path):
    """Get size in bytes via du -sb. Returns 0 if path doesn't exist."""
    try:
        out = subprocess.check_output(
            ['du', '-sb', str(path)], stderr=subprocess.DEVNULL, timeout=120)
        return int(out.split()[0])
    except Exception:
        return 0

def fmt(nbytes):
    if nbytes < 1024**2:
        return f"{nbytes/1024:.1f} KB"
    elif nbytes < 1024**3:
        return f"{nbytes/1024**2:.1f} MB"
    else:
        return f"{nbytes/1024**3:.2f} GB"

# ── Scan ─────────────────────────────────────────────────────────────────
totals = {'behavior': 0, 'pophys': 0, 'root_files': 0}
session_sizes = []

for mi, (mouse, session) in enumerate(sessions):
    sess_dir = BASE_DIR / mouse / session

    beh_sz = du(sess_dir / 'behavior')
    pop_sz = du(sess_dir / 'pophys')

    # Root-level files: total minus all subdirectories
    total_sz = du(sess_dir)
    # Subtract known subdirs (behavior, behavior_video, pophys) to get root files
    vid_sz = du(sess_dir / 'behavior_video')
    root_sz = max(0, total_sz - beh_sz - pop_sz - vid_sz)

    keep_sz = beh_sz + pop_sz + root_sz

    totals['behavior'] += beh_sz
    totals['pophys'] += pop_sz
    totals['root_files'] += root_sz
    session_sizes.append((mouse, session, keep_sz))

    print(f"  [{mi+1:2d}/{len(sessions)}] {mouse} {session}:  "
          f"behavior={fmt(beh_sz):>9s}  pophys={fmt(pop_sz):>9s}  "
          f"total={fmt(keep_sz):>9s}")

# ── Summary ──────────────────────────────────────────────────────────────
grand_total = sum(totals.values())
sizes = [s[2] for s in session_sizes]

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Sessions:          {len(session_sizes)}")
print(f"  behavior/ total:   {fmt(totals['behavior']):>12s}")
print(f"  pophys/ total:     {fmt(totals['pophys']):>12s}")
print(f"  session root:      {fmt(totals['root_files']):>12s}")
print(f"  {'─'*40}")
print(f"  GRAND TOTAL:       {fmt(grand_total):>12s}")
print()
if sizes:
    print(f"  Per session:  min={fmt(min(sizes))}  median={fmt(int(np.median(sizes)))}  max={fmt(max(sizes))}")
