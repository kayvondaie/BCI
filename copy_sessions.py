"""
Copy the ~55 analysis sessions to F:\three_factor_data_2025, preserving
the directory structure ({mouse}/{session}/...).

Copies per session:
  - behavior/          (full)
  - pophys/            (excluding *.tif, *.tiff, and loose *.npy in pophys root)
  - session root files (*.json, *.pdf, etc.)

Uses robocopy for speed over the network.
"""

import sys, os, csv, subprocess
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import numpy as np

RESULTS_DIR = os.path.join(_THIS_DIR, 'meta_analysis_results')
SRC_BASE = Path(r'\\allen\aind\scratch\BCI\2p-raw')
DST_BASE = Path(r'F:\three_factor_data_2025')

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

print(f"{len(sessions)} sessions to copy\n")

# ── Copy ─────────────────────────────────────────────────────────────────
failed = []

for mi, (mouse, session) in enumerate(sessions):
    src_sess = SRC_BASE / mouse / session
    dst_sess = DST_BASE / mouse / session
    tag = f"[{mi+1:2d}/{len(sessions)}] {mouse} {session}"

    if not src_sess.is_dir():
        print(f"  {tag}  MISSING — skipped")
        failed.append((mouse, session, "source missing"))
        continue

    print(f"  {tag}  copying...", end="", flush=True)
    sess_errors = []

    def run_robocopy(src, dst, extra_args, label, timeout=600):
        """Run robocopy, return True on success. Appends to sess_errors on failure."""
        try:
            res = subprocess.run(
                ['robocopy', str(src), str(dst)] + extra_args +
                ['/NJH', '/NJS', '/NDL', '/NC', '/NS', '/NP'],
                capture_output=True, timeout=timeout)
            if res.returncode >= 8:
                sess_errors.append(f"{label}: robocopy exit {res.returncode}")
                return False
            return True
        except subprocess.TimeoutExpired:
            sess_errors.append(f"{label}: timed out after {timeout}s")
            return False

    # --- Session root files (json, pdf, etc.) ---
    run_robocopy(src_sess, dst_sess, ['/LEV:1'], 'root files', timeout=300)

    # --- behavior/ ---
    src_beh = src_sess / 'behavior'
    dst_beh = dst_sess / 'behavior'
    if src_beh.is_dir():
        run_robocopy(src_beh, dst_beh, ['/E'], 'behavior')

    # --- pophys/ (exclude *.tif, *.tiff, and loose *.npy) ---
    src_pop = src_sess / 'pophys'
    dst_pop = dst_sess / 'pophys'
    if src_pop.is_dir():
        # Loose files in pophys root: exclude .tif, .tiff, .npy
        run_robocopy(src_pop, dst_pop,
                     ['/LEV:1', '/XF', '*.tif', '*.tiff', '*.npy'],
                     'pophys root')

        # suite2p_* subdirectories: copy fully
        for d in src_pop.iterdir():
            if d.is_dir() and d.name.startswith('suite2p'):
                run_robocopy(d, dst_pop / d.name, ['/E'],
                             f'suite2p/{d.name}', timeout=1200)

    # --- Create empty data_main.npy so session_counting.counter() finds it ---
    if dst_pop.is_dir():
        stub = dst_pop / 'data_main.npy'
        if not stub.exists():
            np.save(str(stub), np.array([]))

    if sess_errors:
        failed.append((mouse, session, '; '.join(sess_errors)))
        print(f"  ERRORS: {'; '.join(sess_errors)}")
    else:
        print("  done")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Finished: {len(sessions) - len(failed)}/{len(sessions)} sessions copied")
if failed:
    print(f"\nFailed ({len(failed)}):")
    for m, s, reason in failed:
        print(f"  {m} {s}: {reason}")
else:
    print("No errors.")
print(f"\nDestination: {DST_BASE}")
