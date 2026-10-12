"""
Package the NM (neuromodulator / LC-axon) sessions to F:\nm_data_2026 for
Code Ocean upload, preserving {mouse}/{session}/ structure.

Mirrors copy_sessions.py (the 3-factor packager) as closely as the NM layout
allows, so the bundle is complete for re-running any NM analysis on CO:
  - session metadata + data_main_*.npy   (root files, minus raw *.tif/*.tiff)
  - behavior/                            (full)
  - all suite2p* subdirs                 (full, /E)

Two NM-specific differences from copy_sessions.py:
  1. NM data lives at the SESSION ROOT, not under pophys/ (confirmed: data_main
     and suite2p_BCI/suite2p_ch1 sit at {mouse}/{session}/). The loader resolves
     pophys/ first then root, so either layout is preserved.
  2. copy_sessions.py EXCLUDES loose *.npy (its data_main is a rebuilt stub). For
     NM, data_main_{mouse}_{session}_BCI.npy is the REAL product, so *.npy is KEPT;
     only raw *.tif/*.tiff movies are excluded (same as 3-factor).

Session selection mirrors LC_axon_analysis/save_LC_data_on_Gdrive2.py: enumerate
with counter2(mice,'010112',has_pophys=False), include a session iff its
data_main_*_BCI.npy exists on the share (the missing-file skip IS the QC filter).

NOTE: like the 3-factor bundle (~311 GB), this copies full suite2p folders and
will be large. After running, upload F:\nm_data_2026 to Code Ocean as a data asset.
"""

import sys, os, subprocess
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import session_counting

SRC_BASE = Path(r'\\allen\aind\scratch\BCI\2p-raw')
DST_BASE = Path(r'F:\nm_data_2026')
SUFFIX   = 'BCI'

# NM cohort — matches save_LC_data_on_Gdrive2.py (note BCINM_023, not 021)
mice = ["BCINM_017", "BCINM_023", "BCINM_024", "BCINM_027", "BCINM_031", "BCINM_034"]

# has_pophys=False: NM sessions store data at the session root, not under pophys/
sessions = session_counting.counter2(mice, '010112', has_pophys=False)

print(f"{len(sessions)} candidate sessions; copying those with a data_main_*_{SUFFIX}.npy\n")

copied, skipped, failed = 0, 0, []


def run_robocopy(src, dst, extra_args, label, sess_errors, timeout=1800):
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


for i in range(len(sessions)):
    mouse   = sessions['Mouse'][i]
    session = sessions['Session'][i]
    fname   = f"data_main_{mouse}_{session}_{SUFFIX}.npy"
    tag     = f"[{i+1:3d}/{len(sessions)}] {mouse} {session}"

    session_dir = SRC_BASE / mouse / session

    # Resolve the source folder holding data_main (pophys/ first, then session root)
    src_folder = None
    rel = ''  # subpath under {mouse}/{session} to preserve in the destination
    for cand, r in ((session_dir / 'pophys', 'pophys'), (session_dir, '')):
        if (cand / fname).is_file():
            src_folder, rel = cand, r
            break

    if src_folder is None:
        # No data_main → not a "good" session. This is the NM inclusion filter.
        skipped += 1
        continue

    dst_folder = DST_BASE / mouse / session
    dst_target = dst_folder / rel if rel else dst_folder
    sess_errors = []
    print(f"  {tag}  copying...", end="", flush=True)

    # 1. Root/metadata + data_main (keep *.npy; drop raw movies) — mirrors copy_sessions.py
    #    root-file step, but KEEPS npy because NM data_main is real, not a stub.
    run_robocopy(src_folder, dst_target, ['/LEV:1', '/XF', '*.tif', '*.tiff'],
                 'root+data_main', sess_errors, timeout=900)

    # 2. suite2p* subdirs — full, exactly like copy_sessions.py
    for d in src_folder.iterdir():
        if d.is_dir() and d.name.lower().startswith('suite2p'):
            run_robocopy(d, dst_target / d.name, ['/E'], f'suite2p/{d.name}', sess_errors)

    # 3. behavior/ (always at the session root) — full, like copy_sessions.py
    src_beh = session_dir / 'behavior'
    if src_beh.is_dir():
        run_robocopy(src_beh, dst_folder / 'behavior', ['/E'], 'behavior', sess_errors)

    if sess_errors:
        failed.append((mouse, session, '; '.join(sess_errors)))
        print(f"  ERRORS: {'; '.join(sess_errors)}")
    else:
        print("  done")
        copied += 1

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Copied: {copied}   |   No data_main (excluded): {skipped}   |   Failed: {len(failed)}")
if failed:
    print("\nFailed:")
    for m, s, reason in failed:
        print(f"  {m} {s}: {reason}")
print(f"\nDestination: {DST_BASE}")
