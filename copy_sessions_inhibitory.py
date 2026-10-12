"""
Package the inhibitory-photostim sessions to F:\inhibitory_data_2026 for
Code Ocean upload, preserving {mouse}/{session}/ structure.

Session selection reuses the SAME QC filter as
three_factor_variance_explained_inhibitory.py:
    Type == Inhibitory, (Pre OR Post connectivity mapping in {good, ok}),
    and no non-zero offset note (offset = [1-9]) in the notes column.
This is driven by the CSV inventory, so it picks exactly the "good" sessions
the analysis uses.

Copies per session (like copy_sessions.py):
  - behavior/          (full)
  - pophys/            (excluding *.tif, *.tiff, and loose *.npy in pophys root)
  - session root files (*.json, *.pdf, etc.)
  - all suite2p* subdirs (incl. suite2p_photostim*) — full

data_main.npy is rebuilt on CO, so an empty stub is written (as in copy_sessions.py).
Uses robocopy for speed over the network.

After running, upload F:\inhibitory_data_2026 to Code Ocean as a data asset.
"""

import sys, os, csv, re, subprocess
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np

SRC_BASE = Path(r'\\allen\aind\scratch\BCI\2p-raw')
DST_BASE = Path(r'F:\inhibitory_data_2026')

# ── Identify sessions via the CSV QC filter (same as the analysis script) ──
_QC_CSV        = os.path.join(_THIS_DIR, 'Inhibitory & Pan-neuronal BCI summary - Sheet1.csv')
_OK_QUALITY    = {'good', 'ok'}
_BAD_OFFSET_RE = re.compile(r'offset\s*=\s*[1-9]', re.IGNORECASE)

_qc_keep = set()
_qc_dropped_offset = []
_cur_m = _cur_t = None

with open(_QC_CSV, encoding='utf-8') as _f:
    _rows = list(csv.reader(_f))
for _row in _rows[2:]:
    if not _row:
        continue
    _subj = _row[0].strip() if len(_row) > 0 else ''
    _type = _row[1].strip() if len(_row) > 1 else ''
    if _subj: _cur_m = _subj
    if _type: _cur_t = _type
    if (_cur_t or '').lower() != 'inhibitory':
        continue
    _date  = _row[2].strip()         if len(_row) > 2 else ''
    _pre   = _row[4].strip().lower() if len(_row) > 4 else ''
    _post  = _row[5].strip().lower() if len(_row) > 5 else ''
    _notes = _row[6]                 if len(_row) > 6 else ''
    if (not _date or _cur_m is None or
        (_pre not in _OK_QUALITY and _post not in _OK_QUALITY)):
        continue
    try:
        _m, _d, _y = _date.split('/')
        _session = f'{int(_m):02d}{int(_d):02d}{int(_y):02d}'
    except Exception:
        continue
    if _BAD_OFFSET_RE.search(_notes or ''):
        _qc_dropped_offset.append((_cur_m, _session, (_notes or '').strip()))
        continue
    _qc_keep.add((_cur_m, _session))

sessions = sorted(_qc_keep)
mice = sorted({m for m, _ in _qc_keep})
print(f"QC CSV: kept {len(sessions)} inhibitory sessions across mice {mice}")
if _qc_dropped_offset:
    print(f"QC CSV: dropped {len(_qc_dropped_offset)} sessions for non-zero offset:")
    for _m, _s, _n in _qc_dropped_offset:
        print(f'   {_m} {_s}  — "{_n}"')
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

    # --- pophys/ (exclude only raw *.tif/*.tiff movies) ---
    src_pop = src_sess / 'pophys'
    dst_pop = dst_sess / 'pophys'
    if src_pop.is_dir():
        # Unlike copy_sessions.py (3-factor), KEEP loose *.npy here: inhibitory
        # sessions carry real products (data_main_*_BCI.npy, data_photostim*.npy),
        # not a rebuilt stub. The analyses read the *.h5 via ddct.load_hdf5, but we
        # keep the .npy too so the bundle is complete for every inhibitory script.
        run_robocopy(src_pop, dst_pop,
                     ['/LEV:1', '/XF', '*.tif', '*.tiff'],
                     'pophys root')

        # suite2p_* subdirectories: copy fully (incl. suite2p_photostim*)
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
