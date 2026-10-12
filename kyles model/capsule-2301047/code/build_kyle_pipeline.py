#%% ============================================================================
# CELL 1: Build an importable pipeline module from Kyle's notebook
# ============================================================================
"""
Run this ONCE (in Spyder) to generate kyle_pipeline.py.

Kyle's training driver (train_task) and param builder (default_toy_params) live
inside bci_toy_setup.ipynb, resting on a large web of helper functions defined
throughout that notebook. This script parses the notebook and pulls out *every*
top-level function / class / import statement into a single importable module,
kyle_pipeline.py, so we can drive his model from a plain .py without a notebook.

Only definitions and imports are extracted -- execution and plotting statements
are dropped. Where the notebook redefines a function in a later cell, the later
definition wins (matching the notebook's end-of-run state), because statements
are written in notebook order and Python keeps the last definition.
"""
import os, sys, json, ast, re

# IPython magics (%...) and shell escapes (!...) break ast.parse; blank them out
# so the real import/def statements in a mixed cell still parse.
_MAGIC_RE = re.compile(r'^\s*[%!]')
def strip_magics(src):
    return '\n'.join('' if _MAGIC_RE.match(ln) else ln
                     for ln in src.splitlines())

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
NB_PATH  = os.path.join(CODE_DIR, 'bci_toy_setup.ipynb')
OUT_PATH = os.path.join(CODE_DIR, 'kyle_pipeline.py')

KEEP_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
              ast.Import, ast.ImportFrom)

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

kept_segments = []
seen_defs = set()          # dedup: keep the FIRST definition of each name
n_cells = 0
n_syntax_skip = 0
n_dup_skip = 0
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    n_cells += 1
    src = strip_magics(''.join(cell['source']))
    if not src.strip():
        continue
    try:
        tree = ast.parse(src)
    except SyntaxError:
        n_syntax_skip += 1        # cell still unparseable after stripping magics
        continue
    # Statement-level extraction: keep each def/class/import even if the cell
    # also contains execution code around it. For functions/classes, keep only
    # the FIRST definition of a given name -- the notebook's coherent toy-setup
    # pipeline appears first; stale alternate redefinitions trail later.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in seen_defs:
                n_dup_skip += 1
                continue
            seen_defs.add(node.name)
            seg = ast.get_source_segment(src, node)
            if seg:
                kept_segments.append(seg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(src, node)
            if seg:
                kept_segments.append(seg)
        elif isinstance(node, ast.Assign):
            # Keep ONLY top-level assignments whose RHS is a pure literal
            # constant (int/float/str/tuple/list/dict of literals), e.g.
            # MAX_L = 1000. literal_eval rejects subscripts/names/calls, so we
            # don't drag in execution that references incomplete module state.
            if all(isinstance(t, ast.Name) for t in node.targets):
                try:
                    ast.literal_eval(node.value)
                except Exception:
                    continue
                seg = ast.get_source_segment(src, node)
                if seg:
                    kept_segments.append(seg)

header = (
    '"""AUTO-GENERATED from bci_toy_setup.ipynb by build_kyle_pipeline.py.\n'
    'Contains only definition/import statements from the notebook (no execution).\n'
    'Do NOT edit by hand -- re-run build_kyle_pipeline.py to regenerate."""\n'
    'import os as _os, sys as _sys\n'
    '_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))\n\n'
)

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(header)
    f.write('\n\n'.join(kept_segments))
    f.write('\n')

print(f"Wrote {OUT_PATH}")
print(f"  scanned {n_cells} code cells, "
      f"extracted {len(kept_segments)} def/class/import statements, "
      f"skipped {n_syntax_skip} magic/shell cells, "
      f"dropped {n_dup_skip} duplicate redefinitions")

#%% ============================================================================
# CELL 2: Sanity check -- import the generated module and confirm entry points
# ============================================================================
# Re-import cleanly in case it was imported before.
import importlib
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if 'kyle_pipeline' in sys.modules:
    del sys.modules['kyle_pipeline']
import kyle_pipeline as kp

for name in ('default_toy_params', 'train_task'):
    print(f"  {name}: {'FOUND' if hasattr(kp, name) else 'MISSING'}")
