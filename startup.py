import os
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir)
os.chdir(relative_path)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
