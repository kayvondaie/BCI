# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:20:16 2023

@author: kayvon.daie
"""

import os;os.chdir(r'H:/My Drive/Python Scripts/BCI_analysis/');import data_dict_create_module as ddc
import suite2p
import os
import re
from suite2p import default_ops as s2p_default_ops
from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
import folder_props_fun
import extract_scanimage_metadata
#import registration_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
import copy
import shutil