# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:42:08 2023

@author: scanimage
"""

import os
import re

def folder_props_fun(folder):
    if not folder.endswith('/'):
        folder = folder + '/'
        
    files = os.listdir(folder)

    siFiles = [f for f in files if re.search('\.tif$', f)]
    wsFiles = [f for f in files if re.search('\.h5$', f)]

    folder_props = {'siFiles': siFiles, 'wsFiles': wsFiles, 'folder': folder}
    
    base = []
    for name in siFiles:
        a = max([i for i, c in enumerate(name) if c == '_'])
        base.append(name[:a])
        folder_props['bases'] = list(set(base))
    
    
    return folder_props