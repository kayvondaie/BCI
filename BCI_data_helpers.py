# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:21:19 2025

@author: kayvon.daie
"""

import numpy as np
import re

def parse_hdf5_array_string(array_raw, trl):
    """
    Parses an HDF5-stored string or list of arrays, typically from 'step_time' or 'reward_time'.

    Parameters:
        array_raw : str or list/array
            Raw array loaded from HDF5 file, possibly as a string representation.
        trl : int
            Expected number of trials to pad the output if needed.

    Returns:
        parsed : np.ndarray (dtype=object)
            List of arrays, one per trial, padded to length `trl`.
    """
    if isinstance(array_raw, str):
        pattern = r'array\(\[([^\]]*)\](?:, dtype=float64)?\)'
        matches = re.findall(pattern, array_raw.replace('\n', ''))

        parsed = []
        for match in matches:
            try:
                if match.strip() == '':
                    parsed.append(np.array([]))
                else:
                    arr = np.fromstring(match, sep=',')
                    parsed.append(arr)
            except Exception as e:
                print("Skipping array due to error:", e)

        pad_len = trl - len(parsed)
        if pad_len > 0:
            parsed += [np.array([])] * pad_len

        return np.array(parsed, dtype=object)

    else:
        if len(array_raw) < trl:
            pad_len = trl - len(array_raw)
            return np.array(list(array_raw) + [np.array([])] * pad_len, dtype=object)
        return array_raw
