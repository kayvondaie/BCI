# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:20:56 2022

@author: scanimage
"""

import BCI_analysis as bci
file_path = 'D:/KD/BCI_data/BCI_2022/photostim_learning/v2/combined_new_old2.mat'
data = bci.io.io_matlab.read_multisession_mat(file_path)