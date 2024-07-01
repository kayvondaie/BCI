import os
from suite2p import default_ops as s2p_default_ops
from ScanImageTiffReader import ScanImageTiffReader
import folder_props_fun
import extract_scanimage_metadata
#import registration_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import shutil
import data_dict_create_module as ddc
import numpy as np
#import registration_functions
mpl.rcParams['figure.dpi'] = 300
from scipy.signal import medfilt
#import file selection tool
import tifffile as tiff
import tkinter as tk
from fileSelector_BCI import FolderSelectorApp
import sys
from PyQt6.QtWidgets import QApplication
import tkinter.messagebox as mb

#import scripts as functions
from processBCI_Bergamo_Utilities import *

# Message box display
mb.showinfo("Make sure Behavior Data Is In Session Folder", "Close second window if only selecting 1 folder")

#select files
if __name__ == '__main__':
    app = QApplication(sys.argv)
    firstOut = FolderSelectorApp(title='Select Old or Original Data Folder')
    firstOut.show()
    app.exec()
    # After the application has closed, you can access the selected_files variable
    print("Original Data:", firstOut.selected_folders)

    try:
        secondOut = FolderSelectorApp(title='Select New Data Folder Folder')
        secondOut.show()
        app.exec()
        # After the application has closed, you can access the selected_files variable
        print("New Data:", secondOut.selected_folders)
        dataPaths = [firstOut.selected_folders, secondOut.selected_folders]
    except:
        print('Only Selecting Initial Folder')
        dataPaths = [firstOut.selected_folders]
#assumes oldest data is "old_folder"
#i.e. select old folder and folder of interest, manually
dates_int = []
for dataPath in dataPaths:
    dates_int.append(int(dataPath.split('/')[-1]))
oldDate = str(min(dates_int))
old_folder = [dataPath for dataPath in dataPaths if oldDate in dataPath]
folder = [dataPath for dataPath in dataPaths if oldDate not in dataPath] #basically, the only other file... you should only be picking 2 files here



#Let Suite2p Do Its Magic
try:
    loadSuite2pROIS(folder, oldFolder=old_folder)
    print('Suite2p ROIs Created')
except Exception:
    print('Could not load suite2p rois')

#if only 1 file selected (i.e. no old folder), then old_folder becomes folder
if len(dataPaths) == 1:
    folder = old_folder
    old_folder = None
    data = ddc.load_data_dict(folder)
    oldData = None
else:
    data = ddc.load_data_dict(folder)
    oldData = ddc.load_data_dict(old_folder)

generateSessionSummary(data, folder, oldData = oldData)

print('Session Summary Generated')

findConditionedNeurons(data, folder)

print('CNS List Generated')

