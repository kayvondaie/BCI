from utils.helper_functions1 import get_data_dict

import pickle
def run():
    # Cell 3
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import scipy.stats as stats
import io
try:
    import mat73
except:
    import mat73

mypath2 = '/data/bci/combined_new_old_060524.mat'
BEHAV_DATA_PATH = '/data/bci_data/'

print('Loading data_dict...')
data_dict = get_data_dict()
    with open('outputs/data_dict.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
print('Done!')

    print("✅ load_data ran successfully.")
if __name__ == '__main__':
    run()