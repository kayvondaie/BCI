import matplotlib.pyplot as plt
import numpy as np

def plot_cell_3():
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
        !pip install mat73
        import mat73
    %matplotlib inline

    mypath2 = '/data/bci/combined_new_old_060524.mat'
    BEHAV_DATA_PATH = '/data/bci_data/'

    print('Loading data_dict...')
    data_dict = mat73.loadmat(mypath2)
    print('Done!')

if __name__ == '__main__':
    plot_cell_3()
