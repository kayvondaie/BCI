import suite2p
import os, re, copy, shutil, sys
from suite2p import default_ops as s2p_default_ops
from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
# import folder_props_fun, extract_scanimage_metadata
import extract_scanimage_metadata
#import registration_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
import tkinter as tk
from tkinter import simpledialog
import plotting_functions as pf
from scipy.signal import medfilt
from glob import glob
import data_dict_create_module as ddc


def getBases(folder):
    folder_props = folder_props_fun(folder) 
    bases = folder_props['bases']
    return bases

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
        if name[:a].find('000') != -1:
            #handling wierd and rare exception...
            name_fix = name[:a].split('000')[0]
            base.append(name_fix[:-1])
        else:
            base.append(name[:a])
        folder_props['bases'] = list(set(base))
    
    
    return folder_props

def loadSuite2pROIS(basesInd, folder, old_folder):
    #If old exists
    if old_folder is not None:
        print('Using Old Folder', old_folder)
        stat_old = np.load(old_folder + r'suite2p_BCI/' + r'/plane0/stat.npy',allow_pickle = 'True')
        ops_old = np.load(old_folder + r'suite2p_BCI/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
        iscell_old = np.load(old_folder +r'suite2p_BCI/' + r'/plane0/iscell.npy',allow_pickle = 'True')
    else:
        print('Not Using Old Folder')
        stat_old = None
        ops_old = None
        iscell_old = None
    

    ops = s2p_default_ops()

    #SETTING PARAMETERS FOR SUITE2P
    ops['data_path'] = folder
    folder = ops['data_path']
    folder_props = folder_props_fun(folder) #using local to not mess up other scripts
    bases = folder_props['bases']
    #Adjusting this so that filenames are more accurate
    savefolders = dict()
    savefolders[1] = 'spont'
    savefolders[0] = 'BCI' 
    if basesInd is None:
        print("Aborting: No Bases Selected")
        sys.exit()
    else:
        print("User input:", basesInd)
    
    #Initiate Suite2p with specified settings
    #############################################################################################################
    for ei in range(0,len(basesInd)):
        if ei == 1: 
            old_folder = folder
        if old_folder is not None:
            stat_old = np.load(old_folder + r'suite2p_BCI/' + r'/plane0/stat.npy',allow_pickle = 'True')
            ops_old = np.load(old_folder + r'suite2p_BCI/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
            iscell_old = np.load(old_folder +r'suite2p_BCI/' + r'/plane0/iscell.npy',allow_pickle = 'True')
        
        #base = {base for i, base in enumerate(bases) if str(i) in ind}
        base = bases[int(basesInd[ei])]
        siFiles = folder_props['siFiles'] #not sure what this is doing... siFiles does not get called after this...
        
        #grabs only the tif files that are in base
        files = glob(folder + '*.tif')
        good = np.array([filePath  for filePath in files if base in filePath])

        #only storing the good trials
        ops['tiff_list'] = good

        #more suite2p parameter setting
        ops['do_registration']=True
        ops['save_mat'] = True
        ops['do_bidiphase'] = True
        ops['reg_tif'] = False # save registered movie as tif files
        ops['delete_bin'] = 0
        ops['keep_movie_raw'] = 0
        ops['fs'] = 20
        ops['nchannels'] = 1
        ops['tau'] = 1
        ops['nimg_init'] = 500
        ops['nonrigid'] = False
        ops['smooth_sigma'] = .5
        ops['threshold_scaling'] = .7
        ops['batch_size'] = 250
        ops['do_registration'] = True

        #if roi is returned to, we can use previous roi
        if old_folder is not None:
            ops['roidetect'] = False
            ops['refImg'] = ops_old['refImg'];
            ops['force_refImg'] = True
        else:
            ops['roidetect'] = True
        ops['do_regmetrics'] = False
        ops['allow_overlap'] = False
        ops['save_folder'] = 'suite2p_' + savefolders[ei]  

        #run suit2p  with saved paremeter settings
        ops = suite2p.run_s2p(ops)  

        #extract traces if rois already there  
        if old_folder is not None:
            stat_new = copy.deepcopy(stat_old)
            
            #extracting ROI locations and traces
            from suite2p.extraction.masks import create_masks
            from suite2p.extraction.extract import extract_traces_from_masks
            
            #saving traces and rois
            cell_masks, neuropil_masks = create_masks(stat_new,ops['Ly'],ops['Lx'],ops)
            F, Fneu, F_chan2, Fneu_chan2 = extract_traces_from_masks(ops, cell_masks, neuropil_masks)
            
            #need to make dir first before you can save it with np.save apparently...
            try:
                os.makedirs(folder + ops['save_folder'] + r'/plane0/')
            except Exception:
                print('Overwriting exisitng directory')
            
            np.save(folder + ops['save_folder'] + r'/plane0/F.npy',F)
            np.save(folder + ops['save_folder'] + r'/plane0/Fneu.npy',Fneu) 
            np.save(folder + ops['save_folder'] + r'/plane0/stat.npy',stat_new)
            np.save(folder + ops['save_folder'] + r'/plane0/iscell.npy',iscell_old)
            np.save(folder + ops['save_folder'] + r'/plane0/ops.npy',ops)
            
            
            file = ops['tiff_list'][0]
        else:
            file = ops['tiff_list'][0]
        
        #grabbing trial metadata (for each tiff, 1 tiff =  1 trial)             
        siHeader = extract_scanimage_metadata.extract_scanimage_metadata(file)
        siBase = dict()
        for i in range(3):
            try:
                siBase[i] = bases[int(basesInd[i])]
            except:
                siBase[i] = ''
        siHeader['siBase']=siBase
        siHeader['savefolders'] = savefolders
        
        #saving trial metadat
        np.save(folder + 'suite2p_' + savefolders[ei] + r'/plane0/siHeader.npy',siHeader)


def boxoff():
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def generateSessionSummary(data, folder):
    '''
    must specify oldData if exists
    '''
    dt_si = data['dt_si']
    F = data['F']
    iscell = data['iscell']
    cn = data['conditioned_neuron'][0][0]
    f = np.nanmean(F,axis=2)
    for ci in range(np.shape(f)[1]):
        f[:,ci] = f[:,ci] - np.nanmean(f[0:20,ci])  
    reward_time_arrays = data['reward_time']
    reward_time_arrays_with_nans = [arr if len(arr) > 0 else np.array([np.nan]) for arr in reward_time_arrays]
    rew_time = np.concatenate(reward_time_arrays_with_nans)
    rew = np.isnan(rew_time)==0
    
    dt_si = data['dt_si']
    df = data['df_closedloop']
    t = np.arange(0,dt_si*df.shape[1] ,dt_si)
    bins = 10;
    rew_rate = np.convolve(rew, np.ones(bins)/bins, mode='full')
    rew_rate = rew_rate[bins-1:-bins+1]
    df_closedloop = data['df_closedloop']


    #Plot 1
    plt.subplot(234)
    if len(t) == len(df_closedloop[cn,:].T):
        plt.plot(t,df_closedloop[cn,:].T,'k',linewidth = .3)
    else:
        plt.plot(t[:-1],df_closedloop[cn,:].T,'k',linewidth = .3) 
    boxoff()
    plt.xlabel('Time (s)')

    plt.subplot(236)
    tsta = np.arange(0,dt_si*F.shape[0],dt_si)
    plt.plot(tsta,f[:,cn],'m')
    plt.xlabel('Time from trial start (s)')
    plt.ylabel('DF/F')
    boxoff()
    plt.title(data['mouse']+' ' + data['session'])

    plt.subplot(235)
    plt.imshow(np.squeeze(F[:,cn,:]).T,vmin=2,vmax=8)
    plt.xticks([40, 240], ['0', '10'])
    plt.xlabel('Time from trial start (s)')
    plt.ylabel('Trial #')
    boxoff()

    plt.subplot(231);
    plt.plot(rew_rate,'k')
    plt.ylim((-.1,1.1))
    plt.ylabel('Hit rate')
    plt.xlabel('Trial #')
    plt.title(data['mouse'] + '  ' +  data['session'])
    boxoff()

    plt.subplot(232);
    plt.plot(rew_time,'ko',markerfacecolor = 'w',markersize=3)
    plt.xlabel('Trial #')
    plt.ylabel('Time to reward (s)')
    plt.tight_layout()
    boxoff()


    #Save Combined Summary plot
    plt.savefig(os.path.join(folder, 'SessionSummary.png'))
    plt.close() 


def findConditionedNeurons(data, folder):
    '''
    Only need data dict and folder
    '''
    cn = data['conditioned_neuron'][0][0]
    f = data['F'];
    f = np.nanmean(f,axis = 2)
    N = f.shape[1]
    for i in range(N):
        bl = np.nanmean(f[0:19,i])
        f[:,i] = f[:,i] - bl    
    tune = np.mean(f,axis = 0)

    evts = np.zeros((N,))
    #Plot ROI Tunings
    for ci in range(N):
        df = data['df_closedloop'][ci,:]
        df = medfilt(df, 21)
        df = np.diff(df)    
        evts[ci] = len(np.where(df>.2)[0])
    plt.plot(tune,evts,'o',markerfacecolor = 'w')
    plt.savefig(os.path.join(folder, 'Tuning.png'))
    plt.close() 

    #Selecting ROIs based off criteria in np.where statement
    iscell = data['iscell']
    cns = np.where((np.abs(tune)<.2) & (iscell[:,0]==1) & (evts>400))[0] 

    #Plot 1
    for i in range(20):
        plt.subplot(4,5,i+1)
        plt.plot(f[:,cns[i]],'k',linewidth=.2)
        plt.axis('off')
        plt.title(str(cns[i]),fontsize=6)
    plt.savefig(os.path.join(folder, 'cns_AvgTrial.png'))
    plt.close() 

    #Plot2
    df = data['df_closedloop']    
    for i in range(20):
        plt.subplot(4,5,i+1)
        plt.plot(df[cns[i],:],'k',linewidth=.05)
        plt.axis('off')
        plt.title(str(cns[i]),fontsize=6)
    plt.savefig(os.path.join(folder, 'cns_EntireSession.png'))
    plt.close() 

    ops = np.load(data['dat_file']+'/ops.npy', allow_pickle=True).tolist()
    img = ops['meanImg']
    win = 10
    for i in range(20):
        x = np.round(data['centroidX'][cns[i]])
        y = np.round(data['centroidY'][cns[i]])
        x = int(x)
        y = int(y)
        plt.subplot(4,5,i+1)
        a = img[y-win:y+win,x-win:x+win]
        plt.axis('off')
        plt.imshow(a,vmin = 0,vmax=300,cmap='gray')
        plt.title(str(cns[i]),fontsize=6)

    plt.savefig(os.path.join(folder, 'cns_visual.png'))
    plt.close() 

    print(cns[0:19]+1) #adding +1 for matlab because of scanimage
    return cns[0:19]+1 #adding +1 for matlab because of scanimage