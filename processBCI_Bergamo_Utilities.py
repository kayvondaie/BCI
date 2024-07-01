import suite2p
import os, re, copy, shutil, sys
from suite2p import default_ops as s2p_default_ops
from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
import folder_props_fun, extract_scanimage_metadata
#import registration_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
import tkinter as tk
from tkinter import simpledialog
import plotting_functions as pf
from scipy.signal import medfilt


def loadSuite2pROIS(folder, oldFolder = None):
    '''
    Must specify old data if exists
    '''

    #If old exists
    if oldFolder is not None:
        stat_old = np.load(old_folder + r'suite2p_BCI/' + r'/plane0/stat.npy',allow_pickle = 'True')
        ops_old = np.load(old_folder + r'suite2p_BCI/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
        iscell_old = np.load(old_folder +r'suite2p_BCI/' + r'/plane0/iscell.npy',allow_pickle = 'True')
    else:
        stat_old = None
        ops_old = None
        iscell_old = None
    
    savefolders = dict()
    savefolders[1] = 'spont'
    savefolders[0] = 'BCI' 
    ops = s2p_default_ops()

    #SETTING PARAMETERS FOR SUITE2P
    ops['data_path'] = folder
    folder = ops['data_path'][0]
    folder_props = folder_props_fun.folder_props_fun(folder)
    bases = folder_props['bases']
    
    #QUICK GUI FOR BASE SELECTION
    root = tk.Tk()
    root.withdraw() 
    ind = simpledialog.askstring("Bases", "Choose your BCI and spont bases:")
    if ind is None:
        print("Aborting: No Bases Selected")
        sys.exit()
    else:
        ind = np.fromstring(ind[1:-1], sep=',')
        print("User input:", ind)
    
    #Initiate Suite2p with specified settings
    #############################################################################################################
    for ei in range(0,len(ind)):
        if ei == 1: 
            old_folder = folder
        if 'old_folder' in locals():
            stat_old = np.load(old_folder + r'suite2p_BCI/' + r'/plane0/stat.npy',allow_pickle = 'True')
            ops_old = np.load(old_folder + r'suite2p_BCI/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
            iscell_old = np.load(old_folder +r'suite2p_BCI/' + r'/plane0/iscell.npy',allow_pickle = 'True')

        #base = {base for i, base in enumerate(bases) if str(i) in ind}
        base = bases[int(ind[ei])]
        siFiles = folder_props['siFiles'] #not sure what this is doing... siFiles does not get called after this...
        files = os.listdir(folder)
        good = np.zeros([1,np.shape(files)[0]])
        
        #Grabbing tiffs with the epoch name contained in their name
        for fi in range(0,np.shape(files)[0]):
            str = files[fi]
            a = str.find('.tif')
            if a > -1:
                b = max([i for i, char in enumerate(str) if char == '_'])
                b = str[0:b]
                if b == base:
                    good[0][fi] = 1
        
        good = np.where(good == 1)
        good = good[1]
        
        #only storing the good trials
        ops['tiff_list'] = [files[i] for i in good]
        
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
        if 'old_folder' in locals():
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
        if 'old_folder' in locals():
            stat_new = copy.deepcopy(stat_old)
            
            #extracting ROI locations and traces
            from suite2p.extraction.masks import create_masks
            from suite2p.extraction.extract import extract_traces_from_masks
            
            #saving traces and rois
            cell_masks, neuropil_masks = create_masks(stat_new,ops['Ly'],ops['Lx'],ops)
            F, Fneu, F_chan2, Fneu_chan2 = extract_traces_from_masks(ops, cell_masks, neuropil_masks)
            np.save(folder + ops['save_folder'] + r'/plane0/F.npy',F)
            np.save(folder + ops['save_folder'] + r'/plane0/Fneu.npy',Fneu) 
            np.save(folder + ops['save_folder'] + r'/plane0/stat.npy',stat_new)
            np.save(folder + ops['save_folder'] + r'/plane0/iscell.npy',iscell_old)
            
            folder = ops['data_path'][0]
            file = folder + ops['tiff_list'][0]                
            shutil.copy(ops['save_path0'] +r'/suite2p/plane0/data.bin',ops['save_path0']+ops['save_folder'] + r'/plane0/data.bin')
        file = folder + '\\' + ops['tiff_list'][0]   
        
        #grabbing trial metadata (for each tiff, 1 tiff =  1 trial)             
        siHeader = extract_scanimage_metadata.extract_scanimage_metadata(file)
        siBase = dict()
        for i in range(3):
            try:
                siBase[i] = bases[int(ind[i])]
            except:
                siBase[i] = ''
        siHeader['siBase']=siBase
        siHeader['savefolders'] = savefolders
        
        #saving trial metadat
        np.save(folder + 'suite2p_' + savefolders[ei] + r'/plane0/siHeader.npy',siHeader)

    try:
        print('Works:', folder +r'/suite2p_BCI/')
    except Exception as e:
        print('Malfunction', e)

def boxoff():
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def generateSessionSummary(data, folder, oldData = None):
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