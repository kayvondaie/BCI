# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:10:45 2023

@author: scanimage
"""
import numpy as np
import os
import re
import extract_scanimage_metadata
import scipy.io
import pandas as pd
import glob



def main(folder):
    
    data = dict()
    
    # BCI data
    if os.path.isdir(folder +r'/suite2p_BCI/'):
        stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)        
        Ftrace = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
        ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
        iscell = np.load(folder + r'/suite2p_BCI/plane0/iscell.npy', allow_pickle=True)
        siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
    elif os.path.isdir(folder +r'/suite2p_BCI_green/'):
        stat = np.load(folder + r'/suite2p_BCI_green/plane0/stat.npy', allow_pickle=True)        
        Ftrace = np.load(folder +r'/suite2p_BCI_green/plane0/F.npy', allow_pickle=True)
        ops = np.load(folder + r'/suite2p_BCI_green/plane0/ops.npy', allow_pickle=True).tolist()
        iscell = np.load(folder + r'/suite2p_BCI_green/plane0/iscell.npy', allow_pickle=True)
        siHeader = np.load(folder + r'/suite2p_BCI_green/plane0/siHeader.npy', allow_pickle=True).tolist()
    
    data['dat_file'] = folder + r'/suite2p_BCI/plane0/'
    slash_indices = [match.start() for match in re.finditer('/', folder)]
    data['session'] = folder[slash_indices[-2]+1:slash_indices[-1]]
    data['mouse'] = folder[slash_indices[-3]+1:slash_indices[-2]]
    if os.path.isdir(folder +r'/suite2p_BCI/'):
        siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist() 
        dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanVolumeRate']);
        if dt_si < 0.05:
            post = round(10/0.05 * 0.05/dt_si)
            pre = round(2/0.05 * 0.05/dt_si)
        else:
            post = round(10/0.05)
            pre = round(2/0.05)
        data['trace_corr'] = np.corrcoef(Ftrace.T, rowvar=False)
        data['iscell'] = iscell;
        # metadata        
        
        # create F and Fraw
        data['F'], data['Fraw'],data['df_closedloop'],data['centroidX'],data['centroidY'] = create_BCI_F(Ftrace,ops,stat,pre,post);     
        
        # create dist, conditioned_neuron, conditioned_coordinates
        data['dist'], data['conditioned_neuron_coordinates'], data['conditioned_neuron'], data['cn_csv_index'] = find_conditioned_neurons(siHeader,stat)
        data['dt_si'] = 1/float(siHeader['metadata']['hRoiManager']['scanFrameRate'])
        
    
    # photostim data
    if os.path.isdir(folder +r'/suite2p_photostim/'):
        stat = np.load(folder + r'/suite2p_photostim/plane0/stat.npy', allow_pickle=True)#note that this is only defined in the BCI folder
        Ftrace = np.load(folder +r'/suite2p_photostim/plane0/F.npy', allow_pickle=True)
        ops = np.load(folder + r'/suite2p_photostim/plane0/ops.npy', allow_pickle=True).tolist()
        siHeader = np.load(folder + r'/suite2p_photostim/plane0/siHeader.npy', allow_pickle=True).tolist()    
        data['photostim'] = dict()
        data['photostim']['Fstim'], data['photostim']['seq'], data['photostim']['favg'], data['photostim']['stimDist'], data['photostim']['stimPosition'], data['photostim']['centroidX'], data['photostim']['centroidY'], data['photostim']['slmDist'],data['photostim']['stimID'],data['photostim']['Fstim_raw'],data['photostim']['favg_raw'] = create_photostim_Fstim(ops, Ftrace,siHeader,stat)
        data['photostim']['FstimRaw'] = Ftrace
    if os.path.isdir(folder +r'/suite2p_photostim2/'):
        stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)#note that this is only defined in the BCI folder
        Ftrace = np.load(folder +r'/suite2p_photostim2/plane0/F.npy', allow_pickle=True)
        ops = np.load(folder + r'/suite2p_photostim2/plane0/ops.npy', allow_pickle=True).tolist()
        siHeader = np.load(folder + r'/suite2p_photostim2/plane0/siHeader.npy', allow_pickle=True).tolist()    
        data['photostim2'] = dict()
        data['photostim2']['Fstim'], data['photostim2']['seq'], data['photostim2']['favg'], data['photostim2']['stimDist'], data['photostim2']['stimPosition'], data['photostim2']['centroidX'], data['photostim2']['centroidY'], data['photostim2']['slmDist'],data['photostim2']['stimID'] = create_photostim_Fstim(ops, Ftrace,siHeader,stat)
        data['photostim2']['FstimRaw'] = Ftrace
    
    if os.path.isdir(folder +r'/suite2p_photostim_single/'):
        stat = np.load(folder + r'/suite2p_photostim_single/plane0/stat.npy', allow_pickle=True)#note that this is only defined in the BCI folder
        Ftrace = np.load(folder +r'/suite2p_photostim_single/plane0/F.npy', allow_pickle=True)
        ops = np.load(folder + r'/suite2p_photostim_single/plane0/ops.npy', allow_pickle=True).tolist()
        siHeader = np.load(folder + r'/suite2p_photostim_single/plane0/siHeader.npy', allow_pickle=True).tolist()    
        data['photostim'] = dict()
        data['photostim']['Fstim'], data['photostim']['seq'], data['photostim']['favg'], data['photostim']['stimDist'], data['photostim']['stimPosition'], data['photostim']['centroidX'], data['photostim']['centroidY'], data['photostim']['slmDist'],data['photostim']['stimID'],data['photostim']['Fstim_raw'],data['photostim']['favg_raw'] = stimDist_single_cell(ops, Ftrace,siHeader,stat)
        data['photostim']['FstimRaw'] = Ftrace        
    # spont data
    if os.path.isdir(folder +r'/suite2p_spont/'):
        data['spont'] = np.load(folder +r'/suite2p_spont/plane0/F.npy', allow_pickle=True)
    
    #behavioral data
    behav_folder = 'I:/My Drive/Learning rules/BCI_data/behavior//' + 'BCI_' + data['mouse'][3:]
    behav_file = behav_folder + '/' + data['session'] + r'-bpod_zaber.npy';
    if os.path.isfile(folder + folder[-7:-1]+r'-bpod_zaber.npy') or os.path.isfile(behav_file) or os.path.isfile(folder + r'/behavior/' + folder[-7:-1]+r'-bpod_zaber.npy'):
        import folder_props_fun        
        siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
        ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
        dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanFrameRate'])
        if isinstance(siHeader['siBase'], str):
            base = siHeader['siBase']
        else:
            base = siHeader['siBase'][0]
        if os.path.isfile(folder + folder[-7:-1]+r'-bpod_zaber.npy'):
            data['reward_time'], data['step_time'], data['trial_start'], data['SI_start_times'],data['threshold_crossing_time'] = create_zaber_info(folder,base,ops,dt_si)
        elif os.path.isfile(behav_file):
            data['reward_time'], data['step_time'], data['trial_start'], data['SI_start_times'],data['threshold_crossing_time'] = create_zaber_info(behav_file,base,ops,dt_si)
        elif os.path.isfile(folder + r'/behavior/' + folder[-7:-1]+r'-bpod_zaber.npy'):
            behav_file = folder + r'/behavior/' + folder[-7:-1]+r'-bpod_zaber.npy'
            data['reward_time'], data['step_time'], data['trial_start'], data['SI_start_times'],data['threshold_crossing_time'] = create_zaber_info(behav_file,base,ops,dt_si)
 
 
    

    numtrl = data['F'].shape[2]
    BCI_thresholds = np.full((2, numtrl), np.nan)
    siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()

    # Determine the base for file names
    if isinstance(siHeader['siBase'], str):
        base = siHeader['siBase']
    else:
        base = siHeader['siBase'][0]

    # Iterate over trials and attempt to load the corresponding threshold files
    for i in range(numtrl):
        try:
            st = folder + base + r'_threshold_' + str(i+1) + r'.mat'
            
            # Check if the file exists before trying to load it
            if os.path.exists(st):
                threshold_data = scipy.io.loadmat(st)
                BCI_thresholds[:, i] = threshold_data['BCI_threshold'].flatten()
                
        except:
            pass  # Ignore any exceptions and continue with the next iteration
    data['BCI_thresholds'] = BCI_thresholds
    
    csv_files = glob.glob(os.path.join(folder, base+'_IntegrationRois' + '_*.csv'))
    csv_files = sorted(csv_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    csv_data = []
    for i in range(len(csv_files)):
        csv_file = csv_files[i]
        csv_data.append(pd.read_csv(csv_file))        
    data['roi_csv'] = np.concatenate(csv_data)

    np.save(folder + r'data_'+data['mouse']+r'_'+data['session']+r'.npy',data)
    #np.save(folder + r'data_'+data['mouse']+r'_'+data['session']+r'_'+str(int(np.round(np.random.rand()*100000)))+r'.npy',data)
    return data

def create_BCI_F(Ftrace,ops,stat,pre_i,post_i):
    F_trial_strt = [];
    Fraw_trial_strt = [];
    
    strt = 0;
    dff = 0*Ftrace
    for i in range(np.shape(Ftrace)[0]):
        bl = np.std(Ftrace[i,:])
        dff[i,:] = (Ftrace[i,:] - bl)/bl
    for i in range(len(ops['frames_per_file'])):
        ind = list(range(strt,strt+ops['frames_per_file'][i]))    
        f = dff[:,ind]
        F_trial_strt.append(f)
        f = Ftrace[:,ind]
        Fraw_trial_strt.append(f)
        strt = ind[-1]+1
        

    F = np.full((pre_i+post_i,np.shape(Ftrace)[0],len(ops['frames_per_file'])),np.nan)
    Fraw = np.full((pre_i+post_i,np.shape(Ftrace)[0],len(ops['frames_per_file'])),np.nan)
    pre = np.full((np.shape(Ftrace)[0],pre_i),np.nan)
    for i in range(len(ops['frames_per_file'])):
        f = F_trial_strt[i]
        fraw = Fraw_trial_strt[i]
        if i > 0:
            pre = F_trial_strt[i-1][:,-pre_i:]
        pad = np.full((np.shape(Ftrace)[0],post_i),np.nan)
        f = np.concatenate((pre,f),axis = 1)
        f = np.concatenate((f,pad),axis = 1)
        f = f[:,0:pre_i+post_i]
        F[:,:,i] = np.transpose(f)
        
        fraw = np.concatenate((pre,fraw),axis = 1)
        fraw = np.concatenate((fraw,pad),axis = 1)
        fraw = fraw[:,0:pre_i+post_i]
        Fraw[:,:,i] = np.transpose(fraw)
        
        centroidX = []
        centroidY = []
        dist = []
        for i in range(len(stat)):
            centroidX.append(np.mean(stat[i]['xpix']))
            centroidY.append(np.mean(stat[i]['ypix']))
        
    return F, Fraw, dff,centroidX, centroidY

def find_conditioned_neurons(siHeader,stat):
    
    cnName = siHeader['metadata']['hIntegrationRoiManager']['outputChannelsRoiNames']
    g = [i for i in range(len(cnName)) 
         if cnName.startswith("'",i)]
    cnName = cnName[g[0]+1:g[1]]

    rois = siHeader['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois']
    if isinstance(rois, dict):
        rois = [rois]
    a = []
    for i in range(len(rois)):
        name = rois[i]['name']
        a.append(cnName == name)
        
    indices = [i for i, x in enumerate(a) if x]
    cnPos = rois[indices[0]]['scanfields']['centerXY'];

    deg = siHeader['metadata']['hRoiManager']['imagingFovDeg']
    g = [i for i in range(len(deg)) if deg.startswith(" ",i)]
    gg = [i for i in range(len(deg)) if deg.startswith(";",i)]
    for i in gg:
        g.append(i)
    g = np.sort(g)
    num = [];
    for i in range(len(g)-1):
        num.append(float(deg[g[i]+1:g[i+1]]))
    dim = int(siHeader['metadata']['hRoiManager']['linesPerFrame']),int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
    degRange = np.max(num) - np.min(num)
    pixPerDeg = dim[0]/degRange

    cnPosPix = np.array(np.array(cnPos)-num[0])*pixPerDeg

    centroidX = []
    centroidY = []
    dist = []
    for i in range(len(stat)):
        centroidX.append(np.mean(stat[i]['xpix']))
        centroidY.append(np.mean(stat[i]['ypix']))
        dx = centroidX[i] - cnPosPix[0]
        dy = centroidY[i] - cnPosPix[1]
        d = np.sqrt(dx**2+dy**2)
        dist.append(d)
    dist = np.asarray(dist)
    conditioned_neuron_coordinates = cnPosPix
    conditioned_neuron = np.where(dist<10)
    
    return dist, conditioned_neuron_coordinates, conditioned_neuron, indices

def create_photostim_Fstim(ops,F,siHeader,stat):
    numTrl = len(ops['frames_per_file']);
    timepts = 45;
    numCls = F.shape[0]
    Fstim = np.full((timepts,numCls,numTrl),np.nan)
    Fstim_raw = np.full((timepts,numCls,numTrl),np.nan)
    strt = 0;
    dff = 0*F
    pre = 5;
    post = 20
    
    photostim_groups = siHeader['metadata']['json']['RoiGroups']['photostimRoiGroups']
    seq = siHeader['metadata']['hPhotostim']['sequenceSelectedStimuli'];
    list_nums = seq.strip('[]').split();
    seq = [int(num) for num in list_nums]
    seq = seq*10
    seqPos = int(siHeader['metadata']['hPhotostim']['sequencePosition'])-1;
    seq = seq[seqPos:Fstim.shape[2]]
    seq = np.asarray(seq)
    
    stimID = np.zeros((F.shape[1],))
    for ti in range(numTrl):
        pre_pad = np.arange(strt-5,strt)
        ind = list(range(strt,strt+ops['frames_per_file'][ti]))
        strt = ind[-1]+1
        post_pad = np.arange(ind[-1]+1,ind[-1]+20)
        ind = np.concatenate((pre_pad,np.asarray(ind)),axis=0)
        ind = np.concatenate((ind,post_pad),axis = 0)
        ind[ind > F.shape[1]-1] = F.shape[1]-1;
        ind[ind < 0] = 0
        stimID[ind[pre+1]] = seq[ti]
        a = F[:,ind].T
        g = F[:,ind].T
        bl = np.tile(np.mean(a[0:pre,:],axis = 0),(a.shape[0],1))
        a = (a-bl) / bl
        if a.shape[0]>Fstim.shape[0]:
            a = a[0:Fstim.shape[0],:]
        Fstim[0:a.shape[0],:,ti] = a
        try:
            g = g[0:Fstim.shape[0],:]
            Fstim_raw[0:g.shape[0],:,ti] = g
        except ValueError as e:
            print(f"Skipping trial {ti} due to shape mismatch: {e}")
   
    
   
    deg = siHeader['metadata']['hRoiManager']['imagingFovDeg']
    g = [i for i in range(len(deg)) if deg.startswith(" ",i)]
    gg = [i for i in range(len(deg)) if deg.startswith(";",i)]
    for i in gg:
        g.append(i)
    g = np.sort(g)
    num = [];
    for i in range(len(g)-1):
        num.append(float(deg[g[i]+1:g[i+1]]))
    dim = int(siHeader['metadata']['hRoiManager']['linesPerFrame']),int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
    degRange = np.max(num) - np.min(num)
    pixPerDeg = dim[0]/degRange

    centroidX = []
    centroidY = []
    for i in range(len(stat)):
        centroidX.append(np.mean(stat[i]['xpix']))
        centroidY.append(np.mean(stat[i]['ypix']))

    favg = np.zeros((Fstim.shape[0],Fstim.shape[1],len(photostim_groups)))
    favg_raw = np.zeros((Fstim.shape[0],Fstim.shape[1],len(photostim_groups)))
    stimDist = np.zeros([Fstim.shape[1],len(photostim_groups)])
    slmDist = np.zeros([Fstim.shape[1],len(photostim_groups)])
    
    coordinates = photostim_groups[0]['rois'][1]['scanfields']['slmPattern']
    coordinates = np.asarray(coordinates)
    # if coordinates.shape[0] == 0:
    #     coordinates = np.array([[0, 0, 0, 0]])
    if np.ndim(coordinates) == 1:
        coordinates = coordinates.reshape(1,-1)
    xy = coordinates[:,:2] + photostim_groups[0]['rois'][1]['scanfields']['centerXY']
    stimPos = np.zeros(np.shape(xy))
    stimPosition = np.zeros([stimPos.shape[0],stimPos.shape[1],len(photostim_groups)])
    
    for gi in range(len(photostim_groups)):        
        coordinates = photostim_groups[gi]['rois'][1]['scanfields']['slmPattern']
        if np.ndim(coordinates) == 1:
            coordinates = coordinates.reshape(1,-1)
        galvo = photostim_groups[gi]['rois'][1]['scanfields']['centerXY']
        # if coordinates['_ArraySize_'][0] == 0:
        #     coordinates = np.array([[0, 0, 0, 0]])
        #     coordinates = np.asarray(coordinates)
        #     if np.ndim(coordinates) == 1:
        #         coordinates = coordinates.reshape(1,-1)
        # else:
        coordinates = np.asarray(coordinates)

        xy = coordinates[:,:2] + galvo
        xygalvo = coordinates[:,:2]*0 + galvo
        stimPos = np.zeros(np.shape(xy))
        galvoPos = np.zeros(np.shape(xy))
        for i in range(np.shape(xy)[0]):
            stimPos[i,:] = np.array(xy[i,:]-num[0])*pixPerDeg
            galvoPos[i,:] = np.array(xygalvo[i,:]-num[0])*pixPerDeg
        sd = np.zeros([np.shape(xy)[0],favg.shape[1]])        
        for i in range(np.shape(xy)[0]):
            for j in range(favg.shape[1]):
                sd[i,j] = np.sqrt(sum((stimPos[i,:] - np.asarray([centroidX[j], centroidY[j]]))**2))
                slmDist[j,gi] = np.sqrt(sum((galvoPos[i,:] - np.asarray([centroidX[j], centroidY[j]]))**2))                
        stimDist[:,gi] = np.min(sd,axis=0)
        ind = np.where(seq == gi+1)[0]
        favg[:,:,gi] = np.nanmean(Fstim[:,:,ind],axis = 2)
        favg_raw[:,:,gi] = np.nanmean(Fstim_raw[:,:,ind],axis = 2)
        stimPosition[:,:,gi] = stimPos

       
    return Fstim, seq, favg, stimDist, stimPosition, centroidX, centroidY, slmDist, stimID, Fstim_raw, favg_raw

def create_zaber_info(folder,base,ops,dt_si):
    import pandas as pd
    try:
        zaber = np.load(folder + folder[-7:-1]+r'-bpod_zaber.npy',allow_pickle=True).tolist()
    except:        
        zaber = np.load(folder,allow_pickle=True).tolist()    
    #zaber = np.load(folder[:-1]+r'-bpod_zaber.npy',allow_pickle=True).tolist()
    good = np.zeros((1,len(zaber['scanimage_file_names'])))[0]
    
    files_with_movies = []
    for zi in range(len(zaber['scanimage_file_names'])):
        name = str(zaber['scanimage_file_names'][zi])
        b = name.count('_')
        if b > 0:
            a = max([i for i, c in enumerate(name) if c == '_'])
            siBase = name[2:a]
            if siBase == base:
                files_with_movies.append(True)
            else:
                files_with_movies.append(False)
        else:
            files_with_movies.append(False)
    
# =============================================================================
#     files_with_movies = []
#     for k in zaber['scanimage_file_names']:
#         if str(k) == 'no movie for this trial':
#             files_with_movies.append(False)
#         else:
#             files_with_movies.append(True)
# =============================================================================

    trl_strt = zaber['trial_start_times'][files_with_movies]
    trl_end = zaber['trial_end_times'][files_with_movies]
    go_cue = zaber['go_cue_times'][files_with_movies]
    trial_times = [(trl_end[i]-trl_strt[i]).total_seconds() for i in range(len(trl_strt))]
    trial_start = [(trl_strt[i]).timestamp()-(trl_strt[0]).timestamp() for i in range(len(trl_strt))]
    trial_hit = zaber['trial_hit'][files_with_movies]
    lick_L = zaber['lick_L'][files_with_movies]
    rewT = zaber['reward_L'];
    threshold_crossing_times = zaber['threshold_crossing_times'][files_with_movies]
    trial_times = np.array(trial_times)
    L = len(trial_times)
    
    trial_times = ops['frames_per_file']*dt_si
    trial_times = trial_times[0:L]
    tt = np.cumsum(trial_times)
    tt = np.insert(tt,0,0)
    steps = zaber['zaber_move_forward'];
    rewT_abs = np.zeros(len(tt))
    steps_abs = []
    for i in range(len(tt)-1):
        if rewT[i]:
            rewT_abs[i] = rewT[i][0] + tt[i]
        a = steps[i] + tt[i] + zaber['scanimage_first_frame_offset'][i]
        steps_abs.append(a)    
    #steps_abs = np.concatenate(steps_abs)
    #rewT_abs = rewT_abs[rewT_abs!=0]
    #trial_start = np.asarray(trial_start)
    SI_start_times = zaber['Scanimage_trigger_times']
    return rewT[files_with_movies], steps[files_with_movies], trial_start, SI_start_times[files_with_movies],threshold_crossing_times

def load_data_dict(folder):
    data = dict()
    slash_indices = [match.start() for match in re.finditer('/', folder)]
    data['session'] = folder[slash_indices[-2]+1:slash_indices[-1]]
    data['mouse'] = folder[slash_indices[-3]+1:slash_indices[-2]];
    data = np.load(folder + r'data_'+data['mouse']+r'_'+data['session']+r'.npy',allow_pickle=True).tolist()
    return data

def read_stim_file(folder,subfolder):
    import numpy as np
    ops = np.load(folder+subfolder+r'plane0/ops.npy', allow_pickle=True).tolist()
    # Read data from file
    filename = folder + ops['tiff_list'][0][0:-4] + r'.stim'
    hFile = open(filename, 'rb')  # Use 'rb' for reading binary file
    phtstimdata = np.fromfile(hFile, dtype=np.float32)
    hFile.close()

    # Sanity check for file size
    datarecordsize = 3
    lgth = len(phtstimdata)
    if lgth % datarecordsize != 0:
        print('Unexpected size of photostim log file')
        lgth = (lgth // datarecordsize) * datarecordsize
        phtstimdata = phtstimdata[:lgth]

    # Reshape the data
    phtstimdata = np.reshape(phtstimdata, (lgth // datarecordsize, datarecordsize))

    # Extract x, y, and beam power
    out = {}
    out['X'] = phtstimdata[:, 0]
    out['Y'] = phtstimdata[:, 1]
    out['Beam'] = phtstimdata[:, 2]
    return out

def siHeader_get(folder):
    ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
    file = folder + ops['tiff_list'][0]                    
    siHeader = extract_scanimage_metadata.extract_scanimage_metadata(file)
 
    return siHeader

def stimDist_single_cell(ops,F,siHeader,stat):
    
    trip = np.std(F,axis=0)
    trip = np.where(trip<10)[0]

    extended_trip = np.concatenate((trip, trip + 1))
    trip = np.unique(extended_trip)
    trip[trip>F.shape[1]-1] = F.shape[1]-1

    F[:,trip] = np.nan
    numTrl = len(ops['frames_per_file']);
    timepts = 55*round(float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])/16);
    numCls = F.shape[0]
    Fstim = np.full((timepts,numCls,numTrl),np.nan)
    Fstim_raw = np.full((timepts,numCls,numTrl),np.nan)
    strt = 0;
    dff = 0*F
    pre = 5*round(float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])/16);
    post = 20*round(float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])/16)
    
    photostim_groups = siHeader['metadata']['json']['RoiGroups']['photostimRoiGroups']
    seq = siHeader['metadata']['hPhotostim']['sequenceSelectedStimuli'];
    list_nums = seq.strip('[]').split();
    seq = [int(num) for num in list_nums]
    seq = seq*40
    seqPos = int(siHeader['metadata']['hPhotostim']['sequencePosition'])-1;
    seq = seq[seqPos:]
    seq = np.asarray(seq)
    
    stimID = np.zeros((F.shape[1],))
    for ti in range(numTrl):
        pre_pad = np.arange(strt-pre,strt)
        ind = list(range(strt,strt+ops['frames_per_file'][ti]))
        strt = ind[-1]+1
        post_pad = np.arange(ind[-1]+1,ind[-1]+post)
        ind = np.concatenate((pre_pad,np.asarray(ind)),axis=0)
        ind = np.concatenate((ind,post_pad),axis = 0)
        ind[ind > F.shape[1]-1] = F.shape[1]-1;
        ind[ind < 0] = 0
        stimID[ind[pre+1]] = seq[ti]
        a = F[:,ind].T
        g = F[:,ind].T
        bl = np.tile(np.mean(a[0:pre,:],axis = 0),(a.shape[0],1))
        a = (a-bl) / bl
        if a.shape[0]>Fstim.shape[0]:
            a = a[0:Fstim.shape[0],:]
        Fstim[0:a.shape[0],:,ti] = a
        try:
            Fstim_raw[0:a.shape[0],:,ti] = g
        except ValueError as e:
            print(f"Skipping trial {ti} due to shape mismatch: {e}")
                
   
    deg = siHeader['metadata']['hRoiManager']['imagingFovDeg']
    g = [i for i in range(len(deg)) if deg.startswith(" ",i)]
    gg = [i for i in range(len(deg)) if deg.startswith(";",i)]
    for i in gg:
        g.append(i)
    g = np.sort(g)
    num = [];
    for i in range(len(g)-1):
        num.append(float(deg[g[i]+1:g[i+1]]))
    dim = int(siHeader['metadata']['hRoiManager']['linesPerFrame']),int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
    degRange = np.max(num) - np.min(num)
    pixPerDeg = dim[0]/degRange

    centroidX = []
    centroidY = []
    for i in range(len(stat)):
        centroidX.append(np.mean(stat[i]['xpix']))
        centroidY.append(np.mean(stat[i]['ypix']))

    favg = np.zeros((Fstim.shape[0],Fstim.shape[1],len(photostim_groups)))
    favg_raw = np.zeros((Fstim.shape[0],Fstim.shape[1],len(photostim_groups)))
    stimDist = np.zeros([Fstim.shape[1],len(photostim_groups)])
    slmDist = np.zeros([Fstim.shape[1],len(photostim_groups)])
    
    coordinates = photostim_groups[0]['rois'][1]['scanfields']['slmPattern']
    coordinates = np.array([[0, 0, 0, 0]])
    coordinates = np.asarray(coordinates)
    if np.ndim(coordinates) == 1:
        coordinates = coordinates.reshape(1,-1)
    xy = coordinates[:,:2] + photostim_groups[0]['rois'][1]['scanfields']['centerXY']
    stimPos = np.zeros(np.shape(xy))
    stimPosition = np.zeros([stimPos.shape[0],stimPos.shape[1],len(photostim_groups)])
    
    seq = seq[0:Fstim.shape[2]]
    for gi in range(len(photostim_groups)):        
        coordinates = photostim_groups[gi]['rois'][1]['scanfields']['slmPattern']
        if np.ndim(coordinates) == 1:
            coordinates = np.asarray(coordinates)
            coordinates = coordinates.reshape(1,-1)
        galvo = photostim_groups[gi]['rois'][1]['scanfields']['centerXY']
        
        coordinates = np.asarray(coordinates)

        xy = coordinates[:,:2] + galvo
        xygalvo = coordinates[:,:2]*0 + galvo
        stimPos = np.zeros(np.shape(xy))
        galvoPos = np.zeros(np.shape(xy))
        for i in range(np.shape(xy)[0]):
            stimPos[i,:] = np.array(xy[i,:]-num[0])*pixPerDeg
            galvoPos[i,:] = np.array(xygalvo[i,:]-num[0])*pixPerDeg
        sd = np.zeros([np.shape(xy)[0],favg.shape[1]])        
        for i in range(np.shape(xy)[0]):
            for j in range(favg.shape[1]):
                sd[i,j] = np.sqrt(sum((stimPos[i,:] - np.asarray([centroidX[j], centroidY[j]]))**2))
                slmDist[j,gi] = np.sqrt(sum((galvoPos[i,:] - np.asarray([centroidX[j], centroidY[j]]))**2))                
        stimDist[:,gi] = np.min(sd,axis=0)
        ind = np.where(seq == gi+1)[0]
        favg[:,:,gi] = np.nanmean(Fstim[:,:,ind],axis = 2)
        favg_raw[:,:,gi] = np.nanmean(Fstim_raw[:,:,ind],axis = 2)
        stimPosition[:,:,gi] = stimPos

    
    return Fstim, seq, favg, stimDist, stimPosition, centroidX, centroidY, slmDist, stimID, Fstim_raw, favg_raw