import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon
import numpy as np
# List to hold the coordinates of all polygons
polygons = []
ops = np.load('//allen/aind/scratch/BCI/2p-raw/BCINM_006/010524/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
img = ops['meanImg_chan2']

# Plot the image
fig, ax = plt.subplots()
ax.imshow(img,vmin=0,vmax=10)

polygon_selector = None

# Function to finalize the current polygon and start a new one
def finalize_polygon(event):
    global polygon_selector
    if polygon_selector:
        polygon_selector.disconnect_events()
        verts = polygon_selector.verts
        if len(verts) > 2:
            polygons.append(verts)
            print(f"Polygon finalized with vertices: {verts}")
            polygon = Polygon(verts, closed=True, edgecolor='red', facecolor='none', lw=2)
            ax.add_patch(polygon)
            fig.canvas.draw()
    if event is None or event.key == 'enter':
        polygon_selector = PolygonSelector(ax, lambda v: None)

# Bind the finalize function to the 'enter' key
fig.canvas.mpl_connect('key_press_event', finalize_polygon)

# Start the first polygon
finalize_polygon(None)

plt.show()


#%%
from PIL import Image
import numpy as np
import cv2

def get_pixels_inside_polygon(polygon, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    y, x = np.where(mask == 1)
    return list(zip(x, y))

def average_pixel_value(polygon, image):
    pixels = get_pixels_inside_polygon(polygon, image.shape)
    pixel_values = [image[y, x] for x, y in pixels]
    return np.mean(pixel_values)

num_tifs = len(ops['tiff_list'])
F = []  # List to hold average values arrays for each TIFF file

for fi in range(num_tifs):
    # Path to your TIFF file
    file_path = ops['data_path'][0] + ops['tiff_list'][fi]
    print(file_path)
    with Image.open(file_path) as img:
        # Create a list to hold each frame as a numpy array, reading every other frame
        frames = [np.array(img.seek(i) or img) for i in range(0, img.n_frames, 2)]  # Step is 2
    
    # Convert the list of frames to a 3D numpy array
    imgs = np.stack(frames, axis=2)# imgs is now a 3D array where imgs[:,:,i] is the i-th frame of the TIFF file
    
    T = imgs.shape[2]
    N = len(polygons)

    # Initialize an empty array
    average_values_per_image = np.zeros((T, N))
    for i in range(T):
        img = imgs[:, :, i]
        for j, polygon in enumerate(polygons):
            average_values_per_image[i, j] = average_pixel_value(polygon, img)
    F.append(average_values_per_image)
np.savez(folder + 'axonF.npz', *F)    
    #%%
    
sta = np.zeros((680,N,num_tifs))
for ti in range(num_tifs):
    for ci in range(N):
        g = F[ti][:,ci]
        g_padded = np.pad(g, (0, 640 - len(g)), mode='constant', constant_values=np.nan)
        if ti > 0:
            pre = F[ti-1][-41:-1,ci]
        else:
            pre = np.full((1, 40), np.nan)
        pre = pre.reshape(-1)
        c = np.concatenate((pre, g_padded))
        sta[:,ci,ti] = c
f = np.nanmean(sta,axis=2);
plt.subplot(211)
plt.plot(np.nanmean(f[0:180,:],axis=1))
plt.xlim([0, 179])  # Ensure x-axis limits are consistent with the first plot
plt.subplot(212)
plt.imshow(np.nanmean(sta[0:180, :, :], axis=1).T, vmin=5, vmax=11.5, aspect='auto')
plt.xlim([0, 179])  # Ensure x-axis limits are consisten with the first plot

plt.tight_layout()  # This will make sure the plots fit into the figure neatly
plt.show()
#%%
from matplotlib.patches import Polygon

# Assuming 'polygons' is your list of ROI polygons and 'img' is your image
# Example: polygons = [[(x1, y1), (x2, y2), ...], [(x1, y1), (x2, y2), ...], ...]

plt.imshow(img)

# Loop through the polygons and draw each one
for polygon_vertices in polygons:
    polygon = Polygon(polygon_vertices, closed=True, edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(polygon)

plt.show()
#%%
zaber = np.load(folder + folder[-7:-1]+r'-bpod_zaber.npy',allow_pickle=True).tolist()
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

trl_strt = zaber['trial_start_times'][files_with_movies]
trl_end = zaber['trial_end_times'][files_with_movies]
go_cue = zaber['go_cue_times'][files_with_movies]
trial_times = [(trl_end[i]-trl_strt[i]).total_seconds() for i in range(len(trl_strt))]
trial_start = [(trl_strt[i]).timestamp()-(trl_strt[0]).timestamp() for i in range(len(trl_strt))]
trial_hit = zaber['trial_hit'][files_with_movies]
lick_L = zaber['lick_L'][files_with_movies]
rewT = zaber['reward_L'][files_with_movies];
#rewT = zaber['threshold_crossing_times'][files_with_movies]
siTrig = zaber['Scanimage_trigger_times'][files_with_movies]
steps = zaber['zaber_move_forward'][files_with_movies];
valve = np.array(zaber['var_ValveOpenTime_L'][files_with_movies]);
nt = len(F)
N = sta.shape[1]
pre = 20;
post = 20;

rta = np.full((pre+post,N,nt), np.nan)
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
dt_si = 1/float(siHeader['frame_rate'])
for ti in range(nt):
    t = np.arange(0, dt_si * F[ti].shape[0], dt_si)
    if len((rewT[ti])) > 0:
        a = rewT[ti][0] - siTrig[ti][0]
        ind = np.where(t>a)[0][0]
    if pre < ind: 
        if len(F[ti]) > (ind+post):
            rta[:,:,ti] = F[ti][ind-pre:ind+post,:]
plt.plot(np.nanmean(np.nanmean(rta,axis=2),axis=1)) 
#%%
rt = np.nanmean(np.nanmean(rta,axis=0),axis=0)
plt.plot(medfilt(rt[0:100],5)-10)       
plt.plot(valve[0:100])
#%%

pre = 20;
post = 20;

vta = np.full((pre+post,N,nt), np.nan)
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
dt_si = 1/float(siHeader['frame_rate'])
for ti in range(nt):
    t = np.arange(0, dt_si * F[ti].shape[0], dt_si)
    if len((steps[ti])) > 0:
        a = steps[ti][0] - siTrig[ti][0]
        ind = np.where(t>a)[0][0]
    if pre < ind: 
        if len(F[ti]) > (ind+post):
            vta[:,:,ti] = F[ti][ind-pre:ind+post,:]
plt.plot(np.nanmean(np.nanmean(vta,axis=2),axis=1))    