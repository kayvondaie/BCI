folder = r'//allen/aind/scratch/BCI/2p-raw/BCI109/030525/pophys/'
data = np.load(folder + 'data_main.npy',allow_pickle=True)
#%%
import scipy.io
import os
import re
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
stem = siHeader['siBase'][0]

pattern = re.compile(rf'^{re.escape(stem)}_threshold_(\d+)\.mat$')

max_file = None
max_num = -1

for fname in os.listdir(folder):
    match = pattern.match(fname)
    if match:
        num = int(match.group(1))
        if num > max_num:
            max_num = num
            max_file = fname

#%%
filepath = os.path.join(folder, max_file)
bpod = scipy.io.loadmat(filepath)
abc = bpod['abcdef'].squeeze()

# Access fields using field names, like a dict or structured array
bpod_info = abc['bpod_info'].item()[0]
cn_trace = abc['cn_trace'].item()[0]

plt.plot(data['roi_csv'][0:47000,data['cn_csv_index'][0]+2])
plt.plot(cn_trace[0:47000],linewidth=.5)

plt.title(folder)

