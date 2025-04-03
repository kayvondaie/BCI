def run():
    # Cell 105
# mypath = '/data/bci_oct24_upload/'
BEHAV_DATA_PATH = '/data/bci_data/'

behav, data, maps = get_behav_and_data_maps(BEHAV_DATA_PATH, verbose=False)
session_idx_to_behav_idx = maps['session_idx_to_behav_idx']
session_idx_to_data_idx = maps['session_idx_to_data_idx']
print('Done!')

    print("✅ load_behav_data ran successfully.")
if __name__ == '__main__':
    run()