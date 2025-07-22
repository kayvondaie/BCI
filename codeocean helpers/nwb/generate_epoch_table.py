import pandas as pd 

def generate_epoch_table(nwbfile): 
    """
    Finds frame and time ranges for photostim, photostim_post, behavior, spontaneous, and spontaneous_post epochs from the photostim and behavior tables in the NWB files 
    
    Parameters:
    - nwbfile: NWBFile object

    Returns:
    - epoch_table: pandas.DataFrame of epoch frame and time ranges 
    """

    dff = nwbfile.processing["processed"].data_interfaces["dff"].roi_response_series["dff"].data
    frame_rate = nwbfile.imaging_planes["processed"].imaging_rate
    
    photostim_table = nwbfile.stimulus["PhotostimTrials"].to_dataframe()
    behavior_table = nwbfile.trials.to_dataframe()
    behavior_table['stim_name'] = 'BCI'

    # Filter tables for common columns 
    shared_cols = ['start_frame', 'stop_frame', 'stim_name'] 
    filter_photostim_table = photostim_table[shared_cols]
    filter_behavior_table = behavior_table[shared_cols] 
    merged_df = pd.concat([filter_photostim_table, filter_behavior_table])

    # Sort by start_frame
    merged_df = merged_df.sort_values(by='start_frame').reset_index(drop=True)

    # Total number of frames from dff
    total_frames = len(dff)

    # Track how many spontaneous bouts have been added
    spont_count = 0
    new_rows = []

    # Check for a gap before the first event
    first_start = merged_df.iloc[0]['start_frame']
    if first_start > 0:
        spont_count += 1
        label = 'spont' if spont_count == 1 else 'spont_post' if spont_count == 2 else 'spont_uk'
        new_rows.append({
            'start_frame': 0,
            'stop_frame': first_start - 1,
            'stim_name': label
        })

    # Fill internal gaps between events
    for i in range(len(merged_df) - 1):
        current_stop = merged_df.loc[i, 'stop_frame']
        next_start = merged_df.loc[i + 1, 'start_frame']
        if next_start - current_stop > 58:
            spont_count += 1
            label = 'spont' if spont_count == 1 else 'spont_post' if spont_count == 2 else 'spont_uk'
            new_rows.append({
                'start_frame': current_stop + 1,
                'stop_frame': next_start - 1,
                'stim_name': label
            })

    # Check for a gap after the last event
    last_stop = merged_df.iloc[-1]['stop_frame']
    if total_frames - 1 - last_stop > 0:
        spont_count += 1
        label = 'spont' if spont_count == 1 else 'spont_post' if spont_count == 2 else 'spont_uk'
        new_rows.append({
            'start_frame': last_stop + 1,
            'stop_frame': total_frames - 1,
            'stim_name': label
        })

    # Add the spontaneous rows
    gap_filler_df = pd.DataFrame(new_rows)
    filled_df = pd.concat([merged_df, gap_filler_df], ignore_index=True)
    filled_df = filled_df.sort_values(by='start_frame').reset_index(drop=True)

    # Generate the epoch table
    epoch_table = (
        filled_df
        .groupby('stim_name')
        .agg(start_frame=('start_frame', 'min'),
             stop_frame=('stop_frame', 'max'))
        .reset_index()
        .sort_values(by='start_frame')
    )

    # Assume frame_rate is defined
    epoch_table['start_time'] = epoch_table['start_frame'] / frame_rate
    epoch_table['stop_time'] = epoch_table['stop_frame'] / frame_rate

    # Show result
    return epoch_table