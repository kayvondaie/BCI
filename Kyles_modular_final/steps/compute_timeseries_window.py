def run():
    # Cell 109
ts_range = (np.min(TS_PRE), np.max(TS_POST))
ts_idxs = np.arange(
    int((ts_range[0] - T_START) * SAMPLE_RATE), int((ts_range[1] - T_START) * SAMPLE_RATE),
)

n_ts_idxs = ts_idxs.shape[0]

print(ts_idxs.shape[0])

    print("✅ compute_timeseries_window ran successfully.")
if __name__ == '__main__':
    run()