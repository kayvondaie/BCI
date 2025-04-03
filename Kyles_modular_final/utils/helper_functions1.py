def get_data_dict(force_reload=False):
    """
    Loads and caches the data_dict from file.
    """
    import mat73
    if not hasattr(get_data_dict, "_cache") or force_reload:
        print("Loading data_dict...")
        get_data_dict._cache = mat73.loadmat(r'H:/My Drive/Learning rules/BCI_data/combined_new_old_060524.mat')
    return get_data_dict._cache