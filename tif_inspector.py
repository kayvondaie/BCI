for i in range(len(ops['tiff_list'])):
    print(str(i))
    tif = ScanImageTiffReader(folder + ops['tiff_list'][i])