
for i in range(13,len(ops['tiff_list'])):
    print(str(i))
    tif = ScanImageTiffReader(folder + ops['tiff_list'][i])