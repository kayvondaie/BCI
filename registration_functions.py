# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:21:57 2023

@author: scanimage
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.ndimage import affine_transform
import skimage
from skimage.feature import match_descriptors
from skimage.transform import AffineTransform, warp
from skimage.measure import ransac
import copy    
    

# Define the function to find the affine transformation
def find_affine_transform(img1, img2):
    # Find the cross-correlation between the images
    corr = np.fft.fft2(img2) * np.fft.fft2(img1).conj()
    corr = np.fft.fftshift(np.fft.ifft2(corr).real)
    
    # Find the location of the maximum correlation
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    
    # Define the center of the images
    center = np.array([img2.shape[0] / 2, img2.shape[1] / 2])
    
    # Define the translation vector
    translation = center - np.array([y, x])
    
#     # Define the affine transformation matrix
    M = np.array([[1, 0, translation[0]],
                   [0, 1, translation[1]],
                   [0, 0, 1]])
#     stat_new = copy.deepcopy(stat)
#     for ci in range(stat.shape[0]):
#         x = stat[ci]['xpix']
#         x = x.astype(int)    
#         y = stat[ci]['ypix']
#         y = y.astype(int)
# 
#         bw = np.zeros(np.shape(img1))
#         bw[y,x] = 1        
#         bww = affine_transform(bw, M)
#         y,x = np.where(bww==1)
#         b = np.argsort(x);
#         x = x[b]
#         y = y[b]
#         stat_new[ci]['xpix'] = x;
#         stat_new[ci]['ypix'] = y;
# =============================================================================
    
    return M

# Define the objective function to minimize
def objective(params, source, target):
    # Extract the translation and rotation parameters
    tx, ty, theta = params
    # Calculate the affine transformation matrix
    M = np.array([[np.cos(theta), -np.sin(theta), tx],
                  [np.sin(theta), np.cos(theta), ty],
                  [0, 0, 1]])
    # Apply the transformation to the source image
    warped = affine_transform(source, M)
    # Calculate the mean squared error between the warped image and the target image
    mse = np.mean((warped - target)**2)
    return mse

def shift_rois_affine(img1,img2,stat):
    descriptor_extractor = skimage.feature.ORB(n_keypoints=500)
    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    # Match the keypoints between the two images using descriptors
    matches = match_descriptors(descriptors1, descriptors2)

    # Extract the matched keypoints from the two images
    src_pts = keypoints1[matches[:, 0]][:, ::-1]  # swap x, y coordinates
    dst_pts = keypoints2[matches[:, 1]][:, ::-1]

    # Estimate the affine transformation matrix using RANSAC
    model_robust, inliers = ransac((src_pts, dst_pts), AffineTransform, min_samples=3,
                                   residual_threshold=2, max_trials=100)

    # Apply the affine transformation to the source image
    warped = skimage.transform.warp(img2, model_robust)
    matrix = AffineTransform(rotation=np.deg2rad(0), translation=[11,11])
    stat_new = copy.deepcopy(stat)

    for ci in range(stat.shape[0]):
        x = stat[ci]['xpix']
        x = np.reshape(x.astype(int),(x.shape[0],1))    
        y = stat[ci]['ypix']
        y = np.reshape(y.astype(int),(y.shape[0],1))
        z = np.ones(x.shape)

        centroid = (int(round(np.mean(y))),int(round(np.mean(x))))
        ind1 = np.arange(centroid[0]-30,centroid[0]+30)
        ind2 = np.arange(centroid[1]-30,centroid[1]+30)  
        ind1[ind1<0] = 0
        ind1[ind1>=img1.shape[0]] = img1.shape[0]-1
        ind2[ind2<0] = 0
        ind2[ind2>=img1.shape[1]] = img1.shape[1]-1
        
        window1 = img1[np.ix_(ind1, ind2)]
        window2 = warped[np.ix_(ind1, ind2)]                
        
        M = find_affine_transform(window1,window2)
        
        theta = model_robust.rotation
        tx = model_robust.translation[0] - M[0,2]
        ty = model_robust.translation[1] - M[1,2]

        matrix = AffineTransform(rotation=np.deg2rad(theta), translation=[tx, ty])
        xy = np.concatenate((x,y),axis = 1)
        xy_new = matrix(xy)
      
        xnew = xy_new[:,0].astype(int)        
        ynew = xy_new[:,1].astype(int)
        x = x.T
        y = y.T
        
        stat_new[ci]['xpix'] = xnew;
        stat_new[ci]['ypix'] = ynew;
      
    
    return stat_new, model_robust

def show_new_rois(img1,img2,stat,stat_new,numROis=21):
    fig, ax = plt.subplots()
    plt.subplot(1,2,1)
    plt.imshow(img1,cmap='gray', vmin=0, vmax=300)
    for ci in range(numROis):
        x = stat[ci]['xpix']
        x = x.astype(int)    
        y = stat[ci]['ypix']
        y = y.astype(int)
        
        bw = np.zeros(np.shape(img1))
        bw[y,x] = 1

        contours = plt.contour(bw, levels=[0.5], colors='r', linewidths=0.5)
        contour = contours.collections[0].get_paths()[0]
        patch = Polygon(contour.vertices, edgecolor='r', facecolor='none', linewidth=0.01)
        ax.add_patch(patch)
        #ax.set_xlim(0, bw.shape[1])
        #ax.set_ylim(0, bw.shape[0])
    plt.title('Original')
    plt.subplot(1,2,2)
    plt.imshow(img2,cmap='gray', vmin=0, vmax=300)

    for ci in range(numROis):
        x = stat_new[ci]['xpix']
        x = x.astype(int)    
        y = stat_new[ci]['ypix']
        y = y.astype(int)
        
        bw = np.zeros(np.shape(img2))
        bw[y,x] = 1
        bww = bw
        
        contours = plt.contour(bww, levels=[0.5], colors='r', linewidths=0.5)
        contour = contours.collections[0].get_paths()[0]
        patch = Polygon(contour.vertices, edgecolor='r', facecolor='none', linewidth=0.01)
        ax.add_patch(patch)
        #ax.set_xlim(0, bw.shape[1])
        #ax.set_ylim(0, bw.shape[0])
    plt.title('Rotated')
    plt.show()