# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 13:39:01 2024

@author: tomd

Downloaded from:

Koren, I., Dror‐Schwartz, T., Stollar, O.A., & Chekroun, M.D. (2024).
Data for: Cloud vs. void chord length distributions (LvL) as a measure
for cloud field organization (code) [Software]. WIS.
https://doi.org/10.34933/b7f2cded‐40d3‐4be9‐bdc6‐31b2694ca49c

"""
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import ks_2samp
from PIL import Image
from scipy.ndimage import label
import cv2



def LvL(cloud_mask):
    # Convert cloud_mask to double
    cloud_mask = cloud_mask.astype(float)
    sz = cloud_mask.shape
    mnsz = min(sz)  # minimum scale of the field
    sz_cloud_mask = sz[0] * sz[1]
    p = np.sum(cloud_mask) / sz_cloud_mask  # cloud fraction

    # estimating the ideal length to capture almost 100% of the histogram
    # we want that the error < exp(-12) for the perfect rand case
    mxln = int(np.floor(abs(12 / np.log(p))) + 1)
    mxln = min(mxln, mnsz)

    # The cloud part
    # Flatenning along the two directions
    # then we need to divide c1 and c2 by two

    B = cloud_mask.flatten()  # rows
    C = cloud_mask.flatten(order='F') #columns 
    B = np.concatenate((C, B))

    L, num_labels = label(B)
    # Label connected components in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(L.astype(np.uint8))

    # Extract area of each connected component
    areas = stats[1:, cv2.CC_STAT_AREA]
    # Adding one to the max area
    mx_ar = np.max(areas) + 1   

    # To avoid single histogram
    if mx_ar < mxln:
        mx_ar = mxln
    else:
        mxln = mx_ar

    # Get the cloud chord length counts - c1    
    c1, _ = np.histogram(areas, bins=np.arange(1, mx_ar + 2))
    # Correct for flattening in two directions
    c1 = c1 / 2
    s1 = np.sum(c1)

    # Now, the theortical calculations ct, nt for a given cloud fraction (p)
    nt1 = np.arange(1, mxln + 1)
    ct1 = (sz_cloud_mask * (1 - p) ** 2) * p ** nt1
    st1 = np.sum(ct1)
    nt1 = nt1.astype(int)


    # Get the KS score for the cloud part
    adf1 = np.abs(np.cumsum(ct1 / st1) - np.cumsum(c1 / s1))
    KS1 = np.max(adf1)


    # The void part
    q = 1 - p # void fraction
    mxln = int(np.floor(abs(12 / np.log(q))) + 1)
    mxln = min(mxln, mnsz)
    B = -B + 1

    L, num_labels = label(B)
    # Label connected components in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(L.astype(np.uint8))

    # Extract area of each connected component
    areas = stats[1:, cv2.CC_STAT_AREA]
    # Adding one to the max area
    mx_ar = np.max(areas) + 1   

    if mx_ar < mxln:
        mx_ar = mxln
    else:
        mxln = mx_ar

    c2, _ = np.histogram(areas, bins=np.arange(1, mx_ar + 2))
    c2 = c2 / 2
    s2 = np.sum(c2)

    nt2 = np.arange(1, mxln + 1)
    ct2 = (sz_cloud_mask * (1 - q) ** 2) * q ** nt2
    st2 = np.sum(ct2)
    nt2 = nt2.astype(int)

    # Get the KS score for the void part
    adf2 = np.abs(np.cumsum(ct2 / st2) - np.cumsum(c2 / s2))
    KS2 = np.max(adf2)

    return KS1, KS2