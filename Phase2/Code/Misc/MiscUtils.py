"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import time
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Don't generate pyc codes
sys.dont_write_bytecode = True

def tic():
    StartTime = time.time()
    return StartTime

def toc(StartTime):
    return time.time() - StartTime

def remap(x, oMin, oMax, iMin, iMax):
    # Taken from https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratios
    #range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if iMin == iMax:
        print("Warning: Zero output range")
        return None

     # portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    result = np.add(np.divide(np.multiply(x - iMin, oMax - oMin), iMax - iMin), oMin)

    return result

def FindLatestModel(CheckPointPath):
    FileList = glob.glob(CheckPointPath + '*.ckpt.index') # * means all if need specific format then *.csv
    LatestFile = max(FileList, key=os.path.getctime)
    # Strip everything else except needed information
    LatestFile = LatestFile.replace(CheckPointPath, '')
    LatestFile = LatestFile.replace('.ckpt.index', '')
    return LatestFile


def convertToOneHot(vector, n_labels):
    return np.equal.outer(vector, np.arange(n_labels)).astype(np.float)

    
"""
Expects an image, a list of points in tuple form,
and an RGB color in tuple form.
"""
def draw_patch(image, pts, color, thickness=2):
    val_range = np.arange(4)
    for i in range(4):
        cv2.line(image, pts[val_range[i-1]], pts[val_range[i]], color, thickness)
    plt.imshow(image)
    plt.show()
    return image

"""
Expects image size (height x width), patch size (width x height) and rho
Returns patch.
"""
def generate_patches(img_size, patch_size, rho):
    height, width = img_size
    w_lim = (rho, width - patch_size[1] - rho)
    h_lim = (rho, height - patch_size[0] - rho)
    p_x, p_y = (np.random.randint(w_lim[0], w_lim[1]), np.random.randint(h_lim[0], h_lim[1]))
    patch_A = ((p_x, p_y), (p_x + patch_size[1], p_y), (p_x + patch_size[1], p_y + patch_size[0]), (p_x, p_y + patch_size[0]))
    # patch_A = ((p_y, p_x), (p_y, p_x + patch_size[0]), (p_y + patch_size[1], p_x + patch_size[0]), (p_y + patch_size[1], p_x))
    patch_A = np.array(patch_A)
    perturb = np.random.randint(-rho, rho, (4, 2))
    patch_B = patch_A + perturb
    return patch_A, patch_B

"""
Gets patches, computes forward and inverse homography.
Returns warped image.
"""
def warp(image, patch_A, patch_B):
    height, width = image.shape
    hAB = cv2.getPerspectiveTransform(patch_A, patch_B)
    hBA = np.linalg.inv(hAB)
    warped_image = cv2.warpPerspective(image, hBA, (width, height))
    # plt.imshow(warped_image)
    # plt.show()
    return warped_image

"""
Receives original and warped patch.
Shows both patches side by side.
"""
def show_image_pair(image, warped_image, patch_A,):
    fig, axs = plt.subplots(1, 2)
    x_min, x_max = patch_A[0][0], patch_A[1][0]
    y_min, y_max = patch_A[0][1], patch_A[2][1]
    axs[0].imshow(image[y_min:y_max, x_min:x_max])
    axs[1].imshow(warped_image[y_min:y_max, x_min:x_max])
    axs[0].set_title("Original patch (pA)")
    axs[1].set_title("Warped patch (pB)")
    plt.show()