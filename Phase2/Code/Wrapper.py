#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
# Add any python libraries here
import matplotlib.pyplot as plt
from Misc.MiscUtils import *
import zipfile

def unzip_images():
	pass

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
	plt.imshow(warped_image)
	plt.show()
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

def main():
	# Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    
    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""
	rho = 32
	crop_size = (320, 240)
	patch_size = (128, 128)
	image = cv2.imread("../Data/Train/1.jpg", 0)
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, crop_size)
	patch_A, patch_B = generate_patches(image.shape, patch_size, rho)

	image_copy = np.copy(image)
	image_copy = draw_patch(image_copy, patch_A, (137, 207, 240))
	image_copy = draw_patch(image_copy, patch_B, (255, 0, 0))
	warped_image = warp(image, np.float32(patch_A), np.float32(patch_B))
	warped_copy = np.copy(warped_image)
	draw_patch(warped_copy, patch_A, (137, 207, 240))

	plt.imshow(warped_image[20: 150, 30:70])
	plt.show()

	show_image_pair(image, warped_image, patch_A)
	h4pt = patch_B - patch_A

	"""
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""
	
	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

    
if __name__ == '__main__':
    main()
 
