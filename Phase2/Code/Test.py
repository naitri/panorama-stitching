#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import cv2
import os
import sys
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from io import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [128, 128, 2]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.jpg'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.jpg')

    return ImageSize, DataPath
    
def ReadImages(ImageSize, DataPath):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    I1Batch = []
    I2Batch = []
    LabelBatch = []
    C4pt = []
    Patches = []   
    ImageName = DataPath + str(random.randint(1, 1000)) + '.jpg'
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    rho = 32
    crop_size = (320, 240)
    patch_size = ImageSize[:2]
    image = cv2.imread(RandImageName, 0)
    image = cv2.resize(image, crop_size)
    patch_A, patch_B = generate_patches(image.shape, patch_size, rho)

    warped_image = warp(image, np.float32(patch_A), np.float32(patch_B))
    x_min, x_max = patch_A[0][0], patch_A[1][0]
    y_min, y_max = patch_A[0][1], patch_A[2][1]
    imgA = image[y_min:y_max, x_min:x_max]
    imgB = warped_image[y_min:y_max, x_min:x_max]

    # Stack grayscale image channels and reshape 4-point parameterization matrix to 1-D
    stacked_imgs = np.dstack((imgA, imgB))
    if ModelType.lower() == 'sup':
        h4pt = patch_B - patch_A
        h4pt = np.reshape(h4pt, (8))

        # Append All Images and Mask
        I1Batch.append(stacked_imgs)
        LabelBatch.append(h4pt)
        return I1Batch, LabelBatch, image
    elif ModelType.lower() == 'unsup':
        I1Batch.append(np.float32(image))
        I2Batch.append(np.float32(imgB.reshape(ImageSize[0], ImageSize[1], 1)))
        C4pt.append(np.float32(patch_A))
        Patches.append(stacked_imgs)
        return I1Batch, Patches, C4pt, I2Batch, image
                

def TestOperation(ImgPH, ImageSize, ModelPath, DataPath, ModelType):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """

    LabelPH = tf.placeholder(tf.float32, shape=(1, NumClasses)) # OneHOT labels
    I1PH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]))
    I2PH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], 1))
    C4PH = tf.placeholder(tf.float32, shape=(1, 4, 2))
    if ModelType.lower() == 'sup':
        I1Batch, LabelBatch, image = ReadImages(DataPath)
        H4pt = HomographyModel(ImgPH, ImageSize, 1)

        # Setup Saver
        Saver = tf.train.Saver()

        with tf.Session() as sess:
            Saver.restore(sess, ModelPath)
            
            FeedDict = {ImgPH: I1Batch}
            pred = sess.run(H4pt,FeedDict)

        pred = I1Batch[:, :, 0] pred.reshape(4,2)
        image = cv2.polylines(image, np.int32(I1Batch[:, :, 1]),True,(255, 0,0), 3)
        image = cv2.polylines(image, np.int32(pred),True,(255,255,0), 3)
        plt.figure()
        plt.imshow(image)
        plt.show()
    else:
        pred_I2, I2, pred_H4pt = Unsupervised_HomographyModel(ImgPH, CornerPH, patch1PH, patch2PH, ImageSize, 1)
        # Setup Saver
        Saver = tf.train.Saver()

        with tf.Session() as sess:
            Saver.restore(sess, ModelPath)

            I1Batch, Patches, C4pt, I2Batch, image = ReadImages(DataPath)
            FeedDict = {ImgPH: Patches, C4PH: C4pt, I2PH: I2Batch}
            pred = sess.run(pred_H4pt, FeedDict)

        src_new=src+Predicted.reshape(4,2)
        image = cv2.polylines(image,np.int32([dst]),True,(255, 0, 0), 3)
        image = cv2.polylines(image,np.int32([src_new]),True,(0, 0, 255), 3)
        plt.figure()
        plt.imshow(image)
        plt.show()




def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/029model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='../Data/P1TestSet/', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    # Setup all needed parameters including file reading
    ImageSize, DataPath = SetupAll(BasePath)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 2))

    TestOperation(ImgPH, ImageSize, ModelPath, DataPath, ModelType)
     
if __name__ == '__main__':
    main()
 
