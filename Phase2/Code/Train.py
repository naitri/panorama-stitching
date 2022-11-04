#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow as tf
import cv2
import sys
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel, UnsupervisedModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from io import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *

# Don't generate pyc codes
sys.dont_write_bytecode = True

    
def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize, ModelType):
    """
    Inputs: 
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    I2Batch = []
    LabelBatch = []
    C4pt = []
    Patches = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain)-1)
        
        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.jpg'
        ImageNum += 1
    	
    	##########################################################
    	# Add any standardization or data augmentation here!
    	##########################################################

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
        if ModelType == 'sup':
            h4pt = patch_B - patch_A
            h4pt = np.reshape(h4pt, (8))

            # Append All Images and Mask
            I1Batch.append(stacked_imgs)
            LabelBatch.append(h4pt)
        else:
            I1Batch.append(np.float32(image))
            I2Batch.append(np.float32(imgB.reshape(ImageSize[0], ImageSize[1], 1)))
            C4pt.append(np.float32(patch_A))
            Patches.append(stacked_imgs)

    if ModelType == 'sup':
        return I1Batch, LabelBatch
    else:
        return I1Batch, Patches, C4pt, I2Batch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    
def TrainOperation(ImgPH, LabelPH, I1PH, I2PH, C4PH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of data or for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """      
    # Predict output with forward pass
    # prLogits, prSoftMax = HomographyModel(ImgPH, ImageSize, MiniBatchSize)

    if ModelType.lower() == 'sup':
        pred = HomographyModel(ImgPH, ImageSize, MiniBatchSize)

        with tf.name_scope('Loss'):
            ###############################################
            # Fill your loss function of choice here!
            ###############################################
            loss = tf.reduce_sum(tf.squared_difference(pred, LabelPH))

        with tf.name_scope('Adam'):
        	###############################################
        	# Fill your optimizer of choice here!
        	###############################################
            Optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    elif ModelType.lower() == 'unsup':
        pred_I2, I2, _ = UnsupervisedModel(ImgPH, C4PH, I2PH, ImageSize, MiniBatchSize)

        with tf.name_scope('Loss'):
            ###############################################
            # Fill your loss function of choice here!
            ###############################################
            loss = tf.reduce_mean(tf.abs(pred_I2 - I2))

        with tf.name_scope('Adam'):
            ###############################################
            # Fill your optimizer of choice here!
            ###############################################
            Optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)


    # Tensorboard
    # Create a summary to monitor loss tensor
    iter_loss_summary = tf.summary.scalar('LossEveryIter', loss)
    epoch_lossPH = tf.placeholder(tf.float32, shape=None)
    epoch_loss_summary = tf.summary.scalar('LossEveryEpoch', epoch_lossPH)
    # tf.summary.image('Anything you want', AnyImg)
    # Merge all summaries into a single operation
    # MergedSummaryOP = tf.summary.merge_all()
    MergedSummaryOP1 = tf.summary.merge([iter_loss_summary])
    MergedSummaryOP2 = tf.summary.merge([epoch_loss_summary])

    # Setup Saver
    Saver = tf.train.Saver()
    epoch_accuracy = np.array([0, 0])
    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
            
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            epoch_loss = 0
            loss_vals = []
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                if ModelType.lower() == 'sup':
                    I1Batch, LabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize, ModelType)
                    FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch}
                elif ModelType.lower() == 'unsup':
                    I1Batch, Patches, C4pt, I2Batch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize, ModelType)
                    FeedDict = {ImgPH: Patches, C4PH: C4pt, I2PH: I2Batch}

                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP1], feed_dict=FeedDict)
                loss_vals.append(LossThisBatch)
                epoch_loss += LossThisBatch
                
                # Save checkpoint every some SaveCheckPoint's iterations
                if PerEpochCounter % SaveCheckPoint == 0:
                    # Save the Model learnt in this epoch
                    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print('\n' + SaveName + ' Model Saved...')

                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                # Writer.flush()

            epoch_loss = epoch_loss / NumIterationsPerEpoch
            print("Mean loss per epoch: ", np.mean(loss_vals))
            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved...')
            total_summary = sess.run(MergedSummaryOP2, feed_dict={epoch_lossPH: epoch_loss})
            Writer.add_summary(total_summary, Epochs)
            Writer.flush()
            

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='../Data', help='Base path of images, Default: ../Data')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='unsup', help='Model type, Supervised or Unsupervised? Choose from sup and unsup, Default:unsup')
    Parser.add_argument('--NumEpochs', type=int, default=30, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=64, help='Size of the MiniBatch to use, Default:64')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], 2))
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, NumClasses)) # OneHOT labels
    I1PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
    I2PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], 1))
    C4PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 4, 2))

    if ModelType.lower() == 'sup':
        print("Training supervised network...\n")
    elif ModelType.lower() == 'unsup':
        print("Training unsupervised network...\n")
    else:
        print("Please enter the correct model type and try again. [sup/unsup]\n")

    TrainOperation(ImgPH, LabelPH, I1PH, I2PH, C4PH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType)
        
    
if __name__ == '__main__':
    main()
 
