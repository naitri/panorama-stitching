"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Code for DLT and unsupervised model was adapted from
https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018/blob/master/code/homography_model.py
"""

import tensorflow as tf
import sys
import numpy as np
from Misc.unsup_utils import *
from Misc.TFSpatialTransformer import transformer
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomographyModel(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    # Convolutional layer #1
    x = tf.layers.conv2d(inputs=Img, filters=64, kernel_size=(3, 3), padding='same', name='conv1')
    x = tf.layers.batch_normalization(x, name='batch_norm1')
    x = tf.nn.relu(x, name='relu1')

    # Convolutional layer #2
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), padding='same', name='conv2')
    x = tf.layers.batch_normalization(x, name='batch_norm2')
    x = tf.nn.relu(x, name='relu2')

    # Max-pooling layer #1
    x = tf.layers.max_pooling2d(inputs=x, pool_size=(2, 2), strides=(2, 2), name='max_pool1')

    # Convolutional layer #3
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), padding='same', name='conv3')
    x = tf.layers.batch_normalization(x, name='batch_norm3')
    x = tf.nn.relu(x, name='relu3')

    # Convolutional layer #4
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), padding='same', name='conv4')
    x = tf.layers.batch_normalization(x, name='batch_norm4')
    x = tf.nn.relu(x, name='relu4')

    # Max-pooling layer #2
    x = tf.layers.max_pooling2d(inputs=x, pool_size=(2, 2), strides=(2, 2), name='max_pool2')

    # Convolutional layer #5
    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=(3, 3), padding='same', name='conv5')
    x = tf.layers.batch_normalization(x, name='batch_norm5')
    x = tf.nn.relu(x, name='relu5')

    # Convolutional layer #6
    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=(3, 3), padding='same', name='conv6')
    x = tf.layers.batch_normalization(x, name='batch_norm6')
    x = tf.nn.relu(x, name='relu6')

    # Max-pooling layer #3
    x = tf.layers.max_pooling2d(inputs=x, pool_size=(2, 2), strides=(2, 2), name='max_pool3')

    # Convolutional layer #7
    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=(3, 3), padding='same', name='conv7')
    x = tf.layers.batch_normalization(x, name='batch_norm7')
    x = tf.nn.relu(x, name='relu7')

    # Convolutional layer #8
    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=(3, 3), padding='same', name='conv8')
    x = tf.layers.batch_normalization(x, name='batch_norm8')
    x = tf.nn.relu(x, name='relu8')
    x = tf.layers.flatten(x)

    # Fully-connected layer #1
    x = tf.layers.dense(inputs=x, units='1024', name='fc1')
    x = tf.layers.batch_normalization(x, name='batch_norm9')
    x = tf.nn.relu(x, name='relu9')
    x = tf.nn.dropout(x, rate=0.5, name='dropout')

    # Fully-connected layer #2
    x = tf.layers.dense(inputs=x, units='8', name='fc2')

    return x

"""
Inputs:
C4pt consists of the four corners of Image A
H4pt is the predicted homography parameterization from the supervised net
batch_size: batch_size selected for training
Outputs:
3x3 homography parameterization matrix H_mat
"""
def solveDLT(C4pt, H4pt, batch_size):
    pts_1_tile = tf.expand_dims(C4pt, [2]) # BATCH_SIZE x 8 x 1

    # Solve for H using DLT
    pred_h4p_tile = tf.expand_dims(H4pt, [2]) # BATCH_SIZE x 8 x 1

    # 4 points on the second image
    pred_pts_2_tile = tf.add(pred_h4p_tile, pts_1_tile)

    # Auxiliary tensors used to create Ax = b equation
    M1 = tf.constant(Aux_M1, tf.float32)
    M1_tensor = tf.expand_dims(M1, [0])
    M1_tile = tf.tile(M1_tensor,[batch_size,1,1])

    M2 = tf.constant(Aux_M2, tf.float32)
    M2_tensor = tf.expand_dims(M2, [0])
    M2_tile = tf.tile(M2_tensor,[batch_size,1,1])

    M3 = tf.constant(Aux_M3, tf.float32)
    M3_tensor = tf.expand_dims(M3, [0])
    M3_tile = tf.tile(M3_tensor,[batch_size,1,1])

    M4 = tf.constant(Aux_M4, tf.float32)
    M4_tensor = tf.expand_dims(M4, [0])
    M4_tile = tf.tile(M4_tensor,[batch_size,1,1])

    M5 = tf.constant(Aux_M5, tf.float32)
    M5_tensor = tf.expand_dims(M5, [0])
    M5_tile = tf.tile(M5_tensor,[batch_size,1,1])

    M6 = tf.constant(Aux_M6, tf.float32)
    M6_tensor = tf.expand_dims(M6, [0])
    M6_tile = tf.tile(M6_tensor,[batch_size,1,1])


    M71 = tf.constant(Aux_M71, tf.float32)
    M71_tensor = tf.expand_dims(M71, [0])
    M71_tile = tf.tile(M71_tensor,[batch_size,1,1])

    M72 = tf.constant(Aux_M72, tf.float32)
    M72_tensor = tf.expand_dims(M72, [0])
    M72_tile = tf.tile(M72_tensor,[batch_size,1,1])

    M8 = tf.constant(Aux_M8, tf.float32)
    M8_tensor = tf.expand_dims(M8, [0])
    M8_tile = tf.tile(M8_tensor,[batch_size,1,1])

    Mb = tf.constant(Aux_Mb, tf.float32)
    Mb_tensor = tf.expand_dims(Mb, [0])
    Mb_tile = tf.tile(Mb_tensor,[batch_size,1,1])

    # Form the equations Ax = b to compute H
    # Form A matrix
    A1 = tf.matmul(M1_tile, pts_1_tile) # Column 1
    A2 = tf.matmul(M2_tile, pts_1_tile) # Column 2
    A3 = M3_tile                   # Column 3
    A4 = tf.matmul(M4_tile, pts_1_tile) # Column 4
    A5 = tf.matmul(M5_tile, pts_1_tile) # Column 5
    A6 = M6_tile                   # Column 6
    A7 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M72_tile, pts_1_tile)# Column 7
    A8 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M8_tile, pts_1_tile)# Column 8

    A_mat = tf.transpose(tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]),\
                                   tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),\
                                   tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]),\
         tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],axis=1), perm=[0,2,1]) # BATCH_SIZE x 8 (A_i) x 8
    print('--Shape of A_mat:', A_mat.get_shape().as_list())
    # Form b matrix
    b_mat = tf.matmul(Mb_tile, pred_pts_2_tile)
    print('--shape of b:', b_mat.get_shape().as_list())

    # Solve the Ax = b
    H_8el = tf.matrix_solve(A_mat , b_mat)  # BATCH_SIZE x 8.
    print('--shape of H_8el', H_8el)


    # Add ones to the last cols to reconstruct H for computing reprojection error
    h_ones = tf.ones([batch_size, 1, 1])
    H_9el = tf.concat([H_8el,h_ones],1)
    H_flat = tf.reshape(H_9el, [-1,9])
    H_mat = tf.reshape(H_flat,[-1,3,3])   # BATCH_SIZE x 3 x 3

    return H_mat

def UnsupervisedModel(Img, C4pt, I2, ImageSize, MiniBatchSize):
    H4pt = HomographyModel(Img, ImageSize, MiniBatchSize)
    C4pt = tf.reshape(C4pt, [MiniBatchSize, 8])
    H_mat = solveDLT(C4pt, H4pt, MiniBatchSize)
    width, height = ImageSize[:2]
    # Constants and variables used for spatial transformer
    M = np.array([[width/2.0, 0., width/2.0],
                  [0., height/2.0, height/2.0],
                  [0., 0., 1.]]).astype(np.float32)

    M_tensor  = tf.constant(M, tf.float32)
    M_tile   = tf.tile(tf.expand_dims(M_tensor, [0]), [MiniBatchSize, 1,1])
    # Inverse of M
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv   = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [MiniBatchSize,1,1])

    y_t = tf.range(0, MiniBatchSize * width * height, width * height)
    z =  tf.tile(tf.expand_dims(y_t,[1]),[1, ImageSize[0] * ImageSize[1]])
    batch_indices_tensor = tf.reshape(z, [-1]) # Add these value to patch_indices_batch[i] for i in range(num_pairs) # [BATCH_SIZE*WIDTH*HEIGHT]

    # Transform H_mat since we scale image indices in transformer
    H_mat = tf.matmul(tf.matmul(M_tile_inv, H_mat), M_tile)
    # Transform image 1 (large image) to image 2
    out_size = (height, width)
    I1 = tf.slice(Img, [0, 0, 0, 0], [MiniBatchSize, ImageSize[0], ImageSize[1], 1])
    warped_images, _ = transformer(I1, H_mat, out_size)


    # Extract the warped patch from warped_images by flatting the whole batch before using indices
    # Note that input I  is  3 channels so we reduce to gray
    warped_gray_images = tf.reduce_mean(warped_images, 3)
    pred_I2_flat = warped_gray_images
    pred_I2 = tf.reshape(pred_I2_flat, [MiniBatchSize, ImageSize[0], ImageSize[1], 1])

    return pred_I2, I2, H4pt
