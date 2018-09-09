# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
from util.data_process import *

def volume_probability_prediction(temp_imgs, data_shape, label_shape, data_channel,
                                  class_num, batch_size, sess, proby, x, x_shape):
    '''
    Test one image with sub regions along z-axis
    '''
    [D, H, W] = temp_imgs[0].shape
    input_center = [int(D/2), int(H/2), int(W/2)]
    temp_prob = np.zeros([D, H, W, class_num])
    sub_image_baches = []
    for center_slice in range(int(label_shape[0]/2), D + int(label_shape[0]/2), label_shape[0]):
        center_slice = min(center_slice, D - int(label_shape[0]/2))
        sub_image_bach = []
        for chn in range(data_channel):
            temp_input_center = [center_slice, input_center[1], input_center[2]]
            sub_image = extract_roi_from_volume(
                            temp_imgs[chn], temp_input_center, data_shape)
            sub_image_bach.append(sub_image)
        sub_image_bach = np.asanyarray(sub_image_bach, np.float32)
        sub_image_baches.append(sub_image_bach)
    total_batch = len(sub_image_baches)
    max_mini_batch = int((total_batch+batch_size-1)/batch_size)
    sub_label_idx = 0
    for mini_batch_idx in range(max_mini_batch):
        data_mini_batch = sub_image_baches[mini_batch_idx*batch_size:
                                      min((mini_batch_idx+1)*batch_size, total_batch)]
        if(mini_batch_idx == max_mini_batch - 1):
            for idx in range(batch_size - (total_batch - mini_batch_idx*batch_size)):
                data_mini_batch.append(np.random.normal(0, 1, size = [data_channel] + data_shape))
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
        prob_mini_batch = sess.run(proby, feed_dict = {x:data_mini_batch, x_shape: data_shape})
        
        for batch_idx in range(prob_mini_batch.shape[0]):
            center_slice = sub_label_idx*label_shape[0] + int(label_shape[0]/2)
            center_slice = min(center_slice, D - int(label_shape[0]/2))
            temp_input_center = [center_slice, input_center[1], input_center[2], int(class_num/2)]
            sub_prob = np.reshape(prob_mini_batch[batch_idx], label_shape + [class_num])
            temp_prob = set_roi_to_volume(temp_prob, temp_input_center, sub_prob)
            sub_label_idx = sub_label_idx + 1
    return temp_prob 


def volume_probability_prediction_3d_roi(temp_imgs, data_shape, label_shape, data_channel, 
                                  class_num, batch_size, sess, proby, x):
    '''
    Test one image with sub regions along x, y, z axis
    '''
    [D, H, W] = temp_imgs[0].shape
    temp_prob = np.zeros([D, H, W, class_num])
    sub_image_batches = []
    sub_image_centers = []
    roid_half = int(label_shape[0]/2)
    roih_half = int(label_shape[1]/2)
    roiw_half = int(label_shape[2]/2)
    for centerd in range(roid_half, D + roid_half, label_shape[0]):
        centerd = min(centerd, D - roid_half)
        for centerh in range(roih_half, H + roih_half, label_shape[1]):
            centerh =  min(centerh, H - roih_half) 
            for centerw in range(roiw_half, W + roiw_half, label_shape[2]):
                centerw =  min(centerw, W - roiw_half) 
                temp_input_center = [centerd, centerh, centerw]
                sub_image_centers.append(temp_input_center)
                sub_image_batch = []
                for chn in range(data_channel):
                    sub_image = extract_roi_from_volume(temp_imgs[chn], temp_input_center, data_shape)
                    sub_image_batch.append(sub_image)
                sub_image_bach = np.asanyarray(sub_image_batch, np.float32)
                sub_image_batches.append(sub_image_bach)

    total_batch = len(sub_image_batches)
    max_mini_batch = int((total_batch + batch_size - 1)/batch_size)
    sub_label_idx = 0
    for mini_batch_idx in range(max_mini_batch):
        data_mini_batch = sub_image_batches[mini_batch_idx*batch_size:
                                      min((mini_batch_idx+1)*batch_size, total_batch)]
        if(mini_batch_idx == max_mini_batch - 1):
            for idx in range(batch_size - (total_batch - mini_batch_idx*batch_size)):
                data_mini_batch.append(np.random.normal(0, 1, size = [data_channel] + data_shape))
        data_mini_batch = np.asanyarray(data_mini_batch, np.float32)
        data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
        outprob_mini_batch = sess.run(proby, feed_dict = {x:data_mini_batch})
        
        for batch_idx in range(batch_size):
            glb_batch_idx = batch_idx + mini_batch_idx * batch_size
            if(glb_batch_idx >= total_batch):
                continue
            temp_center = sub_image_centers[glb_batch_idx]
            temp_prob = set_roi_to_volume(temp_prob, temp_center + [1], outprob_mini_batch[batch_idx])
            sub_label_idx = sub_label_idx + 1
    return temp_prob

def get_image_adaptive_tensor_size(img_size, tensor_size, div_factor):
    output_size = max(img_size, tensor_size)
    if(output_size % div_factor !=0):
        output_size = output_size + div_factor - output_size%div_factor
    return output_size


def test_one_image_with_three_nets(temp_imgs, data_shapes, label_shapes, data_channel, class_num,
                   batch_size, sess, outputs, inputs, x_shape):
    '''
    Test one image with three anisotropic networks with fixed or adaptable tensor height and width.
    These networks are used in axial, saggital and coronal view respectively.
    '''
    views = ['axial', 'sagittal', 'coronal']
    prob_list = []
    for i in range(3):
        tr_volumes = transpose_volumes(temp_imgs, views[i])
        [D, H, W]  = tr_volumes[0].shape
        data_shape = data_shapes[i]
        label_shape = label_shapes[i]
        data_shape[1]  = get_image_adaptive_tensor_size(H, data_shape[1], 4)
        data_shape[2]  = get_image_adaptive_tensor_size(W, data_shape[2], 4)
        label_shape[1] = data_shape[1]
        label_shape[2] = data_shape[2]
        prob_i = volume_probability_prediction(tr_volumes, data_shape, label_shape, data_channel,
                class_num, batch_size, sess, outputs[i], inputs[i], x_shape)
        if(i == 1):
            prob_i = np.transpose(prob_i, [1,2,0,3])
        elif(i == 2):
            prob_i = np.transpose(prob_i, [1,0,2,3])
        prob_list.append(prob_i)
    prob = (prob_list[0] + prob_list[1] + prob_list[2])/3.0
    return prob    
