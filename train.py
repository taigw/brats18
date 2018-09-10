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

import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers
from niftynet.layer.loss_segmentation import LossFunction
from util.image_io.dataset_from_tfrecord import DataSetFromTFRecord
from util.preprocess.intensity_normalize import IntensityNormalizeLayer
from util.preprocess.foreground_mask import ForegroundMaskLayer
from util.preprocess.crop_image import CropImageToFixedSizeLayer
from util.preprocess.flip import FlipLayer
from util.preprocess.rotate import RotateLayer
from util.preprocess.transpose import TransposeLayer
from util.preprocess.sample_mask import SampleMaskLayer
from util.preprocess.label_mapping import LabelMappingLayer
from util.sampler.random_sampler_with_crop import RandomSamplerWithCrop
from util.train_test_func import *
from util.parse_config import parse_config
from util.MSNet import MSNet

class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        # add your own networks here
        print('unsupported network:', name)
        exit()

def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label 
        input_tensor: tensor with shae [N, D, H, W, 1]
        output_tensor: shape [N, D, H, W, num_class]
    """
    tensor_list = []
    for i in range(num_class):
        temp_prob = tf.equal(input_tensor, i*tf.ones_like(input_tensor,tf.int32))
        tensor_list.append(temp_prob)
    output_tensor = tf.concat(tensor_list, axis=-1)
    output_tensor = tf.cast(output_tensor, tf.float32)
    return output_tensor

def soft_dice_loss(prediction, soft_ground_truth, weight_map=None):
    num_class = prediction.get_shape()[-1]
    pred   = tf.reshape(prediction, [-1, num_class])
    pred   = tf.nn.softmax(pred)
    ground = tf.reshape(soft_ground_truth, [-1, num_class])
    if(weight_map is not None):
        weight_map = tf.reshape(weight_map, [-1])
        weight_map_nclass = tf.reshape(
            tf.tile(weight_map, [num_class]), pred.get_shape())
        ref_vol = tf.reduce_sum(weight_map_nclass*ground, 0)
        intersect = tf.reduce_sum(weight_map_nclass*ground*pred, 0)
        seg_vol = tf.reduce_sum(weight_map_nclass*pred, 0)
    else:
        ref_vol = tf.reduce_sum(ground, 0)
        intersect = tf.reduce_sum(ground*pred, 0)
        seg_vol = tf.reduce_sum(pred, 0)
    dice_score = 2.0*intersect/(ref_vol + seg_vol + 1.0)
    dice_score = tf.reduce_mean(dice_score)
    return 1.0-dice_score

def get_brats_preprocess_layers(config_data, mode = tf.estimator.ModeKeys.TRAIN):
    """
    get preprocess layers for brats images
    mode: one of tf.estimator.ModeKeys.TRAIN, EVAL and PREDICT
    """
    pre_processor = []
        
    #  extract non zero region as foreground mask
    fg_mask = ForegroundMaskLayer()
    pre_processor.append(fg_mask)
    
    #  crop the non zero region to a fixed size
    crop_size = config_data.get('crop_size', None)
    if(crop_size is not None):
        crop_img = CropImageToFixedSizeLayer()
        crop_img.set_crop_output_size(crop_size)
        pre_processor.append(crop_img)
    
    # normalize each input channel with mean and std
    fg_norm = IntensityNormalizeLayer()
    if('feature_channel_num' in config_data.keys()):
        feature_channel_num = config_data['feature_channel_num']
    else:
        feature_channel_num = len(config_data['feature_channel_names'])
    fg_norm.set_channel_number(feature_channel_num)
    pre_processor.append(fg_norm)
    
    # flip
    flip_axes = config_data.get('flip_axes', None)
    if(flip_axes is not None):
        random_flip = FlipLayer()
        random_flip.set_flip_dims(flip_axes)
        flip_mode = config_data.get('flip_mode', 'random')
        random_flip.set_flip_mode(flip_mode)
        pre_processor.append(random_flip)
    
    # convert labels
    label_convert_source = config_data.get('label_convert_source', None)
    label_convert_target = config_data.get('label_convert_target', None)
    if((label_convert_source is not None) and (label_convert_target is not None)):
        label_map = LabelMappingLayer()
        label_map.set_mapping_labels(label_convert_source, label_convert_target)
        pre_processor.append(label_map)

    # transpose
    transpose_view = config_data.get('transpose_view', None)
    if(transpose_view is not None):
        transpose = TransposeLayer(transpose_view = transpose_view)
        pre_processor.append(transpose)

    # rotate
    rotate_angle_range = config_data.get('rotate_angle_range', None)
    if (rotate_angle_range is not None):
        rotate = RotateLayer()
        rotate.set_angle_range(rotate_angle_range)
        pre_processor.append(rotate)
    return pre_processor

def train(config_file):
    train_mode = tf.estimator.ModeKeys.TRAIN
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']
    random.seed(config_train.get('random_seed', 1))

    # 2, setup data generator
    
    # create pre_processor and sampler
    pre_processor = get_brats_preprocess_layers(config_data, mode = train_mode)
    sampler = RandomSamplerWithCrop()
    sample_num_per_image = config_data.get('sample_num_per_image', 5)
    sample_shape = config_data['data_shape']
    sampler.set_sample_patch(sample_num_per_image, sample_shape)

    # create tensorflow dataset, iterator and initializer
    tr_data = DataSetFromTFRecord(config_data, mode = train_mode,
                                 preprocess_layers = pre_processor,
                                 sampler           = sampler)
    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_element = iterator.get_next()
    training_init_op = iterator.make_initializer(tr_data.data)


    # 2, construct graph
    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    class_num   = config_net['class_num']
    batch_size  = config_data['batch_size']
    full_data_shape  = [batch_size] + config_data['data_shape'] + \
                       [config_data['feature_channel_num']]
    full_label_shape = [batch_size] + config_data['label_shape'] + \
                       [config_data['prediction_channel_num']]
    x = tf.placeholder(tf.float32, shape = full_data_shape)
    w = tf.placeholder(tf.float32, shape = full_label_shape)
    y = tf.placeholder(tf.int32,   shape = full_label_shape)
   
    w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net_class = NetFactory.create(net_type)
    net = net_class(num_classes = class_num,
                    w_regularizer = w_regularizer,
                    b_regularizer = b_regularizer,
                    name = net_name)
    net.set_params(config_net)
    predicty = net(x, is_training = True)
    proby    = tf.nn.softmax(predicty)
    
#     loss_func = LossFunction(n_class=class_num)
#     loss = loss_func(predicty, y, weight_map = w)
#     print('size of predicty:',predicty)
    y_soft  = get_soft_label(y, class_num)
    loss = soft_dice_loss(predicty, y_soft, weight_map = w)
    
    # 3, initialize session and saver
    lr = config_train.get('learning_rate', 1e-3)
    opt_step = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.InteractiveSession()   
    sess.run(tf.global_variables_initializer())
    sess.run(training_init_op)
    saver = tf.train.Saver()

    # 4, start to train
    loss_file = config_train['model_save_prefix'] + "_loss.txt"
    start_it  = config_train.get('start_iteration', 0)
    if( start_it > 0):
        saver.restore(sess, config_train['model_pre_trained'])
    loss_list, temp_loss_list = [], []

    margin = int((config_data['data_shape'][0]  - config_data['label_shape'][0])/2)
    for n in range(start_it, config_train['maximal_iteration']):
        try:
            elem = sess.run(next_element)
            name = elem['entry_name']
            tempx = elem['entry_data']['feature']
            tempy = elem['entry_data']['prediction']
            tempw = elem['entry_data']['mask']
            if(margin > 0):
                tempy = tempy[:, margin:-margin, :, :, :]
                tempw = tempw[:, margin:-margin, :, :, :]
            opt_step.run(session = sess, feed_dict={x:tempx, w: tempw, y:tempy})
    
            if(n%config_train['test_iteration'] == 0):
                batch_dice_list = []
                for step in range(config_train['test_step']):
                    elem = sess.run(next_element)
                    tempx = elem['entry_data']['feature']
                    tempy = elem['entry_data']['prediction']
                    tempw = elem['entry_data']['mask']
                    if(margin > 0):
                        tempy = tempy[:, margin:-margin, :, :, :]
                        tempw = tempw[:, margin:-margin, :, :, :]
                    dice = loss.eval(feed_dict ={x:tempx, w:tempw, y:tempy})
                    batch_dice_list.append(dice)
                batch_dice = np.asarray(batch_dice_list, np.float32).mean()
                t = time.strftime('%X %x %Z')
                print(t, 'n', n,'loss', batch_dice)
                loss_list.append(batch_dice)
                np.savetxt(loss_file, np.asarray(loss_list))
    
            if((n+1)%config_train['snapshot_iteration']  == 0):
                saver.save(sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(n+1))
        except tf.errors.OutOfRangeError:
            sess.run(training_init_op)
    sess.close()
    
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train.py config17/train_wt_ax.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    train(config_file)
