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
from random import shuffle
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

class DataSetFromTFRecord(object):
    """Wrapper class around the new Tensorflows dataset pipeline.
    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, config, 
                 mode = tf.estimator.ModeKeys.TRAIN,
                 preprocess_layers = None, sampler = None):
        """
        Read images from tf_record files, preprocess the images and generate samples
        that will be fed into networks.
        Args:
            config: a dictionay with configure parameters
            preprocess_layers: a list of layers for preprcessing
        """
        
        self.mode        = mode
        self.shuffle     = config.get('shuffle', True)
        self.batch_size  = config.get('batch_size', 2)
        if(self.mode == tf.estimator.ModeKeys.PREDICT):
            self.batch_size = 1
        self.buffer_size = config.get('buffer_size', 10*self.batch_size)
        self.input_tfrecord_name  = config['input_tfrecord_name']
        self.entry_data_fields    = config['entry_data_fields'] 
        self.feature_data_type    = config['feature_data_type'] 
        self.prediction_data_type = config.get('prediction_data_type', None)
        self.preprocess_layers = preprocess_layers
        self.sampler           = sampler

        
        # read tfrecord dataset and preprocess
        data = tf.data.TFRecordDataset(self.input_tfrecord_name,"ZLIB")
        data = data.map(self._parse_function, num_parallel_calls = self.batch_size)
        
        # get sampled patches
        if(self.sampler is not None):
            data = data.map(self.sampler.get_output, num_parallel_calls = self.batch_size)
            data = data.apply(tf.contrib.data.unbatch())
 
        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=self.buffer_size)

        # create a new dataset with batches of images
        # https://stackoverflow.com/questions/48777889/tf-data-api-how-to-efficiently-sample-small-patches-from-images
        
        #data = data.batch(self.batch_size)
        data = data.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        data = data.prefetch(self.buffer_size)
        self.data = data

    def _get_datatype_by_name(self, type_name):
        if(type_name == 'float32'):
            return tf.float32
        elif(type_name == 'int32'):
            return tf.int32
        else:
            raise ValueError("Invalid data type {0:}".format(type_name))
        
    def _parse_function(self, item):
        keys_to_features = {'entry_name': tf.FixedLenFeature((), tf.string)}
        for field in self.entry_data_fields:
            field_raw_key = field + "_raw"
            field_shape_raw_key = field + "_shape_raw"
            keys_to_features[field_raw_key]       = tf.FixedLenFeature((), tf.string)
            keys_to_features[field_shape_raw_key] = tf.FixedLenFeature((), tf.string)
        
        # parse the data
        parsed_features   = tf.parse_single_example(item, keys_to_features)
        output = {'entry_name': parsed_features['entry_name']}
        entry_data = {}
        for field in self.entry_data_fields:
            field_raw_key = field + "_raw"
            field_shape_raw_key = field + "_shape_raw"
            field_data_shape = tf.decode_raw(parsed_features[field_shape_raw_key],  tf.int32)
            field_data_shape = tf.reshape(field_data_shape,  [4])
            if(field == 'feature'):
                field_data_type = self._get_datatype_by_name(self.feature_data_type)
            elif(field == 'prediction'):
                field_data_type = self._get_datatype_by_name(self.prediction_data_type)
            else:
                raise ValueError('Invalid entry data field {0:}'.format(field))
            field_data = tf.decode_raw(parsed_features[field_raw_key], field_data_type)
            field_data = tf.reshape(field_data, field_data_shape)
            entry_data[field] = field_data
        output['entry_data'] = entry_data
        
        # pre process
        if((self.preprocess_layers is not None) and (len(self.preprocess_layers) > 0)):
            for one_layer in self.preprocess_layers:
                output = one_layer.tf_operate(output)
        for field in list(output['entry_data'].keys()):
            assert(field in ['feature', 'prediction', 'mask', 'sample_mask'])
        return output

        
