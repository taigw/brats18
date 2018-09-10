# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from util.preprocess.base_layer import Layer

class IntensityNormalizeLayer(Layer):
    def __init__(self, name = 'itensity_normalize', inversible = False):
        super(IntensityNormalizeLayer, self).__init__(name = name)
        self.name = name
        self.channel_number = None

    def get_class_name(self):
        return "CropImageToFixedSizeLayer"
    
    def set_channel_number(self, channel_number):
        self.channel_number = channel_number
        
    def tf_operate(self, input_dict):
        assert(self.channel_number is not None)
        img          = input_dict['entry_data']['feature']
        splitted_img = tf.split(img, self.channel_number, axis = -1)
        mask         = input_dict['entry_data']['mask']
        mask_reshape = tf.reshape(mask, [-1]) > 0
        
        normalized_channels = []
        for item in splitted_img:
            item_reshape   = tf.reshape(item, [-1])
            foreground     = tf.boolean_mask(item_reshape, mask_reshape)
            mean, variance = tf.nn.moments(foreground, [0])
            std            = tf.sqrt(variance)
            normalized     = (item - mean)/std
            
            rand_img   = tf.random_normal(tf.shape(item))
            normalized = normalized * mask + rand_img*(1.0 - mask)
            normalized_channels.append(normalized)
        normalized_channels = tf.concat(normalized_channels, axis = -1)
        input_dict['entry_data']['feature'] = normalized_channels
        return input_dict
    
    def np_operate(self, input_dict):
        assert(self.channel_number is not None) 
        img   = input_dict['entry_data']['feature']
        mask  = input_dict['entry_data']['mask']
        for chanl in range(img.shape[-1]):
            one_channel = img[:, :, :, chanl:chanl + 1]
            pixels = one_channel[mask > 0]
            mean = pixels.mean()
            std  = pixels.std()
            normalized = (one_channel - mean)/std
            rand_img   = np.random.normal(0.0, 1.0, size = mask.shape)
            normalized[mask==0] = rand_img[mask==0]
            img[:, :, :, chanl:chanl + 1] = normalized
        input_dict['entry_data']['feature'] = img
        return input_dict
            
