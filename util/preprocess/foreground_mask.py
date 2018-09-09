# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from preprocess.base_layer import Layer
class ForegroundMaskLayer(Layer):
    def __init__(self, name = 'foreground_mask', inversible = False):
        self.name = name
        super(ForegroundMaskLayer, self).__init__(name = name)

    def get_class_name(self):
        return "ForegroundMaskLayer"
    
    def tf_operate(self, input_dict):
        img = input_dict['entry_data']['feature']
        img_reduce_sum = tf.reduce_sum(img, axis = -1, keep_dims = True)
        mask = tf.cast(img_reduce_sum > 0, tf.float32)
        input_dict['entry_data']['mask'] = mask
        return input_dict

    def np_operate(self, input_dict):
        img = input_dict['entry_data']['feature']
        img_reduce_sum = np.sum(img, axis = -1, keepdims = True) 
        mask = np.asarray(img_reduce_sum > 0, np.float32)
        input_dict['entry_data']['mask'] = mask
        return input_dict