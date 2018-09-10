# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
from scipy import ndimage
from util.preprocess.base_layer import Layer
class RotateLayer(Layer):
    def __init__(self, name = 'rotate', inversible = False):
        self.name = name
        super(RotateLayer, self).__init__(name = name)
        self.rotate_angle = 0
        self.parameters = {}
        
    def get_class_name(self):
        return "RotateLayer"
    
    def set_angle_range(self, angle_range):
        self.angle_range = angle_range
    
    def tf_operate(self, input_dict):
        rotate_angle = tf.random_uniform([1], self.angle_range[0], self.angle_range[1])
        for field in input_dict['entry_data'].keys():
            temp_tensor = input_dict['entry_data'][field]
            origin_shape = temp_tensor.get_shape().as_list()
            new_shape = [origin_shape[0]*origin_shape[1]] + origin_shape[2:]
            temp_tensor_reshape = tf.reshape(temp_tensor, new_shape)
            interp = 'BILINEAR' if field == 'feature' else 'NEAREST'
            rotat_tensor_reshape = tf.contrib.image.rotate(
                                      temp_tensor_reshape,
                                      rotate_angle, interpolation= interp)
            rotat_tensor = tf.reshape(rotat_tensor_reshape, temp_tensor.shape)
            input_dict['entry_data'][field] = rotat_tensor
        return input_dict
    
    def np_operate(self, input_dict):
        rotate_angle = random.uniform(self.angle_range[0], self.angle_range[1])
        self.parameters[input_dict['entry_name']] = rotate_angle
        for field in input_dict['entry_data'].keys():
            temp_array = input_dict['entry_data'][field]
            interp_order = 1 if field == 'feature' else 0
            rotate_array = ndimage.interpolation.rotate(temp_array, rotate_angle, axes = (1, 2),
                            reshape = False, order =interp_order)
            input_dict['entry_data'][field] = rotate_array
        return input_dict

    def inverse_np_operate(self, input_dict):
        rotate_angle = - self.parameters[input_dict['entry_name']]
        for field in input_dict['entry_data'].keys():
            temp_array = input_dict['entry_data'][field]
            interp_order = 1 if field == 'feature' else 0
            rotate_array = ndimage.interpolation.rotate(temp_array, rotate_angle, axes = (1, 2),
                            reshape = False, order =interp_order)
            input_dict['entry_data'][field] = rotate_array
        return input_dict

