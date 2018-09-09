# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
from preprocess.base_layer import Layer
class FlipLayer(Layer):
    def __init__(self, name = 'flip', inversible = False):
        self.name = name
        super(FlipLayer, self).__init__(name = name)
        self.flip_dims = None
        self.flip_mode = 'random'

    def get_class_name(self):
        return "FlipLayer"
    
    def set_flip_dims(self, dims):
        self.flip_dims = dims
    
    def set_flip_mode(self, flip_mode):
        assert(flip_mode in ['random', 'constant'])
        self.flip_mode = flip_mode
        
    def tf_operate(self, input_dict):
        assert(self.flip_dims is not None)
        tensor_list = []
        field_list  = []
        for field in input_dict['entry_data'].keys():
            tensor_list.append(input_dict['entry_data'][field])
            field_list.append(field)
        for d in self.flip_dims:
            if(self.flip_mode == 'random'):
                tensor_list = self._random_flip_tensors_in_one_dim(tensor_list, d)
            else:
                tensor_list = [tf.reverse(xi, tf.constant([d])) for xi in tensor_list]
        for i in range(len(tensor_list)):
            input_dict['entry_data'][field_list[i]] = tensor_list[i]
        return input_dict
    
    def np_operate(self, input_dict):
        assert(self.flip_dims is not None)
        array_list = []
        field_list = []
        for field in input_dict['entry_data'].keys():
            array_list.append(input_dict['entry_data'][field])
            field_list.append(field)
        for d in self.flip_dims:
            if(self.flip_mode == 'random'):
                array_list = self._random_flip_arrays_in_one_dim(array_list, d)
            else:
                array_list = [np.flip(xi, d) for xi in array_list]
        for i in range(len(array_list)):
            input_dict['entry_data'][field_list[i]] = array_list[i]
        return input_dict

    def inverse_np_operate(self, input_dict):
        assert(self.flip_mode == 'constant')
        for field in input_dict['entry_data'].keys():
            temp_array = input_dict['entry_data'][field]
            for d in self.flip_dims:
                temp_array = np.flip(temp_array, d)
            input_dict['entry_data'][field] = temp_array
        return input_dict

    def _random_flip_tensors_in_one_dim(self, tensors, d):
        """
        Random flip a list of tensors in one dimension
        input
            x: a list of tensors
            d: a integer denoting the axis
        output
            a list of flipped tensors
        """
        r = tf.random_uniform([1], 0, 1)
        r = tf.less(r, tf.constant(0.5))
        r = tf.cast(r, tf.int32)
        y = []
        for xi in tensors:
            xi_xiflip = tf.stack([xi, tf.reverse(xi, tf.constant([d]))])
            slice_begin = tf.concat([r, tf.zeros_like(tf.shape(xi))], -1)
            slice_size  = tf.concat([tf.constant([1]), tf.shape(xi)], -1)
            flip = tf.slice(xi_xiflip, slice_begin, slice_size)
            flip = tf.reshape(flip, tf.shape(xi))
            y.append(flip)
        return y
    
    def _random_flip_arrays_in_one_dim(self, arrays, d):
        ratio = random.random()
        if(ratio < 0.5):
            return arrays
        y = [np.flip(xi, d) for xi in arrays]
        return y
            
