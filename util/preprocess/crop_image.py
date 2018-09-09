# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from util.preprocess.base_layer import Layer


def extract_ndarray_subregion(input_array, begin, size):
    """
    extract a subregion from an nd array
    inputs:
        input_array: an nd numpy array
        begin      : an array with n values denoting the start extent of subregion
        size       : an array with n values denoting the size of subregion
    """
    end = begin + size
    if(input_array.ndim == 4):
        output = input_array[begin[0]:end[0],
                             begin[1]:end[1],
                             begin[2]:end[2], 
                             begin[3]:end[3]]
    else:
        raise ValueError("unsupported dimension {0:}".format(input_array.ndim))
    return output

def assign_ndarray_subregion(input_array, sub_region, begin):
    """
    assign a subregion of an nd array
    inputs:
        input_array: an nd numpy array
        sub_region : an nd numpy array denoting the context of the subregion
        begin      : an array with n values denoting the start extent of subregion
    """
    end = begin + np.asarray(sub_region.shape)
    if(input_array.ndim == 4):
        input_array[begin[0]:end[0],
                    begin[1]:end[1],
                    begin[2]:end[2], 
                    begin[3]:end[3]] = sub_region
    else:
        raise ValueError("unsupported dimension {0:}".format(input_array.ndim))
    return input_array

class CropImageToFixedSizeLayer(Layer):
    def __init__(self, name = 'crop_image', inversible = False):
        super(CropImageToFixedSizeLayer, self).__init__(name = name)
        self.name = name
        self.output_size = None
        self.parametres  = {}
        
    def get_class_name(self):
        return "CropImageToFixedSizeLayer"
    
    def set_crop_output_size(self, output_size):
        self.output_size = output_size
                
    def tf_operate(self, input_dict):
        assert(self.output_size is not None)
        assert(len(self.output_size) == 3)
        output_size = self.output_size + [0]
        mask = input_dict['entry_data']['mask']
        indices = tf.cast(tf.where(mask > 0), tf.int32)
        indices_min = tf.reduce_min(indices, reduction_indices=[0])
        indices_max = tf.reduce_max(indices, reduction_indices=[0])
        indices_central = (indices_min + indices_max)/2
        lower_half_size = [i/2 for i in output_size]

        crop_begin = tf.cast(indices_central - lower_half_size, tf.int32)
        crop_begin = tf.maximum(crop_begin, tf.zeros_like(crop_begin))
        crop_begin = tf.minimum(crop_begin, tf.shape(mask) - tf.constant(output_size))
        for field in input_dict['entry_data'].keys():
            field_value = input_dict['entry_data'][field]
            begin = crop_begin * tf.constant([1, 1, 1, 0])
            size  = tf.constant(output_size) + tf.constant([0,0,0,1])*tf.shape(field_value)
            input_dict['entry_data'][field] = tf.slice(field_value, begin, size)
         
        return input_dict
     
    def np_operate(self, input_dict):
        assert(self.output_size is not None)
        assert(len(self.output_size) == 3)
        output_size = np.asarray(self.output_size + [0])
        mask = input_dict['entry_data']['mask']
        indices = np.asarray(np.where(mask > 0))
        indices_min = np.ndarray.min(indices, axis=1)
        indices_max = np.ndarray.max(indices, axis=1)
        indices_central = (indices_min + indices_max)/2
        lower_half_size = [i/2 for i in output_size]
        crop_begin = np.asarray(indices_central - lower_half_size, np.int32)
        crop_begin = np.maximum(crop_begin, np.zeros_like(crop_begin))
        crop_begin = np.minimum(crop_begin, np.asarray(mask.shape) - output_size)
        self.parametres[input_dict['entry_name']] = [crop_begin, np.asarray(mask.shape)]
        for field in input_dict['entry_data'].keys():
            field_value = input_dict['entry_data'][field]
            begin = crop_begin * np.asarray([1, 1, 1, 0])
            size = output_size + np.asarray(field_value).shape*np.asarray([0, 0, 0, 1])
            input_dict['entry_data'][field] = \
                extract_ndarray_subregion(field_value, begin, size)
        return input_dict
            
    def inverse_np_operate(self, input_dict):
        [crop_begin, origin_shape] = self.parametres[input_dict['entry_name']]
        for field in input_dict['entry_data'].keys():
            field_value = input_dict['entry_data'][field]
            origin_shape[-1] = field_value.shape[-1]
            pad_volume = np.zeros(origin_shape, field_value.dtype)
            pad_volume = assign_ndarray_subregion(pad_volume, field_value, crop_begin)
            input_dict['entry_data'][field] = pad_volume
        return input_dict
        
