# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from util.preprocess.base_layer import Layer
class SampleMaskLayer(Layer):
    def __init__(self, name = 'sample_mask', inversible = False):
        self.name = name
        super(SampleMaskLayer, self).__init__(name = name)
        self.mask_labels = None
    
    def get_class_name(self):
        return "SampleMaskLayer"
    
    def set_mask_labels(self, mask_labels):
        self.mask_labels = mask_labels
        
    def tf_operate(self, input_dict):
        assert((self.mask_labels is not None) and len(self.mask_labels) > 0)
        assert('prediction' in input_dict['entry_data'].keys())
        predictions = input_dict['entry_data']['prediction']
        sample_mask = tf.zeros_like(predictions)
        for lab in self.mask_labels:
            temp_mask = tf.cast(tf.equal(predictions, lab*tf.ones_like(predictions)), tf.int32)
            sample_mask = sample_mask + temp_mask
        input_dict['entry_data']['sample_mask'] = sample_mask
        return input_dict

    def np_operate(self, input_dict):
        assert((self.mask_labels is not None) and len(self.mask_labels) > 0)
        assert('prediction' in input_dict['entry_data'].keys())
        predictions = input_dict['entry_data']['prediction']
        sample_mask = np.zeros_like(predictions)
        for lab in self.mask_labels:
            temp_mask = np.asarray(predictions == lab*np.ones_like(predictions), np.int32)
            sample_mask = sample_mask + temp_mask
        input_dict['entry_data']['sample_mask'] = sample_mask
        return input_dict
