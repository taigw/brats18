# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from preprocess.base_layer import Layer
class TransposeLayer(Layer):
    def __init__(self, name = 'transpose_layer', inversible = False, transpose_view = None):
        super(TransposeLayer, self).__init__(name = name)
        self.name = name
        self.transpose_view = transpose_view
    
    def get_class_name(self):
        return "TransposeLayer"
        
    def set_transpose_view(self, transpose_view):
        self.transpose_view = transpose_view
        
    def tf_operate(self, input_dict):
        assert(self.transpose_view is not None)
        assert(self.transpose_view in ['axial', 'sagittal', 'coronal'])
        for field in input_dict['entry_data'].keys():
            field_value = input_dict['entry_data'][field]
            if(self.transpose_view == 'axial'):
                pass
            elif(self.transpose_view == 'sagittal'):
                perm = [2, 0, 1, 3]
                input_dict['entry_data'][field] = tf.transpose(field_value, perm)
            else:
                perm = [1, 0, 2, 3]
                input_dict['entry_data'][field] = tf.transpose(field_value, perm)
        return input_dict

    def np_operate(self, input_dict):
        assert(self.transpose_view is not None)
        assert(self.transpose_view in ['axial', 'sagittal', 'coronal'])
        for field in input_dict['entry_data'].keys():
            field_value = input_dict['entry_data'][field]
            if(self.transpose_view == 'axial'):
                pass
            elif(self.transpose_view == 'sagittal'):
                perm = [2, 0, 1, 3]
                input_dict['entry_data'][field] = np.transpose(field_value, perm)
            else:
                perm = [1, 0, 2, 3]
                input_dict['entry_data'][field] = np.transpose(field_value, perm)
        return input_dict

    def inverse_np_operate(self, input_dict):
        assert(self.transpose_view is not None)
        assert(self.transpose_view in ['axial', 'sagittal', 'coronal'])
        for field in input_dict['entry_data'].keys():
            field_value = input_dict['entry_data'][field]
            if(self.transpose_view == 'axial'):
                pass
            elif(self.transpose_view == 'sagittal'):
                perm = [1, 2, 0, 3]
                input_dict['entry_data'][field] = np.transpose(field_value, perm)
            else:
                perm = [1, 0, 2, 3]
                input_dict['entry_data'][field] = np.transpose(field_value, perm)
        return input_dict