# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from util.preprocess.base_layer import Layer

class LabelMappingLayer(Layer):
    def __init__(self, name = 'label_mapping', inversible = False):
        super(LabelMappingLayer, self).__init__(name = name)
        self.name       = name
        self.label_from = None
        self.label_to   = None

    def get_class_name(self):
        return "LabelMappingLayer"
    
    def set_mapping_labels(self, label_from, label_to):
        self.label_from = label_from
        self.label_to   = label_to
        
    def tf_operate(self, input_dict):
        assert(len(self.label_from) == len(self.label_to))
        if ("prediction" in input_dict['entry_data'].keys()):
            label = input_dict['entry_data']["prediction"]
            label_converted = tf.zeros_like(label)
            for i in range(len(self.label_from)):
                l0 = self.label_from[i]
                l1 = self.label_to[i]
                label_temp = tf.equal(label, tf.multiply(l0, tf.ones_like(label)))
                label_temp = tf.multiply(l1, tf.cast(label_temp,tf.int32))
                label_converted = tf.add(label_converted, label_temp)
            input_dict['entry_data']["prediction"] = label_converted
        return input_dict
    
    def np_operate(self, input_dict):
        assert(len(self.label_from) == len(self.label_to))
        if ("prediction" in input_dict['entry_data'].keys()):
            label = input_dict['entry_data']["prediction"]
            label_converted = np.zeros_like(label)
            for i in range(len(self.label_from)):
                l0 = self.label_from[i]
                l1 = self.label_to[i]
                label_temp = label == l0*np.ones_like(label)
                label_temp = l1*np.asarray(label_temp,np.int32)
                label_converted = label_converted + label_temp
            input_dict['entry_data']["prediction"] = label_converted
        return input_dict        
    
