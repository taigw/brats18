# -*- coding: utf-8 -*-

import tensorflow as tf        
class RandomSampler():
    def __init__(self, name = 'random_sampler'):
        self.name = name

    def set_sample_patch(self, patch_number, patch_size):
        self.patch_number = patch_number
        self.patch_size   = patch_size
        
    def get_output(self, input_dict):
        output = {'entry_name': tf.stack([input_dict['entry_name']]*self.patch_number)}
        output_entry_data = {}
        crop_begin_list = []
        input_shape = tf.shape(input_dict['entry_data']['mask'])
        if("sample_mask" in input_dict['entry_data'].keys()):
            sample_mask = input_dict['entry_data']["sample_mask"]
            indices = tf.cast(tf.where(sample_mask > 0), tf.int32)
            mask_idx_min = tf.reduce_min(indices, reduction_indices=[0])
            mask_idx_max = tf.reduce_max(indices, reduction_indices=[0])
        else:
            mask_idx_min = tf.zeros_like(input_shape)
            mask_idx_max = input_shape
        crop_begin_min_temp = mask_idx_min
        crop_begin_max_temp = mask_idx_max - tf.constant(self.patch_size + [0])
        crop_begin_min = tf.minimum(crop_begin_min_temp, crop_begin_max_temp)
        crop_begin_max = tf.maximum(crop_begin_min_temp, crop_begin_max_temp)
        crop_begin_min = tf.maximum(crop_begin_min, tf.zeros_like(input_shape))
        crop_begin_max = tf.minimum(crop_begin_max, input_shape - tf.constant(self.patch_size + [0]))
        for i in range(self.patch_number):
            data_shape_sub = crop_begin_max - crop_begin_min
            r = tf.random_uniform(tf.shape(data_shape_sub), 0, 1.0)
            crop_begin = tf.cast(data_shape_sub, tf.float32)*r 
            crop_begin = tf.cast(crop_begin, tf.int32) + crop_begin_min
            crop_begin_list.append(crop_begin)
        for field in input_dict['entry_data'].keys():
            field_patches = []
            for crop_begin in crop_begin_list:
                patch_size = tf.constant(self.patch_size + [0])
                patch_size = patch_size + tf.constant([0,0,0,1])*tf.shape(input_dict['entry_data'][field])
                patch = tf.slice(input_dict['entry_data'][field], crop_begin, patch_size)
                field_patches.append(patch)
            field_patches = tf.stack(field_patches)
            output_entry_data[field] = field_patches
        output['entry_data'] = output_entry_data
        return output                