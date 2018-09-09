# -*- coding: utf-8 -*-

import tensorflow as tf

def crop_4D_images(input_tensor, begin, size, ignore_last_dim = True):
    """
    crop a 4d tensor with shahpe [D, H, W, C]
    begin and size are 1D tensors with lengh 4
    the last dim of begin and size are ignored if ingore_last_dim is True
    """
    input_shape = tf.shape(input_tensor)
    if(ignore_last_dim):
        begin = begin*tf.constant([1, 1, 1, 0])
        size  = size*tf.constant([1, 1, 1, 0]) + input_shape*tf.constant([0,0,0,1])
    crop = tf.slice(input_tensor, begin, size)
    return crop

def pad_4D_images_to_desired_shape(input_tensor, desire_shape,
            ignore_last_dim = True, pad_mode = 'random'):
    """
    pad a 4D tensor with shape [D, H, W, C]
    desire_shape: a list of [D1, H1, W1, C]（the last dimention is ignored）
    the output shape is the same as the input if desire shape is smaller than input shape
    pad_mode: random (random number with gaussian distribution) or constant value of 0
    """
    input_shape = tf.shape(input_tensor)
    if(ignore_last_dim):
        desire_shape = desire_shape*tf.constant([1, 1, 1, 0]) + input_shape*tf.constant([0,0,0,1])
    shape_sub  = tf.subtract(input_shape, desire_shape)
    flag = tf.cast(tf.less(shape_sub, tf.zeros_like(shape_sub)), tf.int32)
    flag = tf.scalar_mul(tf.constant(-1), flag)
    pad_all  = tf.multiply(shape_sub, flag)
    pad_all_p1  = tf.add(pad_all, tf.ones_like(pad_all))
    pad_l  = tf.scalar_mul(tf.constant(0.5), tf.cast(pad_all_p1, tf.float32))
    pad_l  = tf.cast(pad_l, tf.int32)
    pad_r  = pad_all - pad_l
    pad_lr = tf.stack([pad_l, pad_r], axis = 1)
    if(pad_mode == 'constant'):
        output_tensor = tf.pad(input_tensor, pad_lr)
    else:
        output_tensor = tf.pad(input_tensor, pad_lr, constant_values = 1000)
        mask = tf.equal(output_tensor, 1000*tf.ones_like(output_tensor))
        mask = tf.cast(mask, tf.float32)
        rand_tensor   = tf.random_normal(tf.shape(output_tensor))
        output_tensor = output_tensor * (1.0 - mask) + rand_tensor * mask
    return output_tensor

class RandomSamplerWithCrop():
    """
    sample a patch from dataset
    the image is croped by the bounding box of a mask
    then patches are sampled from the ROI
    if desired patch size is larger than ROI, regions outside the ROI are 
    filled with random number or zero
    """
    def __init__(self, name = 'random_sampler_with_crop'):
        self.name = name

    def set_sample_patch(self, patch_number, patch_size):
        self.patch_number = patch_number
        self.patch_size   = patch_size
        
    def get_output(self, input_dict):
        output = {'entry_name': tf.stack([input_dict['entry_name']]*self.patch_number)}
        output_entry_data = {}
        sample_begin_list = []
        input_shape = tf.shape(input_dict['entry_data']['mask'])
        if("sample_mask" in input_dict['entry_data'].keys()):
            sample_mask = input_dict['entry_data']["sample_mask"]
            indices = tf.cast(tf.where(sample_mask > 0), tf.int32)
            mask_idx_min = tf.reduce_min(indices, reduction_indices=[0])
            mask_idx_max = tf.reduce_max(indices, reduction_indices=[0])
        else:
            mask_idx_min = tf.zeros_like(input_shape)
            mask_idx_max = input_shape
        mask_size = mask_idx_max - mask_idx_min
        
        # crop based on sample_mask, and pad to desired shape
        for field in input_dict['entry_data'].keys():
            roi = crop_4D_images(input_dict['entry_data'][field], mask_idx_min, mask_size)
            pad_mode = 'random' if field == 'feature' else 'constant'
            roi = pad_4D_images_to_desired_shape(roi, self.patch_size + [0], pad_mode = pad_mode)
            input_dict['entry_data'][field] = roi
        sub_input_shape = tf.shape(input_dict['entry_data']['mask'])
        
        # set random sample index
        sample_begin_min = tf.zeros_like(sub_input_shape)
        sample_begin_max = sub_input_shape - tf.constant(self.patch_size + [0])
        for i in range(self.patch_number):
            data_shape_sub = sample_begin_max - sample_begin_min
            r = tf.random_uniform(tf.shape(data_shape_sub), 0, 1.0)
            sample_begin = tf.cast(data_shape_sub, tf.float32)*r
            sample_begin = tf.cast(sample_begin, tf.int32) + sample_begin_min
            sample_begin_list.append(sample_begin)
        for field in input_dict['entry_data'].keys():
            field_patches = []
            for sample_begin in sample_begin_list:
                patch_size = tf.constant(self.patch_size + [0])
                patch_size = patch_size + tf.constant([0,0,0,1])*tf.shape(input_dict['entry_data'][field])
                patch = tf.slice(input_dict['entry_data'][field], sample_begin, patch_size)
                field_patches.append(patch)
            field_patches = tf.stack(field_patches)
            output_entry_data[field] = field_patches
        output['entry_data'] = output_entry_data
        return output                
