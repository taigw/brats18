import os
import sys
sys.path.append('../')
import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from image_io.dataset_from_pygenerator import DataSetFromPyGenerator
from parse_config import parse_config
from preprocess.transpose import TransposeLayer
from preprocess.intensity_normalize import IntensityNormalizeLayer
from preprocess.foreground_mask import ForegroundMaskLayer
from preprocess.crop_image import CropImageToFixedSizeLayer
from preprocess.random_flip import RandomFlipLayer
from preprocess.sample_mask import SampleMaskLayer
from preprocess.label_mapping import LabelMappingLayer
from sampler.random_sampler import RandomSampler

def get_preprocessors(config_data):
    # add preprocess layers
    pre_processor = []
    
    # 1, get foreground mask
    fg_mask = ForegroundMaskLayer()
    pre_processor.append(fg_mask)
    
    # 2, crop the image to fixed size 
    crop_img = CropImageToFixedSizeLayer()
    crop_img.set_crop_output_size(config_data['crop_size'])
    pre_processor.append(crop_img)
    
    # 3, intensity normalize 
    fg_norm = IntensityNormalizeLayer()
    if('feature_channel_num' in config_data.keys()):
        feature_channel_num = config_data['feature_channel_num']
    else:
        feature_channel_num = len(config_data['feature_channel_names'])
    fg_norm.set_channel_number(feature_channel_num)
    pre_processor.append(fg_norm)

    # 4, random flip
    random_flip = RandomFlipLayer()
    random_flip.set_flip_dims(config_data['flip_axes'])
    pre_processor.append(random_flip)

    # get sample mask
    sample_mask = SampleMaskLayer()
    sample_mask.set_mask_labels(config_data['sample_mask_labels'])
    pre_processor.append(sample_mask)
    
    # 5, label map 
    label_convert_source = config_data.get('label_convert_source', None)
    label_convert_target = config_data.get('label_convert_target', None)
    if((label_convert_source is not None) and (label_convert_target is not None)):
        label_map = LabelMappingLayer()
        label_map.set_mapping_labels(label_convert_source, label_convert_target)
        pre_processor.append(label_map)

    
    # transpose
    transpose_view = config_data.get('transpose_view', None)
    if(transpose_view is not None):
        transpose = TransposeLayer(transpose_view = transpose_view)
        pre_processor.append(transpose)
    
    return pre_processor

def train(config_file):
    train_mode = tf.estimator.ModeKeys.TRAIN
    config = parse_config(config_file)
    config_data       = config['data']
    
    pre_processor = get_preprocessors(config_data)
    # set sampler
    sampler = RandomSampler()
    sample_num_per_image = config_data.get('sample_num_per_image', 5)
    sample_shape = config_data['data_shape']
    sampler.set_sample_patch(sample_num_per_image, sample_shape)
    tr_data = DataSetFromPyGenerator(config_data, mode = train_mode,
                                 preprocess_layers = pre_processor,
                                 sampler = sampler)
    tr_data.get_tf_data()
    
    # create TensorFlow Iterator object
    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,
                                   tr_data.data.output_shapes)
    next_element = iterator.get_next()

    # create two initialization ops to switch between the datasets
    training_init_op = iterator.make_initializer(tr_data.data)

    with tf.Session() as sess:

        # initialize the iterator on the training data
        sess.run(training_init_op)

        # get each element of the training dataset until the end is reached
        for i in range(2):
            try:
                elem = sess.run(next_element)
                name = elem['entry_name']
                x = elem['entry_data']['feature']
                w = elem['entry_data']['mask']
                print(name, x.mean(),x.min(), x.max(), x.shape, w.sum())
                x_sub = x[0, :, :, :, 0]
                img = sitk.GetImageFromArray(x_sub)
                sitk.WriteImage(img, 'data/temp.nii.gz')
                x_sub = w[0, :, :, :, 0]
                img = sitk.GetImageFromArray(x_sub)
                sitk.WriteImage(img, 'data/temp_mask.nii.gz')
                if ('prediction' in elem['entry_data'].keys()):
                    y = elem['entry_data']['prediction']
                    x_sub = y[0, :, :, :, 0]
                    img = sitk.GetImageFromArray(x_sub)
                    sitk.WriteImage(img, 'data/temp_seg.nii.gz')
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break


if __name__ == '__main__':
    train("config/test_dataset_pygenerator.txt")
