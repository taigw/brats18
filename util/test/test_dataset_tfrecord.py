import os
import sys
sys.path.append("../")
import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from parse_config import parse_config
from test_dataset_pygenerator import get_preprocessors
from image_io.dataset_from_tfrecord import DataSetFromTFRecord
from preprocess.transpose import TransposeLayer
from preprocess.intensity_normalize import IntensityNormalizeLayer
from preprocess.foreground_mask import ForegroundMaskLayer
from preprocess.crop_image import CropImageToFixedSizeLayer
from preprocess.random_flip import RandomFlipLayer
from preprocess.sample_mask import SampleMaskLayer
from preprocess.label_mapping import LabelMappingLayer
from sampler.random_sampler_with_crop import RandomSamplerWithCrop

def train(config_file):
    train_mode = tf.estimator.ModeKeys.TRAIN
    config = parse_config(config_file)
    config_data       = config['data']
    
    # add preprocess layers
    pre_processor = get_preprocessors(config_data)
    
    # set sampler
    sampler = RandomSamplerWithCrop()
    sampler.set_sample_patch(10, [50, 100, 100])
    tr_data = DataSetFromTFRecord(config_data, mode = train_mode,
                                 preprocess_layers = pre_processor,
                                 sampler = sampler)
    
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
                y = elem['entry_data']['prediction']
                w = elem['entry_data']['mask']
                print(name, x.mean(),x.min(), x.max(), x.shape, w.sum())
                x_sub = x[0, :, :, :, 0]
                img = sitk.GetImageFromArray(x_sub)
                sitk.WriteImage(img, 'data/temp.nii.gz')
                x_sub = w[0, :, :, :, 0]
                img = sitk.GetImageFromArray(x_sub)
                sitk.WriteImage(img, 'data/temp_mask.nii.gz')
                x_sub = y[0, :, :, :, 0]
                img = sitk.GetImageFromArray(x_sub)
                sitk.WriteImage(img, 'data/temp_seg.nii.gz')
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break


if __name__ == '__main__':
    train("config/test_dataset_tfrecord.txt")
