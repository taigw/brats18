# Load image files as an iterator, and optionally save it as a tfrecord file
#
# Author: Guotai Wang
# 17 July 2018
# Reference for tfrecord:
# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
import os
import numpy as np
import SimpleITK as sitk
from random import shuffle
import tensorflow as tf

class DataSetFromPyGenerator():
    def __init__(self, config, mode = tf.estimator.ModeKeys.TRAIN,
                 preprocess_layers = None, sampler = None):
        self.mode   = mode
        self.config = config
        self.data_dir_list            = config['data_dir_list']
        assert(type(self.data_dir_list) is list)
        self.shuffle                  = config.get('shuffle', False)
        self.entry_names              = self._get_entry_names(config['entry_names_file'])
        self.entry_data_fields        = config['entry_data_fields']

        self.feature_channel_names    = config['feature_channel_names']
        self.prediction_channel_names = config.get('prediction_channel_names', None)
        self.feature_data_type        = config['feature_data_type']
        self.prediction_data_type     = config.get('prediction_data_type', None)
        self.output_tfrecord_name     = config.get('output_tfrecord_name', None)
        self.preprocess_needed = False
        self.preprocess_layers = preprocess_layers
        self.sampler    = sampler
        self.batch_size = config.get('batch_size', 2)
        
    def _get_entry_names(self, entry_names_file):
        with open(entry_names_file) as f:
            raw_entry_names = f.readlines()
        entry_names = [item.strip() for item in raw_entry_names if (len(item) > 3)]
        if(self.shuffle):
            shuffle(entry_names)
        assert(len(entry_names) > 0)
        return entry_names
    
    def _get_np_datatype_by_name(self, type_name):
        if(type_name == 'float32'):
            return np.float32
        elif(type_name == 'int32'):
            return np.int32
        else:
            raise ValueError("Invalid data type {0:}".format(type_name))
    
    def _get_tf_datatype_by_name(self, type_name):
        if(type_name == 'float32'):
            return tf.float32
        elif(type_name == 'int32'):
            return tf.int32
        else:
            raise ValueError("Invalid data type {0:}".format(type_name))        
        
    def _read_one_field_for_one_patient(self, field_channels, entry_name):
        volume_list = []
        for channel_name in field_channels:
            volume_file = "{0:}_{1:}.nii.gz".format(entry_name, channel_name)
            file_exist = False
            for data_dir in self.data_dir_list:
                volume_full_file = "{0:}/{1:}".format(data_dir, volume_file)
                if(os.path.exists(volume_full_file)):
                    file_exist = True
                    break
            if(file_exist == False):
                raise ValueError("{0:} was not found".format(volume_file))
            img_obj   = sitk.ReadImage(volume_full_file)
            img_array = sitk.GetArrayFromImage(img_obj)
            volume_list.append(img_array)
        volume = np.stack(volume_list, axis = -1)
        return volume
    
    def get_data_generator(self):
        for entry_name in self.entry_names:
            output = {"entry_name":entry_name.encode()}
            entry_data = {}
            for one_field in self.entry_data_fields:
                if(one_field == "feature"):
                    field_data = self._read_one_field_for_one_patient(
                                        self.feature_channel_names, entry_name)
                    field_data_type = self._get_np_datatype_by_name(self.feature_data_type)
                elif(one_field == "prediction"):
                    field_data = self._read_one_field_for_one_patient(
                                        self.prediction_channel_names, entry_name)
                    field_data_type = self._get_np_datatype_by_name(self.prediction_data_type) 
                else:
                    raise ValueError("Invalid entry data field {}".format(one_field))
                entry_data[one_field] = field_data = np.asarray(field_data, field_data_type)
            output["entry_data"] = entry_data 
            if(self.preprocess_needed and (self.preprocess_layers is not None)):
                for preprocessor in self.preprocess_layers:
                    if(hasattr(preprocessor, 'np_operate')):
                        output = preprocessor.np_operate(output)       
            yield output
            
    def get_tf_data(self):
        # set datatype for gdata generator
        self.preprocess_needed = True
        data_type = {'entry_name': tf.string}
        data_type['entry_data'] =  {}
        if('feature' in self.entry_data_fields):
            data_type['entry_data']['feature'] = \
                self._get_tf_datatype_by_name(self.feature_data_type)
        if('prediction' in self.entry_data_fields):
            data_type['entry_data']['prediction'] = \
                self._get_tf_datatype_by_name(self.prediction_data_type)
        if(self.preprocess_needed  and (self.preprocess_layers is not None)):
            for preprocess in self.preprocess_layers:
                if(preprocess.get_class_name() == "ForegroundMaskLayer"):
                    data_type['entry_data']['mask'] = tf.float32
                elif(preprocess.get_class_name() == "SampleMaskLayer"):
                    data_type['entry_data']['sample_mask'] = tf.float32
        data = tf.data.Dataset.from_generator(self.get_data_generator, 
                                              output_types = data_type)
        
        # get sampled patches
        if(self.sampler is not None):
            data = data.map(self.sampler.get_output, num_parallel_calls = self.batch_size)
            data = data.apply(tf.contrib.data.unbatch())
 
        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=20*self.batch_size)

        # create a new dataset with batches of images
        # https://stackoverflow.com/questions/48777889/tf-data-api-how-to-efficiently-sample-small-patches-from-images
        if(self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == tf.estimator.ModeKeys.EVAL):
            data = data.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        else:
            data = data.batch(self.batch_size)
        data = data.prefetch(20*self.batch_size)
        self.data = data

    def save_to_tfrecords(self): 
        assert(self.output_tfrecord_name is not None)
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        tfrecord_options= tf.python_io.TFRecordOptions(1)
        writer = tf.python_io.TFRecordWriter(self.output_tfrecord_name , tfrecord_options)
        self.preprocess_needed = False
        data_generator = self.get_data_generator()
        for item in data_generator:
            entry_name = item["entry_name"]
            entry_data = item["entry_data"]
            entry_fields = entry_data.keys()
            tfrecord_item_dict = {"entry_name":_bytes_feature(entry_name)}
            print(entry_name)
            for field in entry_fields:
                field_data_key  = field + "_raw"
                field_shape_key = field + "_shape_raw"
                field_data_raw       = entry_data[field].tostring()
                field_data_shape_raw = np.asarray(entry_data[field].shape, np.int32).tostring()
                tfrecord_item_dict[field_data_key]  = _bytes_feature(field_data_raw)
                tfrecord_item_dict[field_shape_key] = _bytes_feature(field_data_shape_raw)
            example = tf.train.Example(features=tf.train.Features(feature = tfrecord_item_dict))
            writer.write(example.SerializeToString())
        writer.close()
