# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function
import numpy as np
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
from util.image_io.dataset_from_pygenerator import DataSetFromPyGenerator
from util.train_test_func import *
from util.parse_config import parse_config
from train import NetFactory, get_brats_preprocess_layers

class BratsTest:
    def __init__(self, config_file_name):
        # 1, load configure file
        config = parse_config(config_file_name)
        self.config = config
        self.batch_size = config['testing'].get('batch_size', 5)
        self.roi_margin = config['testing'].get('roi_patch_margin', 5)
        self.__construct_networks()
        self.__creat_session_and_load_variables()
    
    def __construct_networks(self):
        self.x_shape = tf.placeholder(tf.int32, shape = 3)
        [self.x1_ax, self.net1_ax, self.p1_ax] = self.__construct_one_network(self.config['network1ax'])
        [self.x1_sg, self.net1_sg, self.p1_sg] = self.__construct_one_network(self.config['network1sg'])
        [self.x1_cr, self.net1_cr, self.p1_cr] = self.__construct_one_network(self.config['network1cr'])
        
        [self.x2_ax, self.net2_ax, self.p2_ax] = self.__construct_one_network(self.config['network2ax'])
        [self.x2_sg, self.net2_sg, self.p2_sg] = self.__construct_one_network(self.config['network2sg'])
        [self.x2_cr, self.net2_cr, self.p2_cr] = self.__construct_one_network(self.config['network2cr'])
        
        [self.x3_ax, self.net3_ax, self.p3_ax] = self.__construct_one_network(self.config['network3ax'])
        [self.x3_sg, self.net3_sg, self.p3_sg] = self.__construct_one_network(self.config['network3sg'])
        [self.x3_cr, self.net3_cr, self.p3_cr] = self.__construct_one_network(self.config['network3cr'])
    
    def __creat_session_and_load_variables(self):
        all_vars = tf.global_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        net_configs = [self.config['network1ax'], self.config['network1sg'], self.config['network1cr'],
                       self.config['network2ax'], self.config['network2sg'], self.config['network2cr'],
                       self.config['network3ax'], self.config['network3sg'], self.config['network3cr']]
        for net_cfg in net_configs:
            net_name   = net_cfg['net_name']
            model_file = net_cfg['model_file']
            net_vars = [x for x in all_vars if x.name[0:len(net_name) + 1] == net_name + '/']
            saver = tf.train.Saver(net_vars)
            saver.restore(self.sess, model_file)
    
    def __construct_one_network(self, config_net):
        net_type    = config_net['net_type']
        net_name    = config_net['net_name']
        data_shape  = config_net['data_shape']
        label_shape = config_net['label_shape']
        class_num   = config_net['class_num']
        
        full_data_shape = [self.batch_size, data_shape[0], None, None, data_shape[-1]]
        x = tf.placeholder(tf.float32, shape = full_data_shape)
        net_class = NetFactory.create(net_type)
        net = net_class(num_classes = class_num,w_regularizer = None,
                    b_regularizer = None, name = net_name)
        net.set_params(config_net)
        predicty = net(x, is_training = True, input_image_shape =  self.x_shape)
        proby    = tf.nn.softmax(predicty)
        return [x, net, proby]

    def __get_result_of_one_tissue(self, tissue, imgs, mask):
        if(tissue == 'whole_tumor'):
            data_shapes  = [ self.config['network1ax']['data_shape'][:-1],
                             self.config['network1sg']['data_shape'][:-1],
                             self.config['network1cr']['data_shape'][:-1]]
            label_shapes = [ self.config['network1ax']['label_shape'][:-1],
                             self.config['network1sg']['label_shape'][:-1],
                             self.config['network1cr']['label_shape'][:-1]]
            p_list   = [self.p1_ax, self.p1_sg, self.p1_cr]
            x_list   = [self.x1_ax, self.x1_sg, self.x1_cr]
            class_num = self.config['network1ax']['class_num']
        elif(tissue == 'tumor_core'):
            data_shapes  = [ self.config['network2ax']['data_shape'][:-1],
                             self.config['network2sg']['data_shape'][:-1],
                             self.config['network2cr']['data_shape'][:-1]]
            label_shapes = [ self.config['network2ax']['label_shape'][:-1],
                             self.config['network2sg']['label_shape'][:-1],
                             self.config['network2cr']['label_shape'][:-1]]
            p_list   = [self.p2_ax, self.p2_sg, self.p2_cr]
            x_list   = [self.x2_ax, self.x2_sg, self.x2_cr]
            class_num = self.config['network2ax']['class_num']
        elif(tissue == 'enhancing_core'):
            data_shapes  = [ self.config['network3ax']['data_shape'][:-1],
                             self.config['network3sg']['data_shape'][:-1],
                             self.config['network3cr']['data_shape'][:-1]]
            label_shapes = [ self.config['network3ax']['label_shape'][:-1],
                             self.config['network3sg']['label_shape'][:-1],
                             self.config['network3cr']['label_shape'][:-1]]
            p_list   = [self.p3_ax, self.p3_sg, self.p3_cr]
            x_list   = [self.x3_ax, self.x3_sg, self.x3_cr]
            class_num = self.config['network3ax']['class_num']
        else:
            raise ValueError("invalid tissue type: {0:}".format(tissue))
        prob = test_one_image_with_three_nets(imgs, data_shapes, label_shapes, 4, class_num,
                       self.batch_size, self.sess, p_list, x_list, self.x_shape)
        pred = np.asarray(np.argmax(prob, axis = 3), np.uint16)
        pred = pred * mask
        return pred

    def __inference_one_case(self, imgs, weight):
        struct = ndimage.generate_binary_structure(3, 2)
        pred1 = self.__get_result_of_one_tissue('whole_tumor', imgs, weight)
        pred1, pred1_componets = get_largest_two_components(pred1, 2000)
        out_label = np.zeros_like(pred1)
        for pred1_component in pred1_componets:
            temp_label1 = pred1_component
            temp_label2 = np.zeros_like(temp_label1)
            temp_label3 = np.zeros_like(temp_label1)
            bbox1 = get_ND_bounding_box(pred1_component, self.roi_margin)
            sub_imgs = [crop_ND_volume_with_bounding_box(one_img, bbox1[0], bbox1[1]) for one_img in imgs]
            sub_weight = crop_ND_volume_with_bounding_box(weight, bbox1[0], bbox1[1])
            pred2 = self.__get_result_of_one_tissue('tumor_core', sub_imgs, sub_weight)
            pred2, pred2_componets = get_largest_two_components(pred2, 1000)
            
            temp_label3_roi = np.zeros_like(pred2)
            for pred2_componet in pred2_componets:
                bbox2 = get_ND_bounding_box(pred2_componet, self.roi_margin)
                subsub_imgs = [crop_ND_volume_with_bounding_box(one_img, bbox2[0], bbox2[1]) \
                                for one_img in sub_imgs]
                subsub_weight = crop_ND_volume_with_bounding_box(sub_weight, bbox2[0], bbox2[1])
                pred3 = self.__get_result_of_one_tissue('enhancing_core', subsub_imgs, subsub_weight)
                if(pred3.sum() < 30):
                    pred3 = np.zeros_like(pred3)
                temp_label3_roi = set_ND_volume_roi_with_bounding_box_range( \
                                    temp_label3_roi, bbox2[0], bbox2[1], pred3)
            temp_label2 = set_ND_volume_roi_with_bounding_box_range(temp_label2, bbox1[0], bbox1[1], pred2)
            temp_label3 = set_ND_volume_roi_with_bounding_box_range(temp_label3, bbox1[0], bbox1[1], temp_label3_roi)
            
            # fuse results at 3 levels
            temp_label1_mask = (temp_label1 + temp_label2 + temp_label3) > 0
            temp_label1_mask = ndimage.morphology.binary_closing(temp_label1_mask, structure = struct)
            temp_label1_mask, temp_label1_mask_components = get_largest_two_components(temp_label1_mask, 2000)
            if(len(temp_label1_mask_components) > 1):
                temp_label1_mask = temp_label1_mask_components[-1]

            temp_label2_mask = (temp_label2 + temp_label3) > 0
            temp_label2_mask = remove_external_core(temp_label1, temp_label2_mask)

            temp_label2_mask = temp_label2_mask * temp_label1_mask
            temp_label3 = temp_label3 * temp_label2_mask
            out_label[temp_label1_mask > 0] = 2
            out_label[temp_label2_mask > 0] = 1
            out_label[temp_label3 > 0] = 4
        return out_label

    def inference(self):
        # setup data set
        config_data = self.config['data']
        test_mode = tf.estimator.ModeKeys.PREDICT
        pre_processor = get_brats_preprocess_layers(config_data, test_mode)
        test_data = DataSetFromPyGenerator(config_data,
                                       mode = tf.estimator.ModeKeys.PREDICT,
                                       preprocess_layers = pre_processor,
                                       sampler = None)
        test_data.get_tf_data()
        iterator = tf.data.Iterator.from_structure(test_data.data.output_types,
                                   test_data.data.output_shapes)
        next_element = iterator.get_next()
        test_init_op = iterator.make_initializer(test_data.data)
        self.sess.run(test_init_op)
        
        # start to test
        save_folder = config_data['save_folder']
        test_time = []
        num = 0
        
        while True:
            try:
                elem = self.sess.run(next_element)
                entry_num = len(elem['entry_name'])
                for i in range(entry_num):
                    name = elem['entry_name'][i]
                    x    = elem['entry_data']['feature'][i]
                    mask = elem['entry_data']['mask'][i]
                    imgs    = [x[:, :, :, 0], x[:, :, :, 1], x[:, :, :, 2], x[:, :, :, 3]]
                    weight  =  mask[:, :, :, 0]
                    t0 = time.time()
                    predict = self.__inference_one_case(imgs, weight)
                    predict = np.reshape(predict, list(predict.shape) + [1])
                    input_dict = {'entry_name': name}
                    input_dict['entry_data'] = {'prediction': predict}
                    for layer_idx in list(reversed(range(len(pre_processor)))):
                        if(hasattr(pre_processor[layer_idx], 'inverse_np_operate')):
                            input_dict = pre_processor[layer_idx].inverse_np_operate(input_dict)
                    predict = input_dict['entry_data']['prediction']
                    predict = np.reshape(predict, predict.shape[:-1])
                    time_i = time.time() - t0
                    test_time.append(time_i)
                    print("{0:} {1:}".format(name, time_i))
                    save_name = name if '/' not in name else name.split('/')[-1]
                    save_name = "{0:}/{1:}.nii.gz".format(save_folder, save_name)
                    reference_name = "{0:}/{1:}_{2:}.nii.gz".format(config_data['data_dir_list'][0],
                                        name, config_data['feature_channel_names'][0])
                    save_array_as_nifty_volume(predict, save_name, reference_name)
                    num = num + 1
            except tf.errors.OutOfRangeError:
                break
        print("Finished {0:} images".format(num))
        test_time = np.asarray(test_time)
        print("average test time: {0:}".format(test_time.mean()))
        np.savetxt(save_folder + "/test_time.txt", test_time)
        self.sess.close()
        

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python test.py config17/test_all_class.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    brats_test = BratsTest(config_file)
    brats_test.inference()
    
    
