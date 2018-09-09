# Created on Wed Oct 11 2017
#
# @author: Guotai Wang
# reference: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
import os
import sys
from parse_config import parse_config
from image_io.dataset_from_pygenerator import DataSetFromPyGenerator

def convert_to_tfrecords(config_file):
    config = parse_config(config_file)
    config_data = config['data']
    data_loader = DataSetFromPyGenerator(config_data)
    data_loader.save_to_tfrecords()

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python convert_to_tfrecords.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    convert_to_tfrecords(config_file)
