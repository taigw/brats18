import os
import sys
sys.path.append('../')
from convert_to_tfrecords import convert_to_tfrecords

config_file = "config/test_convert_tfrecord.txt"
convert_to_tfrecords(config_file)
