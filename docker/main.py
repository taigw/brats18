# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import subprocess
sys.path.append("/data/brats18/")
from test import BratsTest
from util.image_io.get_patient_names import get_brats18_test_names

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python main.py test_cfg.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    get_brats18_test_names("/data", [None], "/data/brats18/docker/test_names.txt")
    brats_test = BratsTest(config_file)
    brats_test.inference()
