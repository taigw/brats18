# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import subprocess
sys.path.append("/data/brats18/")
from test import BratsTest
from util.image_io.get_patient_names import get_brats18_test_names

get_brats18_test_names("/data", [None], "/data/brats18/docker/test_names.txt")
brats_test = BratsTest("/data/brats18/docker/test_cfg.txt")
brats_test.inference()
