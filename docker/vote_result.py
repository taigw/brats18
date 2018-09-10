import os
import sys
import nibabel
import csv
import numpy as np
sys.path.append("/data/brats18/")
from util.parse_config import parse_config
from util.get_voted_result import get_voted_result

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python get_vote_result.py vote_result_cfg.txt')
        exit()
    config_file = str(sys.argv[1])
    config = parse_config(config_file)
    print(config)
    get_voted_result(config['data']['input_folder_list'], config['data']['output_folder'])


        
        
