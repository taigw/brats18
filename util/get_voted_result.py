import os
import nibabel
import csv
import numpy as np
from scipy import stats
from util.data_process import *
from util.parse_config import parse_config

def remove_edema_false_positive(seg):
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(seg,s) # labeling
    sizes = ndimage.sum(seg,labeled_array,range(1,numpatches+1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    
    new_seg = np.zeros_like(seg)
    for temp_size in sizes_list:
        temp_lab = np.where(sizes == temp_size)[0] + 1
        temp_component = labeled_array == temp_lab
        temp_seg = temp_component * seg
        tc_idices = np.where(temp_seg == 1)
        tc_size = len(tc_idices[0])
        str = "tumor core size {0:}".format(tc_size)
        if(tc_size > 0):
            new_seg = new_seg + temp_seg
        else:
            str = str + ", ingore edema with size {0:}".format(temp_size)
        print(str)
    return new_seg

def fill_holes_in_tumor_core(seg):
    seg_backup = seg * np.ones_like(seg)
    wt = np.asarray(seg >0, np.uint8)
    wt = fill_holes(wt)
    seg[wt == 1] = 1
    seg[seg_backup == 4] = 4
    seg[seg_backup == 2] = 2
    
    tc = np.asarray(seg == 1, np.uint8) +  np.asarray(seg == 4, np.uint8)
    tc = fill_holes(tc)
    seg[tc == 1] = 1
    seg[seg_backup == 4] = 4
    return seg

def remove_tumor_core_false_positive(seg):
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    tc = seg*np.ones_like(seg)
    tc[tc == 2] = 0
    
    mask = np.asarray(tc > 0, np.float32)
    if(mask.sum() == 0):
        print("--tumor core component []")
        return seg
    mask = ndimage.morphology.binary_opening(mask, s)
    labeled_array, numpatches = ndimage.label(mask,s) # labeling
    sizes = ndimage.sum(mask,labeled_array,range(1,numpatches+1))
    if(sizes.size == 0):
        return seg
    max_size = sizes.max()
    sizes_list = [sizes[i] for i in range(len(sizes)) \
                   if (sizes[i]/max_size) > 0.1]
    sizes_list.sort()
    if(len(sizes_list) > 2):
        sizes_list = sizes_list[-2:]
    tc_num = len(sizes_list)
    
    print("--tumor core component", sizes_list)
    new_tc = np.zeros_like(seg)
    for temp_size in sizes_list:
        temp_lab = np.where(sizes == temp_size)[0] + 1
        temp_component = labeled_array == temp_lab
        temp_tc = temp_component * tc
        
        num_en = np.asarray(temp_tc == 4, np.float32).sum()
        if(num_en < 30):
            print("----enhacing core", num_en)
            temp_tc[temp_tc == 4] = 1
        new_tc = new_tc + temp_tc
    seg[seg > 0] = 2
    seg[new_tc > 0] = new_tc[new_tc > 0]
    return seg

def brats_post_process(seg):
    new_seg = np.zeros_like(seg)
    
    # find the largest two components
    s = ndimage.generate_binary_structure(3,2)
    mask = np.asarray(seg > 0, np.uint8)
    labeled_array, numpatches = ndimage.label(mask,s) # labeling
    sizes = ndimage.sum(mask,labeled_array,range(1,numpatches+1))
    max_size = sizes.max()
    sizes_list = [sizes[i] for i in range(len(sizes)) \
                   if ((sizes[i]/max_size) > 0.1 or sizes[i] > 2000)]
    sizes_list.sort()
    if(len(sizes_list) > 2):
        sizes_list = sizes_list[-2:]
    print("whole tumor components ", sizes_list)
    
    # iterate each whole tumor component
    for i in range(len(sizes_list)):
        print("whole tumor index {0:}".format(i))
        temp_size = sizes_list[i]
        temp_lab = np.where(sizes == temp_size)[0] + 1
        temp_mask = labeled_array == temp_lab
        temp_seg = seg * temp_mask
        
        # fill holes in tumor core
        temp_seg = fill_holes_in_tumor_core(temp_seg)
        # remove tumor core false positive
        temp_seg = remove_tumor_core_false_positive(temp_seg)
        new_seg = new_seg + temp_seg
    return new_seg

def get_voted_result(input_folder_list, output_folder):
    """
    each folder in input_folder_list contains the segmenatation of all the images
    """
    if(not os.path.isdir(output_folder)):
        os.mkdir(output_folder)
    patient_names = os.listdir(input_folder_list[0])
    patient_names = [name for name in patient_names if "nii.gz" in name]
    for item in patient_names:
        print(' ')
        print(item)
        seg_data = []
        for folder in input_folder_list:
            full_filename = "{0:}/{1:}".format(folder, item)
            img_obj  = nibabel.load(full_filename)
            img_data = img_obj.get_data()
            seg_data.append(img_data)
            
            if(img_data.shape[0] == 0):
                exit()
        seg_data = np.asarray(seg_data)
        vote_data = stats.mode(seg_data, axis = 0)[0][0]

        # post process
        vote_data = brats_post_process(vote_data)
        output_img = nibabel.nifti1.Nifti1Image(vote_data, img_obj.affine,
                            img_obj.header, img_obj.extra, img_obj.file_map)
        save_filename = "{0:}/{1:}".format(output_folder, item)
        nibabel.save(output_img, save_filename)

def get_voted_result_for_single_folder_multiple_predictions(result_root, result_under_patient_dir):
    """
    the root folder contain folders for all the patients, and each patient folder
    contains multiple predictions of the same patient.
    """
    patient_names = os.listdir(result_root)
    print(patient_names)
    patient_names = [item for item in patient_names \
        if ("Brats" in item and os.path.isdir("{0:}/{1:}".format(result_root,item)))]
    for item in patient_names:
        print(item)
        seg_data = []
        if(result_under_patient_dir):
            result_folder = "{0:}/{1:}/results/tumor_tiggw_class".format(result_root, item)
        else:
            result_folder = "{0:}/{1:}".format(result_root, item)
        seg_files = os.listdir(result_folder)
        for seg_file in seg_files:
            if("nii.gz" in seg_file):
                full_seg_name = "{0:}/{1:}".format(result_folder, seg_file)
                img_obj  = nibabel.load(full_seg_name)
                img_data = img_obj.get_data()
                seg_data.append(img_data)

        seg_data = np.asarray(seg_data)
        vote_data = stats.mode(seg_data, axis = 0)[0][0]

        # post process
        vote_data = brats_post_process(vote_data)
        output_img = nibabel.nifti1.Nifti1Image(vote_data, img_obj.affine,
                            img_obj.header, img_obj.extra, img_obj.file_map)

        save_filename = "{0:}.nii.gz".format(result_folder)
        nibabel.save(output_img, save_filename)


if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python get_vote_result.py vote_result_cfg.txt')
        exit()
    config_file = str(sys.argv[1])
    config = parse_config(config_file)
    print(config)


        
        
