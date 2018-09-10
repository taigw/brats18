import os
import sys
sys.path.append("./")
import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from util.preprocess.rotate import RotateLayer

def test_rotate():
    input_name = "../Brats17_2013_2_1/Brats17_2013_2_1_flair.nii.gz"
    img = sitk.ReadImage(input_name)
    data = sitk.GetArrayFromImage(img)
    data = np.reshape(data, [1, 155, 240, 240, 1])
    
    rotate = RotateLayer()
    rotate.set_angle_range([0, 10])
    input_dict = {}
    entry_data = {"feature": data}
    input_dict["entry_data"] = entry_data
    input_dict["entry_name"] = "Brats17_2013_2_1"
    output_dict = rotate.np_operate(input_dict)
    output_dict = rotate.inverse_np_operate(output_dict)
    data_out = output_dict["entry_data"]["feature"]
    data_out = np.reshape(data_out, [155, 240, 240])
    img_out = sitk.GetImageFromArray(data_out)
    img_out.CopyInformation(img)
    sitk.WriteImage(img_out, "Brats17_2013_2_1_flair.nii.gz")
    print(data_out.shape)
if __name__ == '__main__':
    test_rotate()
