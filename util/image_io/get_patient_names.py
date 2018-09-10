import os

def get_patient_names(data_folder, sub_folders):
    patient_names = []
    for sub_folder in sub_folders:
        if sub_folder is None:
            file_names = os.listdir(data_folder)
        else:
            file_names = os.listdir("{0:}/{1:}".format(data_folder, sub_folder))
        for file_name in file_names:
            if("Brats" in file_name):
                if(sub_folder is None):
                    patient_name = "{0:}/{1:}".format(file_name, file_name)
                else:
                    patient_name = "{0:}/{1:}/{2:}".format(sub_folder, file_name, file_name)
                patient_names.append(patient_name)
    for patient_name in patient_names:
            print(patient_name)

    print('total studies: {0:}'.format(len(patient_names)))
    return patient_names

def get_brats18_test_names(data_folder, sub_folders, output_name):
    patient_names = get_patient_names(data_folder, sub_folders)
    with open(output_name, 'w') as f:
        for item in patient_names:
            f.write("%s\n" % item)
    f.close()

