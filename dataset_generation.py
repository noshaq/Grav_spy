from joblib import Parallel, delayed
import multiprocessing
from shutil import copy2
from skimage import io
import pandas as pd
import numpy as np
import glob
import re
import os


######## Read the metadata file ############
###  Downloaded from (https://zenodo.org/record/1476551#.XEHhrca20rh)

metadata = pd.read_csv("./TrainingSet/trainingset_v1d1_metadata.csv")

######## Keep only the metadata necessary for the identification of noise type  

medata_id = metadata.drop(['event_time', 'ifo', 'peak_time', 'peak_time_ns', 'start_time',
       'start_time_ns', 'duration', 'search', 'process_id', 'event_id',
       'peak_frequency', 'central_freq', 'bandwidth', 'channel', 'amplitude',
       'snr', 'confidence', 'chisq', 'chisq_dof', 'param_one_name',
       'param_one_value', 'url1',
       'url2', 'url3', 'url4'], axis = 1)


##### Store the filenames belonging to train, valid, test

train_set = medata_id.loc[(medata_id['sample_type'] == "train")]
validation_set = medata_id.loc[(medata_id['sample_type'] == "validation")]
test_set = medata_id.loc[(medata_id['sample_type'] == "test")]


########## function: move_files ########################################################################################
#
# The function reads the folder "TrainingSet", that contain the spectograms 
# extracted from the file: trainingsetv1d1.tar.gz (https://zenodo.org/record/1476551#.XEHhrca20rh)
# 
# metadata_df: Pandas dataframe with the filenames belonging to train, valid, test (train_set,validation_set,test_set)
# type_of_noise: List with all the noise types described in the metadata file
# number_of_samples: Use only if you dont need the whole dataset.
# dataset_type : enter if the files belong to train, test or valid
#######################################################################################################################


def move_files(metadata_df,type_of_noise, number_of_samples,dataset_type):
      
 
    if os.path.isdir("./data/")== False:
         os.mkdir("./data/")

    list_of_files = []
    filtered_file_paths = []
        
    noise_id = metadata_df.loc[(metadata_df['label'] == type_of_noise)]
    noise_id = noise_id.drop(["label","sample_type"], axis = 1).values
    noise_id = noise_id.flatten()
    noise_id = np.random.choice(noise_id, number_of_samples)
        
        
    for name in glob.glob('./TrainingSet/'+type_of_noise+'/*.png'):
        list_of_files.append(name)
            
    for unique_id in noise_id:      
        for line in list_of_files:
            if re.match(".*_.*"+unique_id+".*_.*\.png",line):
                filtered_file_paths.append(line)
                
    path ="./data/"+dataset_type+"/"+type_of_noise
    
    if os.path.isdir("./data/"+dataset_type+"/")== False:
         os.mkdir("./data/"+dataset_type+"/")
    if os.path.isdir("./data/"+dataset_type+"/"+type_of_noise)== False:
        os.mkdir(path)
    
    for move_files in filtered_file_paths:
        copy2(move_files,path)
        
####### execute the previous function in parallel. 
# Control it with  num_cores argument. 
# Don't use numcores > than the number of noise types. (Domain decomposition)

max_core_num=len(metadata["label"].unique())
total_cores = multiprocessing.cpu_count()

def parallel_move_files(metadata_df, number_of_samples,dataset_type="train",num_cores=total_cores):
   
    noise_type_spectogram = metadata_df['label'].unique()
    if num_cores > max_core_num:
    	num_cores = total_cores

    
    Parallel(n_jobs=num_cores, verbose=2)(delayed(move_files)(metadata_df, type_of_noise,number_of_samples,dataset_type)for type_of_noise in noise_type_spectogram)


####### Resize all the images to keep only the spectrogram #################################


def resize_images(folder_path):
    
    root_folder = [ d for d in os.listdir(folder_path) ]    
    
    for noise_folder in root_folder:
        print("Resizing the files in folder",noise_folder)    
        
        target_path = os.path.join(folder_path+'/',noise_folder+'/' )
        
        for img in  os.listdir(target_path):
            image_data = io.imread(os.path.join(target_path,img))
            x=[66, 532]; y=[105, 671]
            image_data = image_data[x[0]:x[1], y[0]:y[1], :3]
            
            file_to_rewrite = os.path.join(target_path,img)
            io.imsave(file_to_rewrite,image_data)        

    print("done")    
    


####### Do the work


parallel_move_files(validation_set, number_of_samples=22,dataset_type="valid",num_cores=total_cores)


resize_images("./data/valid")   