import numpy as np
import pandas as pd
from numpy import load
from numpy import genfromtxt
import json
from synspy.analyze.util import load_segment_status_from_csv,dump_segment_info_to_csv
import tifffile as tif
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os.path
from os import path

# centroids_tiff: centroid coordinates in the Tiff space
# labels: labels of the centroids (1=synapse/0=not)
def get_centroids_and_labels(csvfilename,npz_filename):
    mpp = np.array([0.4, 0.26, 0.26], dtype=np.float32)  # in ZYX packed order
    parts = np.load(npz_filename)
    props = json.loads(parts['properties'].tostring().decode('utf8'))

    centroids = parts['centroids'].astype(np.int32)
    measures = parts['measures'].astype(np.float32) * np.float32(props['measures_divisor'])
    slice_origin = np.array(props['slice_origin'], dtype=np.int32)
    # build dense status array from sparse CSV
    statuses, saved_parms = load_segment_status_from_csv(centroids, slice_origin, csvfilename)
    # convert cropped centroids back to full SPIM voxel coords
    centroids_tiff = centroids + slice_origin
    # rescale voxel coords into micron coords in SPIM space
    centroids_microns =  mpp * centroids_tiff
    # interpret status flag values
    is_synapse = statuses == 7
    labels = is_synapse.astype(int)
    
    return centroids_tiff,labels

# centroids_tiff: centroid coordinates in the Tiff space
def get_centroids(npz_filename):
    mpp = np.array([0.4, 0.26, 0.26], dtype=np.float32)  # in ZYX packed order
    parts = np.load(npz_filename)
    props = json.loads(parts['properties'].tostring().decode('utf8'))

    centroids = parts['centroids'].astype(np.int32)
    measures = parts['measures'].astype(np.float32) * np.float32(props['measures_divisor'])
    slice_origin = np.array(props['slice_origin'], dtype=np.int32)
    # convert cropped centroids back to full SPIM voxel coords
    centroids_tiff = centroids + slice_origin
    # rescale voxel coords into micron coords in SPIM space
    # centroids_microns =  mpp * centroids_tiff
    
    return centroids_tiff

# get_centroids_from_csv: centroid coordinates in the Tiff space directly as in csv (Z,Y,X)
def get_centroids_from_csv(csv_filename):
    
    # finish it!!! 
    
    return centroids_csv

def saveResultsAsTiff(filename, segmentation , predictions):
    """ Saves results as tiff:
    segmentation : name of the segmentation
    filename: full filename with the extention
    image_file : file of the size for the classifiacation
    """
    csvfilename,npz_filename,image_file_green, _, _ = get_files(segmentation)
    # get file info
    im_info = tif.TiffFile(image_file_green)
    series = im_info.series[0]
    # create results as volume
    centroids_tiff,labels = get_centroids_and_labels(csvfilename,npz_filename)
    labels = labels.astype(bool)

    # all available choises
    # volume = np.zeros(series.shape,dtype = np.int16)
    # volume = make_cube(volume, centroids_tiff, 1)
    # tif.imsave(f"{filename}_all_local_max.tif", volume, photometric='minisblack')
    # print("did all available choises")

    # human 
    # volume = np.zeros(series.shape,dtype = np.int16)
    # volume = make_cube(volume, centroids_tiff[labels], 1)
    # tif.imsave(f"{filename}_human_grader.tif", volume, photometric='minisblack')
    # print("did human")

    # all predictions
    volume = np.zeros(series.shape,dtype = np.int16)
    volume = make_cube(volume, centroids_tiff[predictions], 1)
    tif.imsave(f"{filename}_all.tif", volume, photometric='minisblack')
    print("did all predictions")

    # all correct predictions
    correct = np.logical_and(predictions,labels)
    volume = np.zeros(series.shape,dtype = np.int16)
    volume = make_cube(volume, centroids_tiff[correct], 1)
    tif.imsave(f"{filename}_correct.tif", volume, photometric='minisblack')
    print("did all correct predictions")

    # false positive
    false_pos = np.logical_and(predictions,np.logical_not(labels))
    volume = np.zeros(series.shape,dtype = np.int16)
    volume = make_cube(volume, centroids_tiff[false_pos], 1)
    tif.imsave(f"{filename}_false_pos.tif", volume, photometric='minisblack')
    print("did all false positives")

    # false negative
    false_neg = np.logical_and(labels,np.logical_not(predictions))
    volume = np.zeros(series.shape,dtype = np.int16)
    volume = make_cube(volume, centroids_tiff[false_neg], 1)
    tif.imsave(f"{filename}_false_neg.tif", volume, photometric='minisblack')
    print("did all false negatives")

    # TODO : make similar only for the classifier and humans separately

def applyTransform(points,transform):
    points = np.append(points,np.ones((points.shape[0],1)),axis = 1)
    points = np.matmul(points,transform)
    return points[:,:-1]

def transformAndSaveResultsAsTiff(filename, segmentation , predictions, transform):
    """ Saves results as tiff:
    segmentation : name of the segmentation
    filename: full filename with the extention
    image_file : file of the size for the classifiacation
    """
    csvfilename,npz_filename,image_file_green, _, _ = get_files(segmentation)
    # get file info
    im_info = tif.TiffFile(image_file_green)
    series = im_info.series[0]
    # create results as volume
    centroids_tiff,labels = get_centroids_and_labels(csvfilename,npz_filename)
    centroids_tiff = applyTransform(centroids_tiff,transform)
    labels = labels.astype(bool)

    # all available choises
    # volume = np.zeros(series.shape,dtype = np.int16)
    # volume = make_cube(volume, centroids_tiff, 1)
    # tif.imsave(f"{filename}_all_local_max.tif", volume, photometric='minisblack')
    # print("did all available choises")

    # human 
    # volume = np.zeros(series.shape,dtype = np.int16)
    # volume = make_cube(volume, centroids_tiff[labels], 1)
    # tif.imsave(f"{filename}_human_grader.tif", volume, photometric='minisblack')
    # print("did human")

    # all predictions
    volume = np.zeros(series.shape,dtype = np.int16)
    volume = make_cube(volume, centroids_tiff[predictions], 1)
    tif.imsave(f"{filename}_all.tif", volume, photometric='minisblack')
    print("did all predictions")

    # all correct predictions
    correct = np.logical_and(predictions,labels)
    volume = np.zeros(series.shape,dtype = np.int16)
    volume = make_cube(volume, centroids_tiff[correct], 1)
    tif.imsave(f"{filename}_correct.tif", volume, photometric='minisblack')
    print("did all correct predictions")

    # false positive
    false_pos = np.logical_and(predictions,np.logical_not(labels))
    volume = np.zeros(series.shape,dtype = np.int16)
    volume = make_cube(volume, centroids_tiff[false_pos], 1)
    tif.imsave(f"{filename}_false_pos.tif", volume, photometric='minisblack')
    print("did all false positives")

    # false negative
    false_neg = np.logical_and(labels,np.logical_not(predictions))
    volume = np.zeros(series.shape,dtype = np.int16)
    volume = make_cube(volume, centroids_tiff[false_neg], 1)
    tif.imsave(f"{filename}_false_neg.tif", volume, photometric='minisblack')
    print("did all false negatives")

    # TODO : make similar only for the classifier and humans separately

def make_cube(volume, centroids, padding):
    # radius in pixels
    for centroid in centroids:
        start = centroid - padding
        end = centroid + padding + 1
        volume[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = 100
        volume[centroid[0],start[1]:end[1],start[2]:end[2]] = 200
    return volume


# get the volume around one synapse with the bounding box padding
def get_one_volume(image_file, centroid , padding):
    volume = tif.imread(image_file)
    d0 = centroid[0]
    d1 = centroid[1]
    d2 = centroid[2]
    
    d0_left = d0-padding[0]
    d1_left = d1-padding[1]
    d2_left = d2-padding[2]
    d0_right = d0+1+padding[0]
    d1_right = d1+1+padding[1]
    d2_right = d2+1+padding[2]
    
    cropped_volume = volume[d0_left:d0_right,d1_left:d1_right,d2_left:d2_right]   
    return cropped_volume

def get_all_volumes(image_file, centroids, padding):
    volume = tif.imread(image_file)
    dims = 2*padding+1
    cropped_volumes = np.zeros((centroids.shape[0],dims[0],dims[1],dims[2]))

    for i_syn in range(centroids.shape[0]):
        d0 = centroids[i_syn,0]
        d1 = centroids[i_syn,1]
        d2 = centroids[i_syn,2]
    
        d0_left = d0-padding[0]
        d1_left = d1-padding[1]
        d2_left = d2-padding[2]
        d0_right = d0+1+padding[0]
        d1_right = d1+1+padding[1]
        d2_right = d2+1+padding[2]

        cropped_volumes[i_syn,:,:] = np.squeeze(volume[d0_left:d0_right,
                                                        d1_left:d1_right,
                                                        d2_left:d2_right])
    cropped_volumes = np.squeeze(cropped_volumes)
    # cropped_volumes = np.expand_dims(cropped_volumes, axis=3)
     
    return cropped_volumes 

# returns mean intensity of the square box within the intensity padding 
def get_mean_intensity(volumes, intensity_padding):

    num_syn,d1,d2 = volumes.shape
    intensity =  np.zeros((num_syn,1))

    for i_syn in range(num_syn):
        center1 = d1//2
        center2 = d2//2
    
        d1_left = center1-intensity_padding[1]
        d2_left = center2-intensity_padding[2]

        d1_right = center1+1+intensity_padding[1]
        d2_right = center2+1+intensity_padding[2]

        intensity[i_syn] = np.mean(volumes[i_syn,
                                            d1_left:d1_right,
                                            d2_left:d2_right])
     
    return intensity 

# 1. mean image subtraction - works well if your color 
# or intensity distributions is not consistent 
# throughout the image (e.g. only centered objects)

# 2. per-channel normalization (subtract mean, divide by standard deviation), 
# pretty standard, useful for variable sized input where you can't use 1.

# 3. per-channel mean subtraction - good for variable sized input 
# where you can't use 1 and don't want to make too many assumptions about the distribution.

# 4. whitening (turn the distribution into a normal distribution, 
# sometimes as easy as normalization but only if it's already normally distributed). 
# Maybe others can weigh in on cases where whitening is not a good idea.   

def normalize_volumes1(volumes):
    volumes = volumes - np.mean(volumes,axis = (1,2))[:,None,None]
    volumes = np.expand_dims(volumes, axis=3)
    return volumes



def split_data_equal(data,labels):
    X_1 = data[labels==1]
    X_0 = data[labels==0]
    keep = np.random.choice(range(X_0.shape[0]), size=X_1.shape[0], replace=False)
    X_0 = X_0[keep,:]
    new_label = np.concatenate((np.ones((X_1.shape[0],1)),
                                np.zeros((X_0.shape[0],1))),axis = 0)
    X_train, X_test, y_train, y_test = train_test_split(np.concatenate((X_1,X_0),axis = 0),
                                                        new_label, test_size=0.1)
    return X_train, X_test, y_train, y_test

def split_data_2Xmore0(data,labels):
    X_1 = data[labels==1]
    X_0 = data[labels==0]
    weight_0 = 2
    keep = np.random.choice(range(X_0.shape[0]), size=weight_0*X_1.shape[0], replace=False)
    X_0 = X_0[keep,:]
    new_label = np.concatenate((np.ones((X_1.shape[0],1)),
                                np.zeros((X_0.shape[0],1))),axis = 0)
    X_train, X_test, y_train, y_test = train_test_split(np.concatenate((X_1,X_0),axis = 0),
                                                        new_label, test_size=0.1)
    return X_train, X_test, y_train, y_test

# get mean, max, min and median nuclear intensity for a fish
def get_nuclear_stats(segmentation):
    _,_,_, _, nucfilename = get_files(segmentation)
    if path.exists(nucfilename):
        nuclear_data = pd.read_csv(nucfilename,skiprows=[1])
        Inuc_mean = nuclear_data.loc[:,"raw core"].mean()
        Inuc_med = nuclear_data.loc[:,"raw core"].median()
        Inuc_min = nuclear_data.loc[:,"raw core"].min()
        Inuc_max = nuclear_data.loc[:,"raw core"].max()
    else:
        Inuc_mean = 'NaN'
        Inuc_med = 'NaN'
        Inuc_min = 'NaN'
        Inuc_max = 'NaN'

    return Inuc_mean,Inuc_med,Inuc_min,Inuc_max

# get mean green intensity in a volume centered at synapse and int_padding size
# returns numpy array size (NumSyn,1)
def get_green_Intensity_stats(segmentation,int_padding):

    csvfilename,npz_filename,image_file_green, _, _ = get_files(segmentation)
    centroids,_ = get_centroids_and_labels(csvfilename,npz_filename)
    volumes = get_all_volumes(image_file_green, centroids, padding)
    # mean intensity in each volume
    intensity_green = get_mean_intensity(volumes, int_padding)
    return intensity_green

# get mean red intensity in a volume centered at synapse and int_padding size
# returns numpy array size (NumSyn,1)
def get_green_Intensity_stats(segmentation,int_padding):

    csvfilename,npz_filename,_, image_file_red, _ = get_files(segmentation)
    centroids,_ = get_centroids_and_labels(csvfilename,npz_filename)
    volumes = get_all_volumes(image_file_red, centroids, padding)
    intensity_red = get_mean_intensity(volumes, int_padding)
    return intensity_red






# weight_0: how many times more 0 then 1 in the train+test
# if weight_0 = 2 it is the same as split_data_2Xmore0 function
def split_data_more0(data,labels,weight_0):
    X_1 = data[labels==1]
    X_0 = data[labels==0]

    keep = np.random.choice(range(X_0.shape[0]), size=weight_0*X_1.shape[0], replace=False)
    X_0 = X_0[keep,:]
    new_label = np.concatenate((np.ones((X_1.shape[0],1)),
                                np.zeros((X_0.shape[0],1))),axis = 0)
    X_train, X_test, y_train, y_test = train_test_split(np.concatenate((X_1,X_0),axis = 0),
                                                        new_label, test_size=0.1)
    return X_train, X_test, y_train, y_test

def plot_slice(data, title):
    plt.figure(figsize=(10, 30))
    imshow(data, cmap='gray')
    plt.title(title)
    plt.axis('off')

# get all necessary files for the given segmentation (adds directories and extentions)
#  returns strings with full paths to the files
# segmentation is in form  ImgZfDsy20170915D3A 
def get_files(segmentation):  

    csvfilename = 'C:\\Users\\nadtochi\\csci599\\meExperiment\\math650\\'+segmentation+'\\Syn'+segmentation+'.synapses-only.csv'
    nucfilename = 'C:\\Users\\nadtochi\\csci599\\meExperiment\\math650\\'+segmentation+'\\Nuc'+segmentation[0:-1]+'B.nuclei-only.csv'
    npz_filename = 'C:\\Users\\nadtochi\\csci599\\meExperiment\\math650\\'+segmentation+'\\Syn'+segmentation+'.npz' 
    image_file_green = 'C:\\Users\\nadtochi\\csci599\\meExperiment\\math650\\'+segmentation+'\\green.tif'
    image_file_red = 'C:\\Users\\nadtochi\\csci599\\meExperiment\\math650\\'+segmentation+'\\red2green.tif'

    return csvfilename,npz_filename,image_file_green, image_file_red, nucfilename

def get_files_from_cohort(segmentation):  

    csvfilename = 'D:\\TR01\\1640Cohort\\CsvFiles\\Syn'+segmentation+'.synapses-only.csv'
    npz_filename = 'D:\\TR01\\1640Cohort\\NpzFiles\\Syn'+segmentation+'.npz' 
    image_file_green = 'D:\\TR01\\1640Cohort\\GreenDataFR\\'+segmentation[:-1]+'_green.tif'
    image_file_red = 'D:\\TR01\\1640Cohort\\R2G_Images\\'+segmentation[:-1]+'_r2g.tif'
    nucfilename = 'Missing'

    return csvfilename,npz_filename,image_file_green, image_file_red, nucfilename

# returns volumes and intenseties in two channels
def get_train_data(segmentations, padding, intensity_padding):
    num_datasets = len(segmentations)
    last_dat = num_datasets - 1
    csvfilename,npz_filename,image_file_green,image_file_r2g,nucfilename = get_files(segmentations[last_dat])

    # get centroids and labels
    centroids,labels = get_centroids_and_labels(csvfilename,npz_filename)
    num_syn = centroids.shape[0]
    # add fish ID
    centroids_and_fish = np.concatenate((centroids,np.ones((num_syn,1))*(last_dat+1)),axis = 1)

    # get normalized image and intensity in grenn channel
    volumes = get_all_volumes(image_file_green,centroids, padding)
    intensity_green = get_mean_intensity(volumes, intensity_padding)
    volumes = normalize_volumes1(volumes)

    # get intensity in red channel
    volumes_red = get_all_volumes(image_file_r2g,centroids, padding)
    intensity_red = get_mean_intensity(volumes_red, intensity_padding)

    # get nuclear data
    #Inuc = np.zeros((1,4))
    #Inuc[0,0],Inuc[0,1],Inuc[0,2],Inuc[0,3] = get_nuclear_stats(segmentations[last_dat])
    #Inuc = np.repeat(Inuc,num_syn, axis=0)
    #centroids_nuc = get_centroids(csvfilename,npz_filename)

    for i_fish in range(last_dat):

        csvfilename,npz_filename,image_file_green,image_file_r2g,nucfilename = get_files(segmentations[i_fish])

        # get centroids and labels
        centroids,i_labels = get_centroids_and_labels(csvfilename,npz_filename)
        labels = np.concatenate((labels,i_labels),axis = 0)
        num_syn = centroids.shape[0]
        # add fish ID
        i_centroids_and_fish = np.concatenate((centroids,np.ones((num_syn,1))*(i_fish+1)),axis = 1)
        centroids_and_fish = np.concatenate((centroids_and_fish,i_centroids_and_fish),axis = 0)

        # get normalized image and intensity in grenn channel
        i_volumes = get_all_volumes(image_file_green,centroids, padding)
        i_intensity_green = get_mean_intensity(i_volumes, intensity_padding)
        i_volumes = normalize_volumes1(i_volumes)
        
        volumes = np.concatenate((volumes,i_volumes),axis = 0)
        intensity_green = np.concatenate((intensity_green,i_intensity_green),axis = 0)

        # get intensity in red channel
        volumes_red = get_all_volumes(image_file_r2g,centroids, padding)
        i_intensity_red = get_mean_intensity(volumes_red, intensity_padding)
        intensity_red = np.concatenate((intensity_red,i_intensity_red),axis = 0)

        # get nuclear data
        #i_Inuc = np.zeros((1,4))
        #i_Inuc[0,0],i_Inuc[0,1],i_Inuc[0,2],i_Inuc[0,3] = get_nuclear_stats(segmentations[i_fish])
        #i_Inuc = np.repeat(i_Inuc,num_syn, axis=0)
        #Inuc = np.concatenate((Inuc,i_Inuc),axis = 0)

    print(f'Potential synapses:{labels.shape[0]}')

    # wrap all intensity info into a data frame
    features_labels = ["Label","Z","Y","X",
                    "FishN", # "FishID",
                    "IntGrn_avg", #"IntGrn_med","IntGrn_min","IntGrn_max",
                    "IntRed_avg"] #, #"IntRed_med","IntRed_min","IntRed_max",
                    # "IntNuc_avg","IntNuc_med","IntNuc_min","IntNuc_max"]
    synapse_data = np.concatenate([labels[:,np.newaxis],centroids_and_fish,
                                intensity_green,intensity_red],axis=1) # Inuc
    synapse_dataset = pd.DataFrame(synapse_data)
    synapse_dataset.columns = features_labels
    # synapse_dataset['Label'] = synapse_dataset.Label.astype('category')
    # synapse_dataset['FishN'] = synapse_dataset.FishN.astype('category')
    
    return volumes, synapse_dataset, labels

def predict_and_create_csv(model,segmentation,output_path,output_filename,padding):  
    
    csvfilename,npz_filename,image_file,image_file_red = get_files(segmentation) 
    centroids_tiff,eval_labels = get_centroids_and_labels(csvfilename,npz_filename)
    print('Image file : ',image_file)
    eval_data_green = get_all_volumes(image_file, centroids_tiff, padding)
    eval_data_green = normalize_volumes1(eval_data_green)
    predictions = model.predict(eval_data_green, batch_size=128)
    
    #np.savetxt(output_path+segmentation+"_IntensityProb.csv",
    #           np.concatenate((intensity_green,predictions),axis = 1), delimiter=",")
    
    
    mpp = np.array([0.4, 0.26, 0.26], dtype=np.float32)  # in ZYX packed order
    parts = np.load(npz_filename)
    props = json.loads(parts['properties'].tostring().decode('utf8'))

    centroids = parts['centroids'].astype(np.int32)
    measures = parts['measures'].astype(np.float32) * np.float32(props['measures_divisor'])
    slice_origin = np.array(props['slice_origin'], dtype=np.int32)
    
    # build dense status array from sparse CSV
    statuses, saved_parms = load_segment_status_from_csv(centroids, slice_origin, csvfilename)

    
    thr = 0.6
    statuses[:] = 0
    predictionsArr = (np.concatenate(predictions, axis=0 )>thr)
    #statuses[predictionsArr==0] = 5
    statuses[predictionsArr==1] = 7
    print('Sum 0.6',sum(predictionsArr))
    dump_segment_info_to_csv(centroids, measures, statuses, slice_origin, output_path+"thr0p6\\"+output_filename, all_segments=False)
    
    thr = 0.85
    statuses[:] = 0
    predictionsArr = (np.concatenate(predictions, axis=0 )>thr)
    #statuses[predictionsArr==0] = 5
    statuses[predictionsArr==1] = 7
    print('Sum 0.85',sum(predictionsArr))
    dump_segment_info_to_csv(centroids, measures, statuses, slice_origin, output_path+"thr0p85\\"+output_filename, all_segments=False)
    
    thr = 0.95
    statuses[:] = 0
    predictionsArr = (np.concatenate(predictions, axis=0 )>thr)
    #statuses[predictionsArr==0] = 5
    statuses[predictionsArr==1] = 7
    print('Sum 0.95',sum(predictionsArr))
    dump_segment_info_to_csv(centroids, measures, statuses, slice_origin, output_path+"thr0p95\\"+output_filename, all_segments=False)

    
    
    print('Done')

# well... plot ROC curve ;)
def plot_roc_curve(fpr,tpr,label = None):
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# writes out a csv in Karl-readable format
def create_csv(predictionsArr,segmentation,output_filename) : 

    csvfilename,npz_filename,_,_ = get_files(segmentation) 
    # load data from *.npz
    mpp = np.array([0.4, 0.26, 0.26], dtype=np.float32)  # in ZYX packed order
    parts = np.load(npz_filename)
    props = json.loads(parts['properties'].tostring().decode('utf8'))

    centroids = parts['centroids'].astype(np.int32)
    measures = parts['measures'].astype(np.float32) * np.float32(props['measures_divisor'])
    slice_origin = np.array(props['slice_origin'], dtype=np.int32)
    
    # build dense status array from sparse CSV and override statuses
    statuses, saved_parms = load_segment_status_from_csv(centroids, slice_origin, csvfilename)
    statuses[:] = 0
    statuses[predictionsArr==1] = 7

    dump_segment_info_to_csv(centroids, measures, statuses, slice_origin, output_filename, all_segments=False)   
    print('Saved ',output_filename)

def print_acc_different_threshold(predictions):
    thr = 0.5
    predictionsArr = (np.concatenate(predictions, axis=0 )>thr)
    acc = accuracy_score(predictionsArr, labels)
    print('Threshold    acc')
    print(thr,' ',acc)



