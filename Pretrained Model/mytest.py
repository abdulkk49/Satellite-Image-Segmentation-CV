import PIL
from PIL import Image
import matplotlib.pyplot as plt
from libtiff import TIFF
from libtiff import TIFFfile, TIFFimage
from scipy.misc import imresize
import numpy as np
import glob
import cv2
import os
import math
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from unet import UNet
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave
from keras import backend as K
from iou import iou
#%matplotlib inline

model = UNet()

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# List of file names of actual Satellite images for traininig 
filelist_trainx = sorted(glob.glob('Gn_res1.tif'), key=numericalSort)
# List of file names of classified images for traininig 
filelist_trainy = sorted(glob.glob('my_x_val_mask_rgb.jpg'), key=numericalSort)

#Remember to change your path if it is in a different directory in above glob.glob

#******************************************************************
#------------------------------------------------------------------
#------------------------------------------------------------------

# Padding at the bottom and at the left of images to be able to crop them into 128*128 images for training

def padding(img, w, h, c, crop_size, stride, n_h, n_w):
    
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra
    
    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra
    
    img_pad = np.zeros(((h+h_toadd), (w+w_toadd), c))
    #img_pad[:h, :w,:] = img
    #img_pad = img_pad+img
    img_pad = np.pad(img, [(0, h_toadd), (0, w_toadd), (0,0)], 	mode='constant')
    
    return img_pad
    

# Adding pixels to make the image with shape in multiples of stride

def add_pixels(img, h, w, c, n_h, n_w, crop_size, stride):
        
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra

    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra

    img_add = np.zeros(((h+h_toadd), (w+w_toadd), c))
        
    img_add[:h, :w,:] = img
    img_add[h:, :w,:] = img[:h_toadd,:, :]
    img_add[:h,w:,:] = img[:,:w_toadd,:]
    img_add[h:,w:,:] = img[h-h_toadd:h,w-w_toadd:w,:]
            
    return img_add    


# Slicing the image into crop_size*crop_size crops with a stride of crop_size/4 and makking list out of them

def crops(a, crop_size = 128):
    
    #stride = int(crop_size/2)
    stride = 32

    croped_images = []
    h, w, c = a.shape
    
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    
    # Padding using the padding function we wrote
    # a = padding(a, w, h, c, crop_size, stride, n_h, n_w) 

    # Adding pixels as required
    a = add_pixels(a, h, w, c, n_h, n_w, crop_size, stride)
    
    # Slicing the image into 128*128 crops with a stride of 32
    for i in range(n_h-1):
        for j in range(n_w-1):
            crop_x = a[(i*stride):((i*stride)+crop_size), (j*stride):((j*stride)+crop_size), :]
            croped_images.append(crop_x)
    return croped_images


# Another type of cropping

def new_crops(img, crop_size = 512):
    stride = crop_size 
    
    croped_images = []
    h, w, c = img.shape
    
    n_h = math.ceil(h/stride)
    n_w = math.ceil(w/stride)
    
    for i in range(n_h):
        
        if (h - i*crop_size) >= crop_size:
            stride = crop_size
        elif (h - i*crop_size) <= crop_size:
            stride = (crop_size - (w - i*crop_size))
        for j in range(n_w):
            if (w - i*crop_size) >= crop_size:
                stride = crop_size
            elif (w - i*crop_size) <= crop_size:
                stride = (crop_size - (w - i*crop_size))
                
            crop_x = img[(i*stride):((i*stride)+crop_size), (j*stride):((j*stride)+crop_size), :]
            croped_images.append(crop_x)
    return croped_images




#Not all the above functions are used. Add pixels is used to avoid 
#black pixels that are added when padding is used
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

#Making array of all satellite images by cropping and adding pixels

trainx_list = []

for fname in filelist_trainx[:1]:
    
    # Reading the image
    tif = TIFF.open(fname)
    image = tif.read_image()
    print('Original image shape:')
    print(image.shape)
    # Padding as required and cropping
    crops_list = crops(image)
    #print(len(crops_list))
    trainx_list = trainx_list + crops_list
    
# Array of all the cropped Training sat Images    
X = np.asarray(trainx_list)
print('X length:')
print(len(X))     
#*********************

#Making array of all ground truth images by cropping and adding pixels.

trainy_list = []
import matplotlib.image as mpimage
for fname in filelist_trainy[:1]:

    # Reading the image
    image = mpimage.imread(fname)
    print('Original mask shape:')
    print(image.shape)
    # Padding as required and cropping
    crops_list = crops(image)
    
    trainy_list = trainy_list + crops_list
    
# Array of all the cropped Training gt Images    
y = np.asarray(trainy_list)
print('y length:')
print(len(y))     
#------------------------------------------

#Splitting into validation and training sets
import random
from sklearn.model_selection import train_test_split
trainx, valx, trainy, valy = train_test_split(X, y, test_size=0.2, random_state=42)
num_train_examples = len(trainx)
num_val_examples = len(valx)

print('Num of training examples:' + str(len(trainx)))
print('Num of validation examples:' + str(len(valx)))
print('Original image shape after cropping:' + str(trainx[0].shape))
print('Mask shape after cropping:' + str(trainy[0].shape))

#****************************************
#Visualising a sample
#Chnage the index in trainy[] to view other image

from PIL import Image
import numpy as np

img_pil = Image.fromarray(trainy[65].astype(np.uint8), 'RGB') 
display(img_pil.resize((256,256), PIL.Image.LANCZOS))

#Another way to visualise an image
%matplotlib inline 

import matplotlib.pyplot as plt
plt.imshow(valy[64].astype(np.uint8))
plt.show()
#****************************************

# #!pip3 install tiffile
# from tiffile import imsave as imsaveo
# import numpy as np
# for i in range(num_val_examples):
#     imsaveo('Gn_crop_val/val' + str(i)+ '.tif', valx[i])
# for i in range(num_train_examples):
#   imsave('Gn_crop_train_gt/train_gt' + str(i)+'.jpg', trainy[0])
# for i in range(num_val_examples):
#   imsave('Gn_crop_val_gt/val_gt' + str(i)+'.jpg', valy[0])
# os.mkdir('Gn_crop_val_gt')
# os.mkdir('Gn_crop_train_gt')
# import scipy.misc
#for i in range(num_val_examples):
#    scipy.misc.imsave('Gn_crop_val_gt/val_gt' + str(i)+'.jpg', valy[i])

#--------------------------------------------------------------------

#Defining all colors in training ground truth images
#More classes can be added depending on classes avalable and the code
#further has to be modified in certain places if number of classes have changed.

color_dict = {0: (0, 0, 0),        #road
              1: (0, 125, 0),      #tree
              2: (150, 80, 0),     #soil   
              3: (100, 100, 100),  #Building
              4: (0, 255, 0),      #Field 
              5: (255, 255, 255)}  #Unclassified

#Converting from one hot to rgb and vice versa functions used for one
#hot conversion later

def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    #print(shape)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)

#****************************************************************************************************

#def testing(model, trainx, trainy, testx, testy, weights_file = "model_oneshot.h5"):
#    
#    pred_train_all = []
#    pred_val_all = []
#    
#    model.load_weights(weights_file)
#    
#    Y_pred_train = model.predict(trainx)
#    
#    for k in range(Y_pred_train.shape[0]):
#    
#        pred_train_all.append(Y_pred_train[k])
#    
#    Y_gt_train = [rgb_to_onehot(arr, color_dict) for arr in trainy]
#    
#    Y_pred_val = model.predict(testx)
#    
#    for k in range(Y_pred_val.shape[0]):
#    
#        pred_val_all.append(Y_pred_val[k])
#    
#    Y_gt_val = [rgb_to_onehot(arr, color_dict) for arr in testy]
#    
#    return pred_train_all, Y_gt_train, pred_val_all, Y_gt_val




#Predicting function from trained model
#Specify the name of weights file instead of None
def testing_diffsizes(model, trainx, valx, weights_file = None):
    
    pred_train_all = []
    pred_val_all = []
    
    #Uncomment this if you are using pretained weights
    #model.load_weights(weights_file)

    
    for i in range(len(trainx)):
        img = trainx[i]
        h,w,c = img.shape
        img = np.reshape(img, (1,h,w,c))
        Y_pred_train = model.predict(img)
        bb,h,w,c = Y_pred_train.shape
        Y_pred_train = np.reshape(Y_pred_train, (h,w,c))
        pred_train_all.append(Y_pred_train)
    
    
#     Y_gt_train = [rgb_to_onehot(arr, color_dict) for arr in trainy]
    for i in range(len(valx)):
        img = valx[i]
        h,w,c = img.shape
        img = np.reshape(img, (1,h,w,c))
        Y_pred_val = model.predict(img)
        bb,h,w,c = Y_pred_val.shape
        Y_pred_val = np.reshape(Y_pred_val, (h,w,c))
        pred_val_all.append(Y_pred_val)
    
#    for k in range(Y_pred_val.shape[0]):
    
#        pred_test_all.append(Y_pred_val[k])
    
#     Y_gt_val = [rgb_to_onehot(testy, color_dict)]
    
    return pred_train_all, pred_val_all


pred_train_all, pred_val_all = testing_diffsizes(model, trainx, valx, , weights_file = "model_onehot.h5")

print(pred_val_all[0].shape)
print(Y_gt_val[0].shape)
#print(len(pred_train_all))
#print(len(Y_gt_train))

#--------------------------------------------------------------

# Convert onehot to label
def to_class_no(y_hot_list):
    y_class_list = []
    
    n = len(y_hot_list)
    
    for i in range(n):
        
        out = np.argmax(y_hot_list[i])
        
        y_class_list.append(out)
        
    return y_class_list

#Confusion matrix
def conf_matrix(Y_gt, Y_pred, num_classes = 6):
    
    total_pixels = 0
    kappa_sum = 0
    sudo_confusion_matrix = np.zeros((num_classes, num_classes))
   
#    if len(Y_pred.shape) == 3:
#        h,w,c = Y_pred.shape
#        Y_pred = np.reshape(Y_pred, (1,))
 
    n = len(Y_pred)
    
    for i in range(n):
        y_pred = Y_pred[i]
        y_gt = Y_gt[i]
        
        #y_pred_hotcode = hotcode(y_pred)
        #y_gt_hotcode = hotcode(y_gt)
        
        pred = np.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2]))
        gt = np.reshape(y_gt, (y_gt.shape[0]*y_gt.shape[1], y_gt.shape[2]))
        
        pred = [i for i in pred]
        gt = [i for i in gt]
        
        pred = to_class_no(pred)
        gt = to_class_no(gt)
        
#        pred.tolist()
#        gt.tolist()

        gt = np.asarray(gt, dtype = 'int32')
        pred = np.asarray(pred, dtype = 'int32')

        conf_matrix = confusion_matrix(gt, pred, labels=[0,1,2,3,4,5])
        
        kappa = cohen_kappa_score(gt,pred, labels=[0,1,2,3,4])

        pixels = len(pred)
        total_pixels = total_pixels+pixels
        
        sudo_confusion_matrix = sudo_confusion_matrix + conf_matrix
        
        kappa_sum = kappa_sum + kappa

    final_confusion_matrix = sudo_confusion_matrix
    
    final_kappa = kappa_sum/n

    return final_confusion_matrix, final_kappa

confusion_matrix_train, kappa_train = conf_matrix(trainy_hot, pred_train_all, num_classes = 6)
print('Confusion Matrix for training')
print(confusion_matrix_train)
print('Kappa Coeff for training without unclassified pixels')
print(kappa_train)

confusion_matrix_val, kappa_test = conf_matrix(valy_hot, pred_val_all, num_classes = 6)
print('Confusion Matrix for validation')
print(confusion_matrix_test)
print('Kappa Coeff for validation without unclassified pixels')
print(kappa_test)

#**********************************************************************************************

# Pass Confusion matrix, label to which the accuracy needs to be found, number of classes to be considered
# Returns that particular class accuracy

def acc_of_class(class_label, conf_matrix, num_classes = 8):
    
    numerator = conf_matrix[class_label, class_label]
    
    denorminator = 0
    
    for i in range(num_classes):
        denorminator = denorminator + conf_matrix[class_label, i]
        
    acc_of_class = numerator/denorminator
    
    return acc_of_class


# On training

# Find accuray of all the classes NOT considering the unclassified pixels

for i in range(5):
    acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_train, num_classes = 5)
    print('Accuracy of class '+str(i) + ' WITHOUT unclassified pixels - Training')
    print(acc_of_cl)

# Find accuray of all the classes considering the unclassified pixels

for i in range(6):
    acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_train, num_classes = 6)
    print('Accuracy of class '+str(i) + ' WITH unclassified pixels - Training')
    print(acc_of_cl)
    
# On validation

# Find accuray of all the classes NOT considering the unclassified pixels

for i in range(5):
    acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_val, num_classes = 5)
    print('Accuracy of class '+str(i) + ' WITHOUT unclassified pixels - Validation')
    print(acc_of_cl)

# Find accuray of all the classes considering the unclassified pixels

for i in range(6):
    acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_val, num_classes = 6)
    print('Accuracy of class '+str(i) + ' WITH unclassified pixels - Validation')
    print(acc_of_cl)


# Calulating over all accuracy with and without unclassified pixels

def overall_acc(conf_matrix, include_unclassified_pixels = False):
    
    if include_unclassified_pixels:
        
        numerator = 0
        for i in range(6):
        
            numerator = numerator + conf_matrix[i,i]
        
        denominator = 0   
        for i in range(6):
            for j in range(6):
                
                denominator = denominator + conf_matrix[i,j]
                
        acc = numerator/denominator
        
        return acc
    
    else:
        
        numerator = 0
        for i in range(5):
        
            numerator = numerator + conf_matrix[i,i]
        
        denominator = 0   
        for i in range(5):
            for j in range(5):
            
                denominator = denominator + conf_matrix[i,j]
                
        acc = numerator/denominator
        
        return acc


# Training

# Over all accuracy without unclassified pixels

print('Over all accuracy WITHOUT unclassified pixels - Training')
print(overall_acc(conf_matrix = confusion_matrix_train, include_unclassified_pixels = False))

# Over all accuracy with unclassified pixels

print('Over all accuracy WITH unclassified pixels - Training')
print(overall_acc(conf_matrix = confusion_matrix_train, include_unclassified_pixels = True))

# Validation

# Over all accuracy without unclassified pixels

print('Over all accuracy WITHOUT unclassified pixels - Validation')
print(overall_acc(conf_matrix = confusion_matrix_val, include_unclassified_pixels = False))

# Over all accuracy with unclassified pixels

print('Over all accuracy WITH unclassified pixels - Validation')
print(overall_acc(conf_matrix = confusion_matrix_val, include_unclassified_pixels = True))



# Pred on train, val, test and save outputs
#Change the name of weights file if different

weights_file = "model_onehot.h5"
model.load_weights(weights_file)

#Make sure you have the directories my_train_pred and my_pred val made. Or if you have any other 
#you can change the name in last line


#Predicting and saving outputs

for i_ in range(len(trainx)):
    
    item = trainx[i_]
    
    h,w,c = item.shape
    
    item = np.reshape(item,(1,h,w,c))
    
    y_pred_train_img = model.predict(item)
    
    ba,h,w,c = y_pred_train_img.shape
    
    y_pred_train_img = np.reshape(y_pred_train_img,(h,w,c))
    
    img = y_pred_train_img
    h, w, c = img.shape
        
    for i in range(h):
        for j in range(w):
                
            argmax_index = np.argmax(img[i,j])
                
            sudo_onehot_arr = np.zeros((6))
                
            sudo_onehot_arr[argmax_index] = 1
                
            onehot_encode = sudo_onehot_arr
                
            img[i,j,:] = onehot_encode
    
    y_pred_train_img = onehot_to_rgb(img, color_dict)

    tif = TIFF.open(filelist_trainx[i_])
    image2 = tif.read_image()
    
    h,w,c = image2.shape
    
    y_pred_train_img = y_pred_train_img[:h, :w, :]
    
    imx = Image.fromarray(y_pred_train_img)
    
    imx.save("my_pred_train/pred"+str(i_+1)+".jpg")


#Validation prediction outputs

for i_ in range(len(valx)):
    
    item = valx[i_]
    
    h,w,c = item.shape
    
    item = np.reshape(item,(1,h,w,c))
    
    y_pred_train_img = model.predict(item)
    
    ba,h,w,c = y_pred_train_img.shape
    
    y_pred_train_img = np.reshape(y_pred_train_img,(h,w,c))
    
    img = y_pred_train_img
    h, w, c = img.shape
        
    for i in range(h):
        for j in range(w):
                
            argmax_index = np.argmax(img[i,j])
                
            sudo_onehot_arr = np.zeros((6))
                
            sudo_onehot_arr[argmax_index] = 1
                
            onehot_encode = sudo_onehot_arr
                
            img[i,j,:] = onehot_encode
    
    y_pred_train_img = onehot_to_rgb(img, color_dict)
    print(y_pred_train_img.shape)

    
    imx = Image.fromarray(y_pred_train_img)
    
    imx.save("my_pred_val/pred"+str(i_)+".jpg")

