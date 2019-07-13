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
import tensorflow as tf
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave
from keras import backend as K

#Defining Keras Metric
def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

def tf_mean_iou(y_true, y_pred, num_classes=6):
    return tf.metrics.mean_iou(y_true, y_pred, num_classes)


mean_iou = as_keras_metric(tf_mean_iou)

# Padding at the bottom and at the left of images to be able
# to crop them into 256*256 images for training

def padding(img, w, h, c, crop_size, stride, n_h, n_w):
    
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra
    
    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra
    
    img_pad = np.zeros(((h+h_toadd), (w+w_toadd), c))
    #img_pad[:h, :w,:] = img
    #img_pad = img_pad+img
    img_pad = np.pad(img, [(0, h_toadd), (0, w_toadd), (0,0)], mode='constant')
    
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

def crops(a, crop_size = 256):
    
    stride = 64
    
    croped_images = []
    h, w, c = a.shape
    
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    
    # Padding using the padding function we wrote
    ##a = padding(a, w, h, c, crop_size, stride, n_h, n_w)
    
    # Adding pixels as required
    a = add_pixels(a, h, w, c, n_h, n_w, crop_size, stride)
    
    # Slicing the image into 128*128 crops with a stride of 64
    for i in range(n_h-1):
        for j in range(n_w-1):
            crop_x = a[(i*stride):((i*stride)+crop_size), (j*stride):((j*stride)+crop_size), :]
            croped_images.append(crop_x)
    return croped_images

#Not all above functions may be used below
#***********************************************************
#-----------------------------------------------------------
#***********************************************************

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


#Saving the cropped images
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

#--------------------------------------------------

#Converting training and validation ground truth images from rgb
#to onehot as required by model.fit in later code
trainy_hot = []

for i in range(trainy.shape[0]):
    
    hot_img = rgb_to_onehot(trainy[i], color_dict)
    
    trainy_hot.append(hot_img)
    
trainy_hot = np.asarray(trainy_hot)

valy_hot = []

for i in range(valy.shape[0]):
    
    hot_img = rgb_to_onehot(valy[i], color_dict)
    
    valy_hot.append(hot_img)
    
valy_hot = np.asarray(valy_hot)

print("Hot converted mask's size: " + str(valy_hot[0].shape))
print(np.unique(trainy_hot[4]))

#-------------------------------------------------
#Visualising the ground truth image
from PIL import Image
import numpy as np
img_pil = Image.fromarray(trainy[65].astype(np.uint8), 'RGB') 
display(img_pil.resize((256,256), PIL.Image.LANCZOS))

#Another way of visualising
%matplotlib inline 
import matplotlib.pyplot as plt
plt.imshow(valy[64].astype(np.uint8))
plt.show()


#-------------Transfer Learning------------------

from unet import UNet
pretrained_model = UNet()

pretrained_model.load_weights('model_onehot.h5')

for layer in pretrained_model.layers:
   layer.trainable = False

#You can select any layer you want. I have selected just the second last layer 

#This snippet selects the layer with name as last. You can open unet.py to see it.
#Search for layer with name as last
#This snippet changes the last few layers
# last_layer = pretrained_model.get_layer('last')
# last_output = last_layer.output
# conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(last_output)
# conv9 = Dropout(0.3)(conv9)
# conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv9)
# conv9 = Dropout(0.3)(conv9)
# conv9 = BatchNormalization()(conv9)
# # Output layer of the U-Net with a softmax activation
# #conv9 = Dropout(0.4, name ="my_drop")(conv9)
# conv10 = Conv2D(6, 1, activation = 'softmax')(conv9)


 #OR 


#This selects the second last layer and changes only the last layer
last_layer = pretrained_model.get_layer(index = -2)
last_output = last_layer.output
conv10 = Conv2D(6, 1, activation = 'softmax')(last_output)
from iou import iou
model = Model(pretrained_model.input, outputs = conv10)
model.compile(optimizer = 'adam', loss = 'bce_jaccard_loss', metrics = ['accuracy', iou])
model.summary()

#------------------------------------------------------------

#Data Augmentation
#This is optional to use but it is prefered for a small dataset
#Your results can change if you do or do not use it
trainx = trainx/np.max(trainx)
trainy_hot = trainy_hot/np.max(trainy_hot)
valx = valx/np.max(valx)
valy_hot = valy_hot/np.max(valy_hot)
train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip = True,
        height_shift_range = 0.2, width_shift_range = 0.2, rotation_range = 30, fill_mode = 'nearest')
        
val_datagen = ImageDataGenerator()

train_image_generator = train_datagen.flow(
trainx,
batch_size = 16, seed = 1)

train_mask_generator = train_datagen.flow(
trainy_hot,
batch_size = 16, seed =1)

val_image_generator = val_datagen.flow(
valx,
batch_size = 16, seed = 1)
# 
  
val_mask_generator = val_datagen.flow(
valy_hot,
batch_size = 16, seed = 1)



train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

#-----------------------------------------------------------

batch_size = 16

#Use this if you use Data Augmentation
history = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=int(np.ceil(num_val_examples / float(batch_size))),steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))), epochs = 10)
#Use this if you do not use Data Augmentation
history = model.fit(trainx, trainy_hot, validation_data=(valx, valy_hot), batch_size = 16, epochs = 10)
#Uncomment this below code if you want to save #weights                                                                                 
#model.save("model.h5")

#--------------------------------------------------------

#Predicting for our Model
def testing_diffsizes(model, trainx, valx, weights_file = None):
    
    pred_train_all = []
    pred_val_all = []
    
    #Uncomment if you are using saved weights
    #If you have trained the model just now, leave this commented
    #model.load_weights(weights_file)
    
    for i in range(len(trainx)):
        img = trainx[i]
        h,w,c = img.shape
        img = np.reshape(img, (1,h,w,c))
        Y_pred_train = model.predict(img)
        bb,h,w,c = Y_pred_train.shape
        Y_pred_train = np.reshape(Y_pred_train, (h,w,c))
        pred_train_all.append(Y_pred_train)
    

    for i in range(len(valx)):
        img = valx[i]
        h,w,c = img.shape
        img = np.reshape(img, (1,h,w,c))
        Y_pred_val = model.predict(img)
        bb,h,w,c = Y_pred_val.shape
        Y_pred_val = np.reshape(Y_pred_val, (h,w,c))
        pred_val_all.append(Y_pred_val)
    
    return pred_train_all, pred_val_all

pred_train_all, pred_val_all = testing_diffsizes(model, trainx, valx)

#----------------------------------------------------------

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
   
    n = len(Y_pred)
    
    for i in range(n):
        y_pred = Y_pred[i]
        y_gt = Y_gt[i]
        
        pred = np.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2]))
        gt = np.reshape(y_gt, (y_gt.shape[0]*y_gt.shape[1], y_gt.shape[2]))
        
        pred = [i for i in pred]
        gt = [i for i in gt]
        
        pred = to_class_no(pred)
        gt = to_class_no(gt)
        

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
print(confusion_matrix_val)
print('Kappa Coeff for validation without unclassified pixels')
print(kappa_test)

#--------------------------------------------------------------

def acc_of_class(class_label, conf_matrix, num_classes = 6):
    
    numerator = conf_matrix[class_label, class_label]
    
    denominator = 0
    
    for i in range(num_classes):
        denominator = denominator + conf_matrix[class_label, i]
        
    acc_of_class = numerator/denominator
    
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

# Convert decimal onehot encode from prediction to actual onehot code

# def dec_to_onehot(pred_all):
    
#     pred_all_onehot_list = []
    
#     for img in pred_all:
#         png
#         h, w, c = img.shape
        
#         for i in range(h):
#             for j in range(w):
                
#                 argmax_index = np.argmax(img[i,j])
                
#                 sudo_onehot_arr = np.zeros((6))
                
#                 sudo_onehot_arr[argmax_index] = 1
                
#                 onehot_encode = sudo_onehot_arr
                
#                 img[i,j,:] = onehot_encode
                
#         pred_all_onehot_list.append[img]
        
#     return pred_all_onehot_list

#---------------------------------------------------------------

#Predicting and saving outouts
#I have only saved first 4 outputs of validation set. 
#You can modify it for saving train predictions or more outputs
for i_ in range(len(valx[0:4])):
    
    item = valx[0:4][i_]
    
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
#     tif = TIFF.open(filelist_trainx[i_])
#     image2 = tif.read_image()
    
#     h,w,c = image2.shape
    
#     y_pred_train_img = y_pred_train_img[:h, :w, :]
    
    imx = Image.fromarray(y_pred_train_img)
    
    imx.save("Gn_predictions/pred"+str(i_)+".jpg")
