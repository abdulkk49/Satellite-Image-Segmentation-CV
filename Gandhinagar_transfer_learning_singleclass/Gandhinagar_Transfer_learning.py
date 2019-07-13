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

#You need to install this library as well
#!pip3 install git+https://github.com/qubvel/segmentation_models

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

def tf_mean_iou(y_true, y_pred, num_classes=2):
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
    
    # Slicing the image into 256*256 crops with a stride of 64
    for i in range(n_h-1):
        for j in range(n_w-1):
            crop_x = a[(i*stride):((i*stride)+crop_size), (j*stride):((j*stride)+crop_size), :]
            croped_images.append(crop_x)
    return croped_images

#--------------------------------------------------------
#********************************************************
#--------------------------------------------------------

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
filelist_trainy = sorted(glob.glob('road.jpg'), key=numericalSort)

#Making array of all images and cropping them

trainx_list = []

for fname in filelist_trainx[:1]:
    
    # Reading the image
    tif = TIFF.open(fname)
    image = tif.read_image()
    print(np.unique(image[:,:,1]))
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

# Making array of all the *training gt images* as it is without any cropping. Padding all images

trainy_list = []
import matplotlib.image as mpimage
for fname in filelist_trainy[:1]:

    # Reading the image
    image = mpimage.imread(fname)
    
    print(np.unique(image))
    image[image > 0] = 1 
    image = np.expand_dims(image, axis =2)
    print(np.unique(image))
#     print(np.unique(image))
    print('Original mask shape:')
    print(image.shape)
    # Padding as required and cropping
    crops_list = crops(image)
    
    trainy_list = trainy_list + crops_list
 
# Array of all the cropped Training gt Images    
y = np.asarray(trainy_list)
print('y length:')
print(len(y)) 

#---------------------------------------------
#Splitting into validation and training sets
   
import random
from sklearn.model_selection import train_test_split
trainx, valx, trainy, valy = train_test_split(X, y, test_size=0.2, random_state=281)
num_train_examples = len(trainx)
num_val_examples = len(valx)


print('Num of training examples:' + str(len(trainx)))
print('Num of validation examples:' + str(len(valx)))
print('Original image shape after cropping:' + str(trainx[0].shape))
print('Mask shape after cropping:' + str(trainy[0].shape))


#Saving Ground truth Images
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
# for i in range(num_val_examples):
#    scipy.misc.imsave('Gn_crop_val_gt/val_gt' + str(i)+'.jpg', valy[i])

#Mask Visualisation
print(trainy[8].shape)
image = trainy[3]
print(np.unique(trainy[3]))
image[image > 0] = 255
image = image.reshape(image.shape[0], image.shape[1])
print(image.shape)
from matplotlib import pyplot as plt
plt.imshow(image, interpolation='nearest')
plt.show()

#---------------Transfer LEarning---------------
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from segmentation_models.utils import set_trainable
from keras.layers import Input, Conv2D
from keras.models import Model

# read/scale/preprocess data

# define number of channels
N = trainx.shape[-1]

base_model = Unet(backbone_name='resnet34', encoder_weights='imagenet', freeze_encoder=True)

inp = Input(shape=(None, None, N))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)

model = Model(inp, out, name=base_model.name)

model.summary()

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score
from tensorflow.python.keras import losses
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

model.compile(optimizer = 'adam', loss = bce_jaccard_loss , metrics = ['accuracy', iou_score, dice_coeff])

#Data Augmentation
#This is optional to use but it is prefered for a small dataset
#Your results can change if you do or do not use it

trainx = trainx/np.max(trainx)
trainy_hot = trainy/np.max(trainy)
valx = valx/np.max(valx)
valy_hot = valy/np.max(valy)
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
trainy,
batch_size = 16, seed =1)

val_image_generator = val_datagen.flow(
valx,
batch_size = 16, seed = 1)
# 
  
val_mask_generator = val_datagen.flow(
valy,
batch_size = 16, seed = 1)



train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)


# # #Data Augmentation

# datagen_args = dict(rotation_range=45.,
#                           width_shift_range=0.1,
#                           height_shift_range=0.1,
#                           shear_range=0.2,
#                           zoom_range=0.2,
#                           horizontal_flip=True,
#                           vertical_flip=True,
#                           fill_mode='reflect')

# x_datagen = ImageDataGenerator(**datagen_args)
# y_datagen = ImageDataGenerator(**datagen_args)

# seed = 1
# batch_size = 16
# x_datagen.fit(trainx, augment=True, seed = seed)
# y_datagen.fit(trainy_hot, augment=True, seed = seed)

# x_generator = x_datagen.flow(trainx, batch_size = 16, seed=seed)
  
# y_generator = y_datagen.flow(trainy_hot, batch_size = 16, seed=seed)

# train_generator = zip(x_generator, y_generator)

# X_datagen_val = ImageDataGenerator()
# Y_datagen_val = ImageDataGenerator()
# X_datagen_val.fit(valx, augment=True, seed=seed)
# Y_datagen_val.fit(valy_hot, augment=True, seed=seed)
# X_val_augmented = X_datagen_val.flow(valx, batch_size=batch_size, seed=seed)
# Y_val_augmented = Y_datagen_val.flow(valy_hot, batch_size=batch_size, seed=seed)

# test_generator = zip(X_val_augmented, Y_val_augmented)

batch_size = 16

#Use this if you use Data Augmentation
history = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=int(np.ceil(num_val_examples / float(batch_size))),steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))), epochs = 10)
#Use this if you do not use Data Augmentation
history = model.fit(trainx, trainy_hot, validation_data=(valx, valy_hot), batch_size = 16, epochs = 10)
#Uncomment this below code if you want to save weights                                                                                 
#model.save("model.h5")

#--------------------------------------------
#Now unfreeze all layers and train again
#This is done to improve accuracy
from segmentation_models.utils import set_trainable
set_trainable(model)
#set all layers trainable and recompile model
model.summary()

#Use this if you use Data Augmentation
history = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=int(np.ceil(num_val_examples / float(batch_size))),steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))), epochs = 10)
#Use this if you do not use Data Augmentation
history = model.fit(trainx, trainy, validation_data=(valx, valy), batch_size = 16, epochs = 10)
#Uncomment this below code if you want to save weights                                                                                 
#model.save("model.h5")
#---------------------------------------------------


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
acc_graph = plt.figure()
# acc_graph.savefig('Gn_plots/accuracy_plots_further.png')
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation Loss')
plt.legend(loc=0)
loss_graph = plt.figure()
# loss_graph.savefig('Gn_plots/loss_plots_further.png')

plt.show()

#----------------------------------------------------

#Predicting masks

idx = random.randint(0, len(trainx))
x=np.array(trainx[idx])
x=np.expand_dims(x, axis=0)
predict = model.predict(x, verbose=1)
 
predict = (predict > 0.5).astype(np.uint8)
print(predict.shape)
plt.imshow(np.squeeze(predict[0]))
print(np.squeeze(predict[0]).shape)
print(predict.shape)
plt.show()
 
image = trainy[idx]
image[image > 0] = 255
image = image.reshape(image.shape[0], image.shape[1])
print(image.shape)
from matplotlib import pyplot as plt
plt.imshow(image, interpolation='nearest')

plt.show()
