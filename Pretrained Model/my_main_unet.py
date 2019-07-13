import PIL
from PIL import Image
import matplotlib.pyplot as plt
from libtiff import TIFF
from libtiff import TIFFfile, TIFFimage
from scipy.misc import imresize
import numpy as np
import math
import glob
import cv2
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as keras
%%matplotlib inline

#Defines iou metric for accuracy
def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    #sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    iou_acc = (intersection + smooth) / (union + smooth)
    return iou_acc

#For converting to Keras metric to used while model.fit
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

def tf_mean_iou(y_true, y_pred, num_classes=8):
    return tf.metrics.mean_iou(y_true, y_pred, num_classes)


mean_iou = as_keras_metric(tf_mean_iou)


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


def unet(shape = (None,None,4)):
    
    # Left side of the U-Net
    inputs = Input(shape)
#    in_shape = inputs.shape
#    print(in_shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bottom of the U-Net
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Upsampling Starts, right side of the U-Net
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    # Output layer of the U-Net with a softmax activation
    conv10 = Conv2D(9, 1, activation = 'softmax')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model.summary()
    
    #filelist_modelweights = sorted(glob.glob('*.h5'), key=numericalSort)
    
    #if 'model_nocropping.h5' in filelist_modelweights:
     #   model.load_weights('model_nocropping.h5')
    ##model.load_weights("model_onehot.h5")
    return model

#******************************************************************
#******************************************************************

model = unet()

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

#--------------------------------------------

#Training the model
history = model.fit(trainx, trainy_hot, epochs=20, validation_data = (testx, testy_hot),batch_size=64, verbose=1)
model.save("model_onehot.h5")

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('acc_plot.png')
plt.show()
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('loss_plot.png')
plt.show()
plt.close()
