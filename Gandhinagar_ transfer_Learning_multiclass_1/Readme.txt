This model uses Gandhinagar image as training data.
The Gandhinagar image is cropped to get 258 training images and 65 validation images.
Transfer learning is used to get results.

The model uses weights from eye in the sky manideep repository for transfer learning.
This is the model 1 for multiclass transfer learning.
This is not a standard model and is not recommended.

In order to use this,
Download weights from above github repository and save them as 'model_onehot.h5' and run the code.
Make directory named Gn_predictions to save the outputs.
Take a look at the code. It is thoroughly commented.

There is one this I haven't tried. 
Once the last layers are trained using transfer leaning, you can
unfreeze all layers and train again.
This might lead to better accuracy for this model.
Look at Keras documnetation on how to unfreeze the model layers.
