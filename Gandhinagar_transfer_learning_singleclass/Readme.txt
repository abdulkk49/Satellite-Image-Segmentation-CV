This model uses Gandhinagar image as training data.
We use just a single class for prediction. I have used roads. You can use any class. Just change the image in glob.glob. You can find training ground truth images in TIF Masks Corrected folder or JPG
Masks Corrected folder
The Gandhinagar image is cropped to get 258 training images and 65 validation images.
Transfer learning is used to get results.

The model uses library called segmentation models.
You need to pip install it to run the model. See the documentation:
https://segmentation-models.readthedocs.io/en/latest/
This is the model 2 for multiclass transfer learning.
This is a standard model and is recommended.
You can improve using this model.
The main advantage this has over the first model is that it can take any number of channels . They are then mapped to three channels usinf Convnet and imagenet satndard weights are used. So accuracy can be better with this model.
Also look at github repository for quvbel/segmentation models:
https://github.com/qubvel/segmentation_models

In order to use this,
Make directory named Gn_predictions to save the outputs.
Take a look at the code. It is thoroughly commented.
