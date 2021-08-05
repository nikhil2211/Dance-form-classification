# Dance-form-classification
This task was released by HackerEarth (https://www.hackerearth.com/challenges/competitive/hackerearth-deep-learning-challenge-identify-dance-form/) and has a very small training dataset - only 364 training images in total.

India is a culturally rich country where almost every state has its own language, cuisine, and dance forms. However, identifying them with just images is a difficult task.

The given dataset has the following 8 classes:

Manipuri
Bharatanatyam
Odissi
Kathakali
Kathak
Sattriya
Kuchipudi
Mohiniyattam
I have used VGG16,Xception as the base_model and added a Pooling Layer, Fully Connected Layers and Dropout Layers on the top of it before predicting the classes as model alone will not give good results. Since the dataset is very small, I have used the image augmentation techniques using OpenCV to generate new augmented images for the training purpose.

On training, I achieved validation accuracy of 90% - 93% which is pretty good.
