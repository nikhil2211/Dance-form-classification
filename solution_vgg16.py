# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 00:55:45 2020

@author: nikhi
"""
#importing libraries
import scipy as sp
import seaborn as sns
import scipy.misc
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import PIL

#importing dataset
data_labels = pd.read_csv("train.csv")
target_labels = data_labels['target']

"""What we do next is to add in the exact image path for each image present in 
the disk using the following code"""
train_folder = 'train/'
data_labels['image_path'] = data_labels.apply(lambda row: (train_folder + row["Image"] ), 
                                              axis=1)
data_labels.head()

"""
It’s now time to prepare our train, test and validation datasets.
We will leverage the following code to help us build these datasets!"""

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array,load_img

# load dataset
train_data = np.array([img_to_array(load_img(img, target_size=(299, 299)))
                           for img in data_labels['image_path'].values.tolist()
                      ]).astype('float32')

# create train and test datasets
x_train, x_test, y_train, y_test = train_test_split(train_data, target_labels, 
                                                    test_size=0.2, 
                                                    stratify=np.array(target_labels), 
                                                    random_state=0)

# create train and validation datasets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                  test_size=0.15, 
                                                  stratify=np.array(y_train), 
                                                  random_state=0)

print('Initial Dataset Size:', train_data.shape)
print('Initial Train and Test Datasets Size:', x_train.shape, x_test.shape)
print('Train and Validation Datasets Size:', x_train.shape, x_val.shape)
print('Train, Test and Validation Datasets Size:', x_train.shape,
      x_test.shape, x_val.shape)

"""
We also need to convert the text class labels to one-hot encoded labels
, else our model will not run.
"""

y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).as_matrix()
y_test_ohe = pd.get_dummies(y_test.reset_index(drop=True)).as_matrix()

y_train_ohe.shape, y_test_ohe.shape, y_val_ohe.shape

#data augmentation
from keras.preprocessing.image import ImageDataGenerator
BATCH_SIZE = 32

# Create train generator.
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=30, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   horizontal_flip = 'true')
train_generator = train_datagen.flow(x_train, y_train_ohe, shuffle=False, 
                                     batch_size=BATCH_SIZE, seed=1)
                                     
# Create validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = train_datagen.flow(x_val, y_val_ohe, shuffle=False, 
                                   batch_size=BATCH_SIZE, seed=1) 

"""
Transfer Learning with VGG16 Model
"""

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.applications import VGG16
from keras.utils.np_utils import to_categorical

# Get the VGG16 model so we can do transfer learning
base_inception = VGG16(weights='imagenet', include_top=False, 
                             input_shape=(299, 299, 3))
                             
# Add a global spatial average pooling layer
out = base_inception.output
out = GlobalAveragePooling2D()(out)
out = Dense(512, activation='relu')(out)
#out = Dropout(0.5)(out)
out = Dense(512, activation='relu')(out)
total_classes = y_train_ohe.shape[1]
predictions = Dense(total_classes, activation='softmax')(out)

model = Model(inputs=base_inception.input, outputs=predictions)

# only if we want to freeze layers
for layer in base_inception.layers:
    layer.trainable = False
    
# Compile 
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy']) 
model.summary()

# Train the model
batch_size = BATCH_SIZE
train_steps_per_epoch = x_train.shape[0] // batch_size
val_steps_per_epoch = x_val.shape[0] // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs=600, verbose=1)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# scaling test features
x_test /= 255.

# getting model predictions
test_predictions = model.predict(x_test)
labels_ohe_names = pd.get_dummies(target_labels, sparse=True)
predictions = pd.DataFrame(test_predictions, columns=labels_ohe_names.columns)
predictions = list(predictions.idxmax(axis=1))
test_labels = list(y_test)

# evaluate model performance
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=labels_ohe_names.columns, yticklabels=labels_ohe_names.columns)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#predicting the test images
test_folder ='test/'
test_data = pd.read_csv('test.csv')

test_data['image path'] = test_data.apply(lambda row :(test_folder + row['Image']),axis =1)
test_data.head()

#converting images into arrays
test_data_fin = np.array([img_to_array(load_img(img, target_size=(299, 299)))
                           for img in test_data['image path'].values.tolist()
                      ]).astype('float32')
#making final predictions
test_data_fin /=255.
test_predictions_fin = model.predict(test_data_fin)
labels_ohe_names = pd.get_dummies(target_labels, sparse=True)
predictions_fin = pd.DataFrame(test_predictions_fin, columns=labels_ohe_names.columns)
predictions_fin = list(predictions_fin.idxmax(axis=1))

#adding the predictions to the csv file
test_data['target'] = predictions_fin
test_data = test_data.drop(['image path'],axis =1)
#saving the csv file
test_data.to_csv('test_vgg16.csv',header =False,index = False)