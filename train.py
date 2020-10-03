import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import utils
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
#import keras
#from keras.models import Sequential
#from keras.optimizers import Adam
#from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath #edit the path
import random

import tensorflow as tf

from arch_collection import nvidia_model


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def path_leaf(path):
  #Triming the data by keeping the tail only, i.e. center/right/left_***.jpg
  head, tail = ntpath.split(path)
  return tail

def balance_data(thres, num_bins, data): 
  #Balancing the data
  remove_list = []
  hist, bins = np.histogram(data['steering'], num_bins) #bins = categories of steering angle
  for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
      if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
        list_.append(i) #categorizing the steering angle
    random.shuffle(list_)
    #print(len(list_))
    list_ = list_[thres: ] #eject samples that is beyond the threshold
    remove_list.extend(list_) #listing the unwanted data
  data.drop(data.index[remove_list], inplace=True) #removing the unwanted data by accessing their index
  return data, remove_list

def load_img_steering(datadir, df): #df = dataframe 
  image_path = []
  steering = []
  for i in range(len(df)):
    indexed_data = df.iloc[i] #iloc: select data via index i
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    image_path.append(os.path.join(datadir, center.strip()))#strip() to remove any spaces
    steering.append(float(indexed_data[3]))
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

def zoom(image):
  zoom = iaa.Affine(scale=(1,1.3))
  image = zoom.augment_image(image)
  return image

def pan(image):
  pan = iaa.Affine(translate_percent={'x': (-0.1,0.1), 'y': (-0.1,0.1)})
  image = pan.augment_image(image)
  return image

def img_random_brightness(image):
  brightness = iaa.Multiply((0.2,1.2))
  image = brightness.augment_image(image)
  return image

def img_random_flip(image, steering_angle):
  image = cv2.flip(image, 1)

  steering_angle = -steering_angle
  return image, steering_angle

def img_preprocess(img):
  img = img[60:135,:,:]
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img, (3,3), 0)
  img = cv2.resize(img, (200,66)) #Resize to fit NVidia data architecture
  return img/255

def random_augment(image, steering_angle):
  image = mpimg.imread(image)
  if np.random.rand() < 0.5:
    image = pan(image)
  if np.random.rand() < 0.5:
    image = zoom(image)
  if np.random.rand() < 0.5:
    image = img_random_brightness(image)
  if np.random.rand() < 0.5:
    image, steering_angle = img_random_flip(image, steering_angle)
  return image, steering_angle

def batch_generator(image_paths, steering_ang, batch_size, istraining):
  while True:
    batch_img = []
    batch_steering = []
    
    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths)-1)

      if istraining: 
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]
      im = img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.array(batch_img), np.asarray(batch_steering))


def main():
  #Retrieve the data
  dir_path = os.path.dirname(os.path.realpath(__file__))
  columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
  data = pd.read_csv(os.path.join(dir_path, 'data/driving_log.csv'), names = columns)
  pd.set_option('display.max_colwidth', None)
  data['center'] = data['center'].apply(path_leaf)
  data['left'] = data['left'].apply(path_leaf)
  data['right'] = data['right'].apply(path_leaf)
  print(data.head()) #Check-pt

  #Balancing the data
  print("\nBalancing the data...." )
  samples_per_bin = 500 #Threshold for uniforming the samples 
  num_bins = 25 #no. of categories of steering angles
  print('total data: ', len(data))
  balanced_data, remove_list = balance_data(samples_per_bin, num_bins, data)
  print('removed: {}     remaining: {}'.format(len(remove_list), len(data)))

  #Loading the img_path, steerings angle
  image_paths, steerings = load_img_steering(dir_path + '\data\IMG', balanced_data)

  #Splitting Data
  print("\nSplitting the Data....")
  X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
  print('Training Samples: {}     Valid Samples: {}'. format(len(X_train), len(X_valid)))

  #Data Augmentation
  '''
  # Check point
  image = image_paths[100]
  print(image)
  original_image = mpimg.imread(image)
  preprocessed_image = zoom(original_image)

  fig, axs = plt.subplots(1, 2, figsize=(15,10))
  fig.tight_layout()
  axs[0].imshow(original_image)
  axs[0].set_title('original_image')
  axs[1].imshow(preprocessed_image)
  axs[1].set_title('preprocessed_image')
  plt.show()
  '''

  #Batch Generator
  '''
  X_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 2, 1))
  X_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 2, 0))
  fig, axs = plt.subplots(1, 2, figsize=(15,10))
  fig.tight_layout()
  axs[0].imshow(X_train_gen[0])
  axs[0].set_title('Training_image')
  axs[1].imshow(X_valid_gen[0])
  axs[1].set_title('Valid_image')
  plt.show()
  '''
  #Import Model
  model = nvidia_model()
  print(model.summary())

  history = model.fit(batch_generator(X_train, y_train,130, 1),
                                steps_per_epoch=350, 
                                epochs=10, 
                                validation_data=batch_generator(X_valid, y_valid,130,0),
                                validation_steps=250,
                                verbose=1, shuffle=1)

  save = input("Would like to save trained model as model.h5? y/n")
  if save == "y":
    model.save('model.h5')
    print("model is saved")
  else:
    print("model is not saved.")
    pass
if __name__== "__main__":
  main()