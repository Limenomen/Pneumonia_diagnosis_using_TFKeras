import tensorflow as tf
import numpy as np
import cv2
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical



#'Путь, где находятся изображения для датасета'
train_path = '../chest_xray/train'
test_path = '../chest_xray/test'

train_gen = tf.keras.preprocessing.image_dataset_from_directory(
  train_path,
  validation_split=0.2,
  color_mode="grayscale",
  batch_size=32,
  image_size=(180, 180),
  shuffle=True,
  seed=123,
  subset="training",
)

test_gen = tf.keras.preprocessing.image_dataset_from_directory(
  test_path,
  validation_split=0.2,
  color_mode="grayscale",
  batch_size=32,
  image_size=(180, 180),
  shuffle=True,
  seed=123,
  subset="validation",
)


model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(2)
])


model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history = model.fit(
  train_gen,
  validation_data=test_gen,
  epochs=3
)

model.save('my_model')
