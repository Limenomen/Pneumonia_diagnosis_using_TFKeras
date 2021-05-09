import tensorflow as tf
import numpy as np
from cv2 import cv2
import os
import PIL
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Flatten, Conv2D, Activation, Dense, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import seaborn as sns
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

epochs_count = 10


def get_images_data(dir):
    img_size = 150
    data = []
    x_arr = []
    y_arr = []
    for label in labels:
        path = os.path.join(dir, label)
        class_num = labels.index(label)
        for img_path in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_path),
                             cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, (img_size, img_size))
            data.append([resized_img, class_num])
    for img, label in data:
        x_arr.append(img)
        y_arr.append(label)
    x_arr = np.array(x_arr) / 255
    y_arr = np.array(y_arr)
    x_arr = x_arr.reshape(-1, img_size, img_size, 1)

    return(data, x_arr, y_arr)


def graphics():
    epochs = [i for i in range(epochs_count)]
    fig, ax = plt.subplots(1, 2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    fig.set_size_inches(20, 10)

    ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
    ax[0].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'r-o', label='Validation Loss')
    ax[1].set_title('Testing Accuracy & Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Training & Validation Loss")
    plt.show()
    l = []
    for i in train:
        if(i[1] == 0):
            l.append("Pneumonia")
        else:
            l.append("Normal")
    sns.set_style('darkgrid')
    sns.countplot(l)


#'Путь, где находятся изображения для датасета'
train_path = '../../chest_xray/train'
test_path = '../../chest_xray/test'
val_path = '../../chest_xray/val'

labels = ['Pneumonia', 'Normal']

val, x_val, y_val = get_images_data(val_path)
train, x_train, y_train = get_images_data(train_path)
test, x_test, y_test = get_images_data(test_path)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=30,
    zoom_range=0.2,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


model = Sequential()

model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1 , activation = 'sigmoid'))
model.compile(optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()

"""
earlystopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=3,
                              verbose=1)
                              """
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)


history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=epochs_count,
                    validation_data=datagen.flow(x_val, y_val), callbacks=[learning_rate_reduction])


print("Loss of the model is - ", model.evaluate(x_test, y_test)[0])
print("Accuracy of the model is - ",
      model.evaluate(x_test, y_test)[1]*100, "%")

graphics()
