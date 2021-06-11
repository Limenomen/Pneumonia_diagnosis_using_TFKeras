import numpy as np
from cv2 import cv2
import random
import os
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

epochs_count = 13


def get_images_data(dir):
    labels = ['Pneumonia', 'Normal']
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

    l = []
    for i in train:
        if(i[1] == 0):
            l.append("Pneumonia")
        else:
            l.append("Normal")
    sns.set_style('darkgrid')
    sns.countplot(l)

    epochs = [i for i in range(epochs_count)]
    fig, ax = plt.subplots(1, 2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    fig.set_size_inches(2, 1)

    ax[0].plot(epochs, train_acc, 'go-',
               label='точность на тренировочной выборке')
    ax[0].plot(epochs, val_acc, 'ro-',
               label='точность на валидационной выборке')
    ax[0].set_title('Точность')
    ax[0].legend()
    ax[0].set_xlabel("эпоха")
    ax[0].set_ylabel("точность")

    ax[1].plot(epochs, train_loss, 'g-o',
               label='потери на тренировочной выборке')
    ax[1].plot(epochs, val_loss, 'r-o',
               label='потери на валидационной выборке')
    ax[1].set_title('потери')
    ax[1].legend()
    ax[1].set_xlabel("эпоха")
    ax[1].set_ylabel("потери")
    plt.show()


#'Путь, где находятся изображения для датасета'
train_path = '../../chest_xray/train'
test_path = '../../chest_xray/test'
val_path = '../../chest_xray/val'


val, x_val, y_val = get_images_data(val_path)
train, x_train, y_train = get_images_data(train_path)
test, x_test, y_test = get_images_data(test_path)


datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,  
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,  
    vertical_flip=False)  
datagen.fit(x_train)


model = Sequential()
model.add(Conv2D(32, (3, 3), strides=1, padding='same',
                 activation='relu', input_shape=(150, 150, 1)))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer="adam", loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

earlystopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=5,
                              verbose=1)

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=epochs_count,
                    validation_data=datagen.flow(x_val, y_val), callbacks=[learning_rate_reduction])


evaluate = model.evaluate(x_test, y_test)
print("Loss of the model is - ", evaluate[0])
print("Accuracy of the model is - ",
      evaluate[1]*100, "%")
if (evaluate[1]*100 > 91):
    model.save('my_modddel.h5')
graphics()
