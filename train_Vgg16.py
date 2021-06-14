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
from keras.applications.vgg16 import VGG16


epochs_count = 13

def get_images_data(dir):
    #метки классов
    labels = ['PNEUMONIA', 'NORMAL']
    #размер выходного изображения
    img_size = 150
    data = []
    x_arr = []
    y_arr = []
    #проход по текущей директории
    for label in labels:
        #путь, где находится текущее изображение
        path = os.path.join(dir, label)
        #индекс класса
        class_num = labels.index(label)
        #выбираем изображения, которые лежат в директории, соответствующей данному классу
        for img_path in os.listdir(path):
            #считывание изображения
            img = cv2.imread(os.path.join(path, img_path),
                             cv2.IMREAD_GRAYSCALE)
            #изменение размера изображения
            resized_img = cv2.resize(img, (img_size, img_size))
            x_arr.append(resized_img)
            y_arr.append(class_num)
    #преобразование в numpy array, нормализация
    x_arr = np.repeat(x_arr, 3)
    x_arr = np.array(x_arr) / 255
    y_arr = np.array(y_arr)
    x_arr = x_arr.reshape(-1, img_size, img_size, 3)
    #plt.imshow(x_arr[2][1][1][1], cmap="gray", interpolation='none')
    #plt.show()
    return(x_arr, y_arr)


def graphics():

    l = []
    for i in y_train:
        if(i[1] == 0):
            l.append("Пневмония")
        else:
            l.append("Норма")
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


x_val, y_val = get_images_data(val_path)
x_train, y_train = get_images_data(train_path)
x_test, y_test = get_images_data(test_path)


datagen = ImageDataGenerator(
    #случайный поворот изображения
    rotation_range=30,
    #случайное приближение 
    zoom_range=0.2, 
    #смещение по горизонтали 
    width_shift_range=0.1,
    #смещение по вертикали
    height_shift_range=0.1,
    #отражение по горизонтали
    horizontal_flip=True)  
datagen.fit(x_train)


vgg16_net = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
vgg16_net.trainadle = True

model = Sequential()
model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer="adam", loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=epochs_count,
                    validation_data=datagen.flow(x_val, y_val), callbacks=[learning_rate_reduction])

evaluate = model.evaluate(x_test, y_test)

print("точность - ",
      evaluate[1]*100, "%")

model.save('my_model.h5')
graphics()