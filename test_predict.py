import tensorflow as tf
import numpy as np
from cv2 import cv2
import random
import os
import PIL
from tensorflow import keras
import keras
import matplotlib.pyplot as plt
from keras.models import load_model
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix, confusion_matrix

params = {'legend.fontsize': 'small',
          'figure.figsize': (15, 10),
          'axes.labelsize': 'small',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small'}
plt.rcParams.update(params)


def get_Images_Data(dir):
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
            data.append([img, label])
            x_arr.append(resized_img)
            y_arr.append(class_num)
    x_arr = np.array(x_arr) / 255
    x_arr = x_arr.reshape(-1, img_size, img_size, 1)
    y_arr = np.array(y_arr)

    return(data, x_arr, y_arr)


def plot_samples(data):
    for i in range(1, 10):
        image = random.choice(data)
        image[1] = add_class_name(image[1])
        plt.subplot(3, 3, i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image[0], cmap="gray", interpolation='none')
        plt.title("{}".format(image[1]))
        plt.tight_layout()
    plt.show()


def add_class_name(data):
    new_data = []
    for element in data:
        if element == 0 or element == 'Pneumonia':
            new_data.append('Пневмония')
        else:
            new_data.append('Норма')
    return new_data


def confusion_Matrix(predictions):
    m = confusion_matrix(y_test, predictions)
    print(m)
    plot_confusion_matrix(y_test, predictions)

def predict_Images(predictions):
    
    correct = []
    incorrect = []
    for index in range(len(predictions)):
        if predictions[index] == test[index][1]:
            correct.append([test[index][0], predictions[index], test[index][1]])
        else:
            incorrect.append(
                [test[index][0], predictions[index], test[index][1]])
    print(len(correct), len(incorrect))
    data = correct[:len(incorrect)] + incorrect

    for i in range(1, 16):
        d = random.choice(data)
        plt.subplot(4, 4, i)
        plt.xticks([])
        plt.yticks([])
        plt.title(
            "определено как:{0},\n на самом деле: {1}".format(d[1], d[2]))
        plt.imshow(d[0], cmap="gray", interpolation='none')
        plt.tight_layout()
    plt.show()


test_path = '../../chest_xray/test'
test, x_test, y_test = get_Images_Data(test_path)


model = load_model('my_model.h5')
model.compile(optimizer="adam", loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


evaluate = model.evaluate(x_test, y_test)
print("Loss of the model is - ", evaluate[0])
print("Accuracy of the model is - ",
      evaluate[1]*100, "%")
predictions = model.predict_classes(x_test)
predictions = predictions.reshape(1, -1)[0]
predictions = add_class_name(predictions)

predict_Images(predictions)
plot_samples(test)

print('successful')
