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
from sklearn.metrics import confusion_matrix

params = {'legend.fontsize': 'small',
          'figure.figsize': (15, 10),
          'axes.labelsize': 'small',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small'}
plt.rcParams.update(params)

def Get_Images_Data(dir):
    labels = ['PNEUMONIA', 'NORMAL']
    labels1 = ['Пневмония', 'Норма']
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
            data.append([img, labels1[class_num]])
            x_arr.append(resized_img)
            y_arr.append(class_num)
    x_arr = np.array(x_arr) / 255
    x_arr = x_arr.reshape(-1, img_size, img_size, 1)
    y_arr = np.array(y_arr)
    return(data, x_arr, y_arr)


def Predict_Images(Predictions, Test):
    correct = []
    incorrect = []
    for index in range(len(Predictions)):
        if Predictions[index] == Test[index][1]:
            correct.append(
                [Test[index][0], Predictions[index], Test[index][1]])
        else:
            incorrect.append(
                [Test[index][0], Predictions[index], Test[index][1]])
    print(len(correct), len(incorrect))
    data = correct[:len(incorrect)] + incorrect
    for i in range(1, 17):
        d = random.choice(data)
        plt.subplot(4, 4, i)
        plt.xticks([])
        plt.yticks([])
        plt.title(
            "определено как:{0},\n на самом деле: {1}".format(d[1], d[2]))
        plt.imshow(d[0], cmap="gray", interpolation='none')
        plt.tight_layout()
    plt.show()


def Samples(data):
    for i in range(1, 10):
        image = random.choice(data)
        plt.subplot(3, 3, i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image[0], cmap="gray", interpolation='none')
        plt.title("{}".format(image[1]))
        plt.tight_layout()
    plt.show()


def Change_Class_Name(data):
    new_data = []
    for element in data:
        if element == 0:
            new_data.append('Пневмония')
        else:
            new_data.append('Норма')
    return new_data


def Confusion_Matrix(predictions, Test):
    labels=['Пневмония', 'Норма']
    m = confusion_matrix(Test, predictions, labels=labels)
    print(m)
    plt.figure(figsize=(10, 10))
    sns.heatmap(m, cmap="Blues", linecolor='black', linewidth=1,
                annot=True, fmt='', xticklabels=labels, yticklabels=labels)
    plt.show()

test_path = '../../chest_xray/test'
test, x_test, y_test = Get_Images_Data(test_path)

model = load_model('my_model.h5')
model.compile(optimizer="adam", loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
evaluate = model.evaluate(x_test, y_test)
print("Точность - ",
      evaluate[1]*100, "%")


predictions = model.predict_classes(x_test)
predictions = predictions.reshape(1, -1)[0]
y_pred = Change_Class_Name(predictions)
y_true = [x[1] for x in test]
Confusion_Matrix(y_pred, y_true)
Samples(test)
Predict_Images(y_pred, test)
