import tensorflow as tf
import numpy as np
from cv2 import cv2
import os
import PIL
from tensorflow import keras
import matplotlib.pyplot as plt




#'Путь, где находятся изображения для датасета'
train_path = '../../chest_xray/train'
test_path = '../../chest_xray/test'
val_path = '../../chest_xray/val'
labels = ['Pneumonia', 'Normal']

def get_images_data(dir):
    img_size = 150
    data = []
    for label in labels:
        path = os.path.join(dir, label)
        class_num = labels.index(label)
        for img_path in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_path), cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, (img_size, img_size))
            data.append([resized_img, class_num])
    data = np.array(np.array(data) / 255)
    return(data)

val = get_images_data(val_path)
train = get_images_data(train_path)
test = get_images_data(test_path)



