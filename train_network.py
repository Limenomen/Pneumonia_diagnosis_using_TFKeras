import tensorflow as tf
import numpy as np
import cv2
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
    data = []
    for label in labels:
        path = os.path.join(dir, label)
        for img in os.listdir(path):
            image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_img = 


    

