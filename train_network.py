import tensorflow as tf
import numpy as np
import cv2
import PIL
from tensorflow import keras
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4, 5],[1, 2, 3, 4, 5])
plt.show()


#'Путь, где находятся изображения для датасета'
train_path = '../../chest_xray/train'
test_path = '../../chest_xray/test'
val_path = '../../chest_xray/val'


def load_img():
    

