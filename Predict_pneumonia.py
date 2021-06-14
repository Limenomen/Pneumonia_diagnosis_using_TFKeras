from typing import Text
from myUI import Ui_MainWindow
import numpy as np
from cv2 import cv2
from keras.models import load_model
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog
import os
import sys
import shutil


class mywindow(QtWidgets.QMainWindow):

    fileName = ""

    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.invisible_label.hide()
        self.ui.invisible_label_2.hide()
        self.ui.overview_button.clicked.connect(self.buttonClicked)
        self.ui.pneumonia_button.clicked.connect(self.pneumonia_button_clicked)
        self.ui.normal_button.clicked.connect(self.normal_button_clicked)

    def color_label(self):
        if (self.ui.result_label.text() != self.ui.result_label_2.text()):
            self.ui.result_label.setStyleSheet("QLabel { color: red}")
        else:
            self.ui.result_label.setStyleSheet("QLabel { color: green}")

    def buttonClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Jpg Images (*.jpeg)", options=options)
        pixmap = QtGui.QPixmap(self.fileName)
        pixmap = pixmap.scaled(331, 331, QtCore.Qt.KeepAspectRatio)
        self.ui.label_2.setPixmap(pixmap)
        class_name = get_class_name(self.fileName)
        result = prediction(self.fileName)
        self.ui.result_label.setText(result)
        self.ui.result_label_2.setText(class_name)
        self.color_label()
        self.ui.invisible_label.show()
        self.ui.invisible_label_2.show()
        print(result)

    def pneumonia_button_clicked(self):
        directory_path = os.path.abspath('../../chest_xray/train/PNEUMONIA')
        self.ui.result_label_2.setText('Пневмония')
        self.color_label()
        shutil.copy(self.fileName, directory_path)

    def normal_button_clicked(self):
        directory_path = os.path.abspath('../../chest_xray/train/NORMAL')
        self.ui.result_label_2.setText('Норма')
        self.color_label()
        shutil.copy(self.fileName, directory_path)


model = load_model('my_model.h5')
model.compile(optimizer="adam", loss='binary_crossentropy',
              metrics=['accuracy'])


def get_class_name(path):
    class_name = path.split('/')[-2]
    if class_name == 'PNEUMONIA':
        return 'Пневмония'
    elif class_name == 'NORMAL':
        return 'Норма'
    else:
        return '???'


def prediction(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = np.array(img) / 255
    img = img.reshape(1, 150, 150, 1)
    result = model.predict(img)
    if result > 0.5:
        return 'Норма'
    elif result <= 0.5:
        return 'Пневмония'


app = QtWidgets.QApplication([])
application = mywindow()
application.show()
sys.exit(app.exec())
