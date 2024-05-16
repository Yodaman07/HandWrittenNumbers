import PIL.ImageOps
import numpy
import tensorflow as tf
import keras.src

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv


# https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/ used for image to numpy conversion
# LOTS of help from this google tutorial
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb#scrollTo=oZTImqg_CaW1

class Model:
    def __init__(self, dataset):
        (self.train_img, self.train_label), (self.test_img, self.test_label) = dataset
        self.train_img = self.train_img / 255.0
        self.test_img = self.test_img / 255.0

    @staticmethod  # count is the number of images to set up for the model, start is for defining a new beginning index, the default one is 0
    def evalCustomImages(count: int,
                         start: int = 0) -> numpy.ndarray:  # loads images individually, converts it to a numpy array in grayscale, compresses it and groups them together
        group = []
        for i in range(start, count + start):
            img = Image.open(
                f"custom_training/num{i}.png").convert(
                "L")  # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image resizing help
            img = PIL.ImageOps.invert(img)  # implement auto invert
            numpyImg = numpy.array(
                img)  # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python <-- explains color channels with image loading
            compressed = cv.resize(numpyImg, dsize=(28, 28), interpolation=cv.INTER_AREA)

            group.append(compressed)

        groupedImages = numpy.array(group)
        groupedImages = groupedImages / 255.0
        return groupedImages

    @staticmethod  # img_list and label_list are the original labels and images. Predictions will be the predicted values
    def displayData(self, count, img_list, label_list, predictions: [] = None):
        col, row = 7, 7
        if count > (col * row):
            print("Cannot render: see the column and row counts for a possible fix")
            return None
        plt.figure(figsize=(12, 7))
        for c, i in enumerate(range(count)):
            plt.subplot(row, col, c + 1)
            plt.imshow(img_list[i])  # cmap=plt.cm.binary
            plt.grid(False)
            plt.subplots_adjust(top=0.95, hspace=0.98)
            if predictions is None:
                plt.xlabel(label_list[i])
            else:
                print(i)
                prediction = predictions[i]
                num = int(np.argmax(prediction))
                percentage = np.max(prediction) * 100
                color = 'red'
                if num == int(label_list[i]):
                    color = 'blue'
                else:
                    pass
                    # print(num, realLabels[i])
                plt.xlabel(f"{self.classNames[num]} ; {np.round(percentage, 2)}").set_color(color)

        plt.show()


class HandWrittenNumbersModel(Model):
    def __init__(self):
        super().__init__(keras.datasets.mnist.load_data(path="mnist.npz"))
        self.classNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.model = None

    def trainModel(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(258, activation="relu"),
            keras.layers.Dense(10)
        ])  # setting up the model
        model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])  # compiling everything, now ready to train the model
        model.fit(self.train_img, self.train_label)  # training the model
        self.model = model

    def predict(self, CUSTOM_IMAGE: numpy.ndarray = None):  # Predicts and graphs test images
        #  If CUSTOM_IMAGE is left blank, program will use the default testing images
        prediction_model = keras.Sequential([self.model, keras.layers.Softmax()])
        if CUSTOM_IMAGE is not None:
            predictions = prediction_model.predict(CUSTOM_IMAGE)
            self.displayData(self, 11, CUSTOM_IMAGE, ["1", "2", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                             predictions)
        else:
            predictions = prediction_model.predict(self.test_img)
            self.displayData(self, 49, self.test_img, self.test_label, predictions)


m = HandWrittenNumbersModel()
# model = Model(keras.datasets.mnist.load_data(path="mnist.npz"))
# model.evalCustomImages(2)
# model.displayData(2, model.evalCustomImages(2), ["1", "2"], False)

input("Ready? ")
m.trainModel()
input("Trained! Ready for the predictions? ")
m.predict(CUSTOM_IMAGE=m.evalCustomImages(11))
