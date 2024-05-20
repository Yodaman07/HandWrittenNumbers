import PIL.ImageOps
import matplotlib.image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import os


# https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/ used for image to numpy conversion
# LOTS of help from this google tutorial
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb#scrollTo=oZTImqg_CaW1
# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python <-- explains color channels with image loading
# https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image resizing help

class Model:
    def __init__(self, dataset):
        (self.train_img, self.train_label), (self.test_img, self.test_label) = dataset
        self.train_img = self.train_img / 255.0
        self.test_img = self.test_img / 255.0

    # Loads images individually, makes it grayscale it in a numpy array and compressed it

    # Count = num of images set up for the model
    # Start = new beginning index (default is 0)
    @staticmethod
    def evalCustomImages(count: int, start: int = 0) -> np.ndarray:
        group = []
        for i in range(start, count + start):
            img = Image.open(
                f"custom_predictions/num{i}.png").convert("L")
            img = PIL.ImageOps.invert(img)  # TODO: implement auto invert

            npImg = np.array(img)
            compressed = cv.resize(npImg, dsize=(28, 28), interpolation=cv.INTER_AREA)
            group.append(compressed)

        groupedImages = np.array(group)
        groupedImages = groupedImages / 255.0
        return groupedImages

    # Only works with CUSTOM IMAGES not the testing images
    @staticmethod
    def saveCustomImages(images: np.ndarray, label):  # `images` are grayscale
        # Gets the last used index
        files = os.listdir("custom_training")

        # Saves images
        currentSize = len(files)
        index = 0
        for c, l in enumerate(label):  # loops over every custom image
            dupe = False  # `dupe` = duplicate image
            img = Image.fromarray(images[c] * 255).convert("L")
            # Checks for duplicates and overwrites files if needed
            for i in files:  # loops over every already saved image
                existingImg = Image.open("custom_training/" + i)
                dupe = (np.array(existingImg) == np.array(img)).all()
                if dupe: break
            if not dupe:
                img.save(f"custom_training/TRAINING_{currentSize + index}_{l}.png")
                index += 1

    # Updates self.train_img and self.train_label to contain the images/labels from custom_training
    def updateTrainingData(self):
        pass

    @staticmethod  # img_list and label_list are the original labels and images. Predictions will be the predicted values
    def displayData(self, img_list, label_list, predictions: [] = None):
        count = len(label_list)
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
                prediction = predictions[i]
                num = int(np.argmax(prediction))
                percentage = np.max(prediction) * 100
                color = 'red'
                if num == int(label_list[i]):
                    color = 'blue'
                else:
                    # print(num, label_list[i])
                    pass
                plt.xlabel(f"{self.classNames[num]} ; {np.round(percentage, 2)}").set_color(color)

        plt.show()
