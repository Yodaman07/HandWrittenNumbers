import numpy as np
from model import Model
import tensorflow as tf
import keras.src


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
        model.compile(optimizer="Adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])  # compiling everything, now ready to train the model

        # i = np.concatenate(self.train_img, self.updateTrainingData()[0])
        # l = np.concatenate(self.train_label, self.updateTrainingData()[1])

        model.fit(self.train_img, self.train_label)  # training the model
        self.model = model

    def predict(self, CustomImages: np.ndarray = None, UpdateTrainingData=False):  # Predicts and graphs test images
        #  If CUSTOM_IMAGE is left blank, program will use the default testing images
        prediction_model = keras.Sequential([self.model, keras.layers.Softmax()])
        if CustomImages is not None:
            CustomLabels = np.array([1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 3, 5])  # num0 to num13
            predictions = prediction_model.predict(CustomImages)
            if UpdateTrainingData:
                self.saveCustomImages(CustomImages,CustomLabels)
            self.displayData(self, CustomImages, CustomLabels, predictions)
        else:
            predictions = prediction_model.predict(self.test_img)
            self.displayData(self, self.test_img[0:49], self.test_label[0:49], predictions)


m = HandWrittenNumbersModel()
input("Ready? ")
m.trainModel()
input("Trained! Ready for the predictions? ")
m.predict(m.evalCustomImages(14), True)
