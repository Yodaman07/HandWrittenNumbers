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

        # print(type(self.train_img))
        # print(type(self.train_img[0]))
        # print()
        # print(type(self.updateTrainingData()[0][0]))
        imgList = self.updateTrainingData()[0]
        if len(imgList) > 0:
            i = np.concatenate((self.train_img, self.updateTrainingData()[0]), axis=0)
        else:
            i = self.train_img
        l = np.concatenate((self.train_label, self.updateTrainingData()[1]), axis=0)

        model.fit(i, l)  # training the model
        self.model = model

    def predict(self, CustomImages: np.ndarray = None, CustomLabels: np.ndarray = None, UpdateTrainingData=False):  # Predicts and graphs test images
        #  If CUSTOM_IMAGE is left blank, program will use the default testing images
        prediction_model = keras.Sequential([self.model, keras.layers.Softmax()])
        if CustomImages is not None:

            predictions = prediction_model.predict(CustomImages)
            if UpdateTrainingData:
                for i in range(len(CustomLabels)):
                    prediction = predictions[i]
                    if CustomLabels[i] == np.argmax(prediction):
                        self.saveCustomImages(np.array([CustomImages[i]]), np.array([CustomLabels[i]]))

                self.displayData(self, CustomImages, CustomLabels, predictions, update=True)
            else:
                self.displayData(self, CustomImages, CustomLabels, predictions)
        else:
            predictions = prediction_model.predict(self.test_img)
            self.displayData(self, self.test_img[0:49], self.test_label[0:49], predictions)


m = HandWrittenNumbersModel()
input("Ready? ")
m.trainModel()
input("Trained! Ready for the predictions? ")

label1 = np.array([1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7])  # num0 to num11
label2 = np.array([3,5])

m.predict(m.evalCustomImages(12), label1, False)
