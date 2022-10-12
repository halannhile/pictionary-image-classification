from DataGenerator import  BaggingDataGenerator
from Generation import Generation
from Model import  Model
from Layer import Layer
import pandas as pd
from keras.utils import np_utils
import keras as k
from sklearn.model_selection import train_test_split
import numpy as np
import datetime


class Bagger:
    def __init__(self, model_count, boostrap_sample_size = None):
        self.data = BaggingDataGenerator(0.33, boostrap_sample_size)
        self.model_count = model_count
        self.models = []
    """
    Builds and compiles keras model
    """
    def build_model(self, train_x, train_y):
        m = k.models.Sequential()
        m.add(k.layers.Reshape((28, 28, 1), input_shape=(784,)))

        m.add(k.layers.Conv2D(32, 3, activation="relu"))
        m.add(k.layers.MaxPooling2D(2, 2))
        m.add(k.layers.Dropout(0.25))

        m.add(k.layers.Conv2D(64, (2, 2), activation="relu"))
        m.add(k.layers.MaxPooling2D(2, 2))
        m.add(k.layers.Dropout(0.25))

        m.add(k.layers.Flatten())

        m.add(k.layers.Dense(512, activation="softmax"))
        m.add(k.layers.Dropout(rate=0.3))

        m.add(k.layers.Dense(1536, activation="relu"))
        m.add(k.layers.Dropout(rate=0.3))

        m.add(k.layers.Dense(512, activation="sigmoid"))
        m.add(k.layers.Dropout(rate=0.3))

        # Output layer, class prediction
        m.add(k.layers.Dense(6, activation="softmax"))

        m.compile(k.optimizers.Adam(), loss="kullback_leibler_divergence", metrics=["accuracy"])

        m.fit(x=train_x, y=train_y, epochs=30, batch_size=70, verbose=0)

        return m

    """
    Scores the model and returns a confusion table and accuracy
    """
    def score_model(self, m):
        (test_x, test_y) = self.data.get_test()

        predictions = m.predict(x=test_x, batch_size=200)
        predicted = self.categorise_answers(predictions)
        actual = self.test_answers(test_y)

        correct = 0
        confusion = [[0 for _ in range(6)] for _ in range(6)]

        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                correct += 1
            confusion[actual[i]][predicted[i]] += 1

        accuracy = correct / len(predictions)

        return (accuracy, confusion, predicted, actual)

    """
    Tests model on second test set
    """
    def validate_model(self,m):
        val_x, val_y = self.data.get_validation()
        predictions = m.predict(x=val_x, batch_size=200)
        predicted = self.categorise_answers(predictions)
        actual = self.test_answers(val_y)

        correct = 0
        confusion = [[0 for _ in range(6)] for _ in range(6)]

        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                correct += 1
            confusion[actual[i]][predicted[i]] += 1

        accuracy = correct / len(predictions)
        print("for val data: ")
        self.print_results(accuracy, confusion)

        return accuracy

    """
    Finds where the model makes the most mistakes and flags the indexes of those observation as important (the data object will add them to our training set)
    """
    def adjust_data(self, confusion, predictions, actual):
        (test_x, test_y) = self.data.get_test()

        max = 0
        index_max = [0,0]

        for i in range(len(confusion)):
            for j in range(len(confusion)):
                if (i != j) and (confusion[i][j] > max):
                    max = confusion[i][j]
                    index_max = [i,j]

        print("Worse: predicting {} when is actually {}".format(index_max[1], index_max[0]))
        error_indexes = []

        for i in range(len(predictions)):
            if predictions[i] == index_max[1] and actual[i] == index_max[0]:
                error_indexes.append(test_x.index[i])

        print("Found {} such predictions".format(len(error_indexes)))

        self.data.promote_indexes(error_indexes)

    """
    Runs the boosting
    """
    def run(self):
        for i in range(self.model_count):
            train_x, train_y = self.data.get_train()

            print("Doing model {} of {} ({} obvs)...".format(i + 1, self.model_count, len(train_x)))

            model = self.build_model(train_x, train_y)

            accuracy, confusion, predictions, actual = self.score_model(model)

            #self.print_results(accuracy, confusion)
            accuracy = self.validate_model(model)
            self.adjust_data(confusion, predictions, actual)
            self.models.append([accuracy, model])
            print("--------------------------------------\n\n\n")

    """
    Saves the best model to file
    """
    def save_best(self,folder_name, specifier):
        sorted_models = sorted(self.models, key=lambda x: x[0], reverse=True)
        sorted_models[0][1].save("{}\model-{}-{}.h5".format(folder_name,sorted_models[-1][0], specifier))

# ----------------- PRIVATES -----------------------
    def test_answers(self, test):
        a = []
        for val in test:
            a.append(np.argmax(val))
        return a

    def print_results(self, accuracy, c):
        print("Accuracy: {}".format(accuracy))

        for i in range(len(c)):
            print(i, c[i])

    def categorise_answers(self, predictions):
        categories = []
        # dic = {0: 'banana', 1: 'boomerang', 2:'cactus', 3:'crab', 4:'flip flops',5: 'kangaroo'}
        for pred in predictions:
            categories.append(np.argmax(pred))
        return categories


def run_only_boosted():
    b = Bagger(5)
    b.run()