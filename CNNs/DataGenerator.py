from Generation import Generation
from Model import  Model
from Layer import Layer
import pandas as pd
from keras.utils import np_utils
import keras as k
from sklearn.model_selection import train_test_split
import numpy as np
import datetime

class BoostrapDataGenerator:

    def __init__(self, test_split, boot_sample_size):
        y = pd.read_csv("sketches.csv", usecols=[784])
        x = pd.read_csv("sketches.csv", usecols=[i for i in range(0, 784)])
        x = x / 255
        y = pd.Categorical(y["word"]).codes

        y = np_utils.to_categorical(y, num_classes=6)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_split, random_state=78)
        self.boot_indexes = [i for i in range(len(self.x_train.index))]
        self.boot_sample_size = boot_sample_size
        self.promoted_indexes = []

    def get_train(self):
            # generate bootstrap sample
            random_choices = np.random.choice(len(self.boot_indexes), self.boot_sample_size - len(self.promoted_indexes))
            sample_indexes = [self.boot_indexes[i] for i in random_choices]

            train_x = self.x_train.iloc[sample_indexes, :]
            train_y = self.y_train[sample_indexes, :]

            # Append promoted indexes
            train_x = train_x.append([self.x_train.iloc[self.promoted_indexes, :]] * 2, ignore_index=True)
            train_y = np.vstack([train_y, self.y_train[self.promoted_indexes, :]])
            train_y = np.vstack([train_y, self.y_train[self.promoted_indexes, :]])

            return (train_x, train_y)

    def get_test(self):
        return (self.x_test, self.y_test)


    def promote_indexes(self, indexes):
        for index in indexes:
            for i in range(len(self.x_train.index)):
                if self.x_train.index[i] == index:
                    self.promoted_indexes.append(i)


class BaggingDataGenerator:

    def __init__(self, test_split, boostrap_sample_size=None):
        y = pd.read_csv("sketches.csv", usecols=[784])
        x = pd.read_csv("sketches.csv", usecols=[i for i in range(0, 784)])
        x = x / 255
        y = pd.Categorical(y["word"]).codes

        y = np_utils.to_categorical(y, num_classes=6)
        self.x = x
        self.y = y

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_split,random_state=78)
        self.x_test, self.x_validation, self.y_test, self.y_validation = train_test_split(self.x_test, self.y_test, test_size=0.2, random_state=66)

        if boostrap_sample_size is not None:
            self.boostrap_train_set(boostrap_sample_size)

    def boostrap_train_set(self, sample_size):
        indexes = np.random.choice(len(self.x_train), sample_size)
        self.x_train = self.x_train.iloc[indexes, :]
        self.y_train = self.y_train[indexes, :]

    def get_validation(self):
        return self.x_validation, self.y_validation

    def get_train(self):
        return (self.x_train, self.y_train)

    def get_test(self):
        return (self.x_test, self.y_test)

    def promote_indexes(self, indexes):
        for index in indexes:
            self.x_train = self.x_train.append([self.x.iloc[index, :]] * 2, ignore_index=True)
            self.y_train = np.vstack([self.y_train, self.y[index, :]])
            self.y_train = np.vstack([self.y_train, self.y[index, :]])

