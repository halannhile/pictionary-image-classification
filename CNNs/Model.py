import keras as k
from Layer import Layer
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class Model:

    def __init__(self, start_end_nodes):
        self.start_end_nodes = start_end_nodes
        self.accuracy = None
        self.layers = []
        self.model = None

    def __str__(self):
        out = "---------------------\n"

        out += "Loss : {} \nOptimiser: {}\nEpochs: {}\nlayers : [\n".format(self.loss, self.optimiser, self.epochs)
        for layer in self.layers:
            out += str(layer) + "\n"

        out += "]\n---------------------\n"

        return out

    """
    Sets the properties of the model (needed when reproducing)
    """
    def set_properties(self, loss, optimiser, layers, layers_count, epochs):
        self.layers = layers
        self.loss = loss
        self.optimiser = optimiser
        self.layers_count = layers_count
        self.epochs = epochs

    """
    Randomises the properties of the model  (needed for first generation and mutants)
    """
    def randomise_properties(self):
        # Loss
        loss_functions = [ "categorical_crossentropy", "kullback_leibler_divergence"]#,"sparse_categorical_crossentropy"]
        index = random.randint(0, len(loss_functions) - 1)
        self.loss = loss_functions[index]

        # Optimiser
        optimisers = [k.optimizers.SGD(), k.optimizers.Adam()]
        index = random.randint(0, len(optimisers) - 1)
        self.optimiser = optimisers[index]

        # Epochs
        self.epochs = 9

        # Layers
        self.randomise_layers()

    """
    Returns a new Model object based on self and partner
    """
    def reproduce(self, partner):
        # Mutant is a model with random properties
        mutant = Model(self.start_end_nodes)
        mutant.randomise_properties()

        parents = [self, partner, mutant]

        # Loss
        random_index = self.chose_parent()
        loss = parents[random_index].loss

        # Epochs
        random_index = self.chose_parent()
        epochs = parents[random_index].epochs

        # Optimiser
        random_index = self.chose_parent()
        optimiser = parents[random_index].optimiser

        # Layers
        layers = []
        random_index = self.chose_parent()
        layers_count = parents[random_index].layers_count

        # First layer
        index = random.randint(0,1)
        layers.append(parents[index].layers[0])
        # Hidden layers
        for i in range(1, layers_count):
            child_layer = parents[0].layers[i].reproduce(parents[1].layers[i])
            layers.append(child_layer)
        # Last layer
        index = random.randint(0, 1)
        layers.append(parents[index].layers[-1])

        child = Model(self.start_end_nodes)
        child.set_properties(loss, optimiser, layers, layers_count, epochs)

        return child

    """
    Scores the models (if model already has an accuracy (was a survivor from last generation) - no need to score :)
    """
    def score(self, data):
        try:
            if self.accuracy is None:
                X_train, X_test, y_train, y_test = train_test_split(data['x'],data['y'], test_size = 0.33, random_state = 778)
                newdata = {'x': X_train, 'y': y_train}

                # compile keras model
                if(self.model is None):
                    self.train_model(newdata)

                # predict test set and calculate also generates confusion table (very messy code)
                predictions = self.model.predict(x=X_test, batch_size=200)
                predicted = self.categorise_answers(predictions)
                actual = self.test_answers(y_test)
                correct = 0
                confusion = [[0 for _ in range(6)]for _ in range(6)]
                for i in range(len(predicted)):
                    if predicted[i] == actual[i]:
                        correct += 1
                    confusion[actual[i]][predicted[i]] += 1

                for i in range(len(confusion)):
                    errors = 0
                    total = 0
                    for j in range(len(confusion)):
                        if j !=i:
                            errors += confusion[i][j]
                        total += confusion[i][j]
                    print(i, confusion[i], errors, int(errors *100/total))

                print(len(predictions))
                self.accuracy = correct/len(predictions)

            return self.accuracy

        except:
            return 0

    """
    Maps test array to class value ( [0,0,0,0,1,0] -> 4)
    """
    def test_answers(self,test):
        a = []
        for val in test:
            a.append(np.argmax(val))
        return a

    """
    Maps model predictions to class value ([0.003,0.0004,0.0001,0.0006,0.825,0.0002] -> 4)
    """
    def categorise_answers(self,predictions):
        categories = []
        # dic = {0: 'banana', 1: 'boomerang', 2:'cactus', 3:'crab', 4:'flip flops',5: 'kangaroo'}
        for pred in predictions:
            categories.append(np.argmax(pred))
        return categories

    """
    Choses if child will inherit form mum, dad or mutant (1/30 chance)
    returns 0 - mum, 1 - dad or 2 - mutant
    """
    def chose_parent(self):
        mutation = random.randint(0,30)
        if mutation == 50:
            return 2
        else:
            return random.randint(0,1)

    """
    Helper function to randomise layer construction
    """
    def randomise_layers(self):
        layer_counts = [2]#,3,4]
        index = random.randint(0, len(layer_counts) - 1)
        layer_count = layer_counts[index]
        self.layers_count = layer_count

        input_layer = Layer()
        input_layer.randomise_properties()
        self.layers.append(input_layer)

        for i in range(layer_count - 1):
            hidden_layer = Layer()
            hidden_layer.randomise_properties()
            self.layers.append(hidden_layer)

        output_layer = Layer()
        output_layer.set_properties("softmax", self.start_end_nodes[1])
        self.layers.append(output_layer)

    # creates and compiles and fits a keras model
    def train_model(self, data):
        x = data['x']
        y = data['y']

        # Build Model - I added these convolutions and pooling layers as defaults later (they wont evolve)
        self.model = k.models.Sequential()
        self.model.add(k.layers.Reshape((28, 28, 1), input_shape=(784,)))

        self.model.add(k.layers.Conv2D(32, 3, activation="relu"))
        self.model.add(k.layers.MaxPool2D(2, 2))

        self.model.add(k.layers.Conv2D(64, (2, 2), activation="relu"))
        self.model.add(k.layers.MaxPool2D(2, 2))

        self.model.add(k.layers.Flatten())

        # Here we add the layers that the model is evolving
        for i in range(self.layers_count + 1):
            self.layers[i].add_to_model(self.model)

            self.model.add(k.layers.Dropout(0.4))

        self.model.compile(loss=self.loss, optimizer=self.optimiser, metrics=["accuracy"])
        self.model.layers.remove(self.model.layers[-1])

        print(self.model.summary())





