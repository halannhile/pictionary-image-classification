import keras as k
import random

class Layer:
    activation = ""
    nodes = 0

    """
    Randomises layer properties (needed for first generation and mutants)
    """
    def randomise_properties(self):

        activations = ["sigmoid","softmax", "sigmoid", "tanh" , "relu"]
        index = random.randint(0, len(activations) - 1)
        self.activation = activations[index]

        nodes = [258, 512, 768, 1536, 2048, 3584 ,4096]
        index = random.randint(0, len(nodes) - 1)
        self.nodes = nodes[index]

    """
    Sets layer properties (when reproducing layers)
    """
    def set_properties(self, activation, nodes):
        self.nodes = nodes
        self.activation = activation

    """
    When model is being compiled, the Model class will call this function so the layer can add itself to a keras model
    """
    def add_to_model(self, model):
        model.add(k.layers.Dense(self.nodes, activation=self.activation))

        return model

    """
    Reproduce 2 layers, will return a new layer
    """
    def reproduce(self,partner):
        mutant = Layer()
        mutant.randomise_properties()

        parents = [self, partner, mutant]

        index = self.chose_parent()
        nodes = parents[index].nodes

        index = self.chose_parent()
        activation = parents[index].activation

        child = Layer()
        child.set_properties(activation,nodes)

        return child

    """
    Choses if child will inherit form mum, dad or mutant (1/30 chance)
    returns 0 - mum, 1 - dad or 2 - mutant
    """
    def chose_parent(self):
        mutation = random.randint(0, 35)
        if mutation == 25:
            return 2
        else:
            return random.randint(0, 1)

    def __str__(self):
        return "Dense({}, activation={})".format(self.nodes,self.activation)
