from Generation import Generation
from Model import  Model
from Layer import Layer
import sklearn
import pandas as pd
from keras.utils import np_utils
import keras as k
from sklearn.model_selection import train_test_split
import numpy as np

"""
Builds keras model and fits data and returns model
"""
def build_model(train_x, train_y):
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
    m.fit(x=train_x, y=train_y, epochs=40, batch_size=200)

    return m
"""
loads saved models
"""
def load_all():
    models = []
    for i in range(9):
        model = k.models.load_model("bagged\model_{}".format(i))
        models.append(model)

    return models


"""
Builds n models, bosstraps a samples of 7000 observation to train model
"""
def build_all(count,x, y):
    models = []

    for i in range(count):
        indexes = np.random.choice(len(x), 7000)
        train_x = x.iloc[indexes, :]
        train_y = y[indexes, :]
        print("building model {}".format(i+1))
        models.append(build_model(train_x, train_y))

    return models

"""
Predicts values for all models, then calculates the most chosen ouput
"""
def predict_all(models, test_x):
    yhats = [model.predict(test_x) for model in models]
    yhats = np.array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = np.argmax(summed, axis=1)

    return result

"""
Saves results to csv file (for predictions competition)
"""
def publishresults(predictions):
    results = []
    dic = {0: 'banana', 1: 'boomerang', 2:'cactus', 3:'crab', 4:'flip flops',5: 'kangaroo'}
    for pred in predictions:
        results.append(dic[pred])
    output = {'Id':[i+1 for i in range(len(predictions))], 'Category': results}

    df = pd.DataFrame(output, columns=['Id','Category'])
    df.to_csv("prediction_bagged.csv", index=False)


"""
calculates accuracy that the combination of model 1 -> index has ( allows us to see how accuracy improves as we average out more and more models)
"""
def score_all(models, test_x, test_y, index, publish=False):
    # select models 1 -> index
    step = []
    for i in range(index):
        step.append(models[i])

    predicted = predict_all(step, test_x)
    if(publish):
        publishresults(predicted,)
    else:
        actual = np.argmax(test_y, axis=1)
        return sklearn.metrics.accuracy_score(actual, predicted)


"""
Train of 66% and test on 33%
"""
def run():
    # get data
    y = pd.read_csv("sketches.csv", usecols=[784])
    x = pd.read_csv("sketches.csv", usecols=[i for i in range(0, 784)])
    x = x / 255
    y = pd.Categorical(y["word"]).codes
    y = np_utils.to_categorical(y, num_classes=6)

    # split
    train_x, test_x, train_y, test_y= train_test_split(x, y, test_size=0.25, random_state=78)

    # build 9 models
    models = build_all(9, train_x, train_y)

    # calculate score of combined model when adding extra models 1 by 1
    for i in range(1, len(models)):
        score = score_all(models, test_x, test_y, i)
        print("{} -> {}".format(i+1, score))

    # save the models to file
    for i in range(len(models)):
        models[i].save("bagged\model_{}".format(i))



"""
Train on 100% of data and then output predictions for competition
"""
def run_to_test():
    y = pd.read_csv("sketches.csv", usecols=[784])
    x = pd.read_csv("sketches.csv", usecols=[i for i in range(0, 784)])
    x = x / 255
    y = pd.Categorical(y["word"]).codes
    y = np_utils.to_categorical(y, num_classes=6)

    models = build_all(9, x, y)

    x_t = pd.read_csv("sketches_test.csv", usecols=[i for i in range(1, 785)])
    x_t = x_t / 255
    score_all(models, x_t, [], len(models), True)
    for i in range(len(models)):
        models[i].save("bagged\model_{}".format(i))


run_to_test()
