from Generation import Generation
from Model import  Model
from Layer import Layer
import pandas as pd
from keras.utils import np_utils
import keras as k
from sklearn.model_selection import train_test_split
import numpy as np
import datetime

def test_answers(test):
    a = []
    for val in test:
        a.append(np.argmax(val))
    return a


def categorise_answers(predictions):
    categories = []
    # dic = {0: 'banana', 1: 'boomerang', 2:'cactus', 3:'crab', 4:'flip flops',5: 'kangaroo'}
    for pred in predictions:
        categories.append(np.argmax(pred))
    return categories


def run_generations():
    # no in generation, epoch count and node counts and no layers
    g = Generation(20,0.2,1)
    g.generate_population()

    for i in range(10):
        print("doing generation {}".format(i +1))

        y = pd.read_csv("sketches.csv", usecols=[785])
        x = pd.read_csv("sketches.csv", usecols=[i for i in range(1, 785)])
        x = x / 255
        y = pd.Categorical(y["word"]).codes
        y = np_utils.to_categorical(y, num_classes=6)
        data = {'x': x, "y": y}

        g = g.evolve(data)


def publishresults(predictions):
    results = []
    dic = {0: 'banana', 1: 'boomerang', 2:'cactus', 3:'crab', 4:'flip flops',5: 'kangaroo'}
    for pred in predictions:
        results.append(dic[pred])
    output = {'Id':[i+1 for i in range(len(predictions))], 'Category': results}

    df = pd.DataFrame(output, columns=['Id','Category'])
    df.to_csv("output.csv", index=False)
#run_generations()

def test_model():
    layers = []
    l1 = Layer()
    l1.set_properties("relu",2500)
    layers.append(l1)
    l2 = Layer()
    l2.set_properties("sigmoid", 1536)
    layers.append(l2)
    l6 = Layer()
    l6.set_properties("relu", 700)
    layers.append(l6)
    l7 = Layer()
    l7.set_properties("tanh", 200)
    layers.append(l7)
    l3 = Layer()
    l3.set_properties("softmax", 6)
    layers.append(l3)
    m = Model([784,6])

    m.set_properties("kullback_leibler_divergence",k.optimizers.Adam(),layers,4,15)

    print(str(m))
    y = pd.read_csv("sketches.csv", usecols=[785])
    x = pd.read_csv("sketches.csv", usecols=[i for i in range(1, 785)])
    x = x / 255
    y = pd.Categorical(y["word"]).codes


    y = np_utils.to_categorical(y, num_classes=6)
    data = {'x': x, "y": y}

    print(m.score(data))

def test2():
    m = k.models.Sequential()
    m.add(k.layers.Reshape((28, 28, 1), input_shape=(784,)))

    m.add( k.layers.Conv2D(32, 3, activation="relu"))
    m.add(k.layers.MaxPooling2D( 2, 2))
    m.add(k.layers.Dropout(0.25))
    #m.add(k.layers.Conv2D(64, (2,2), activation="tanh"))
    #m.add(k.layers.AveragePooling2D( 2, 2))

    m.add(k.layers.Conv2D(64, (2, 2), activation="relu"))
    m.add(k.layers.MaxPooling2D(2, 2))
    m.add(k.layers.Dropout(0.25))

    m.add(k.layers.Flatten())

    m.add(k.layers.Dense(512,activation="softmax"))
    m.add(k.layers.Dropout(rate=0.3))

    m.add(k.layers.Dense(1536, activation="relu"))
    m.add(k.layers.Dropout(rate=0.3))

    m.add(k.layers.Dense(512, activation="sigmoid"))
    m.add(k.layers.Dropout(rate=0.3))

    # Output layer, class prediction
    m.add(k.layers.Dense(6, activation="softmax"))

    m.compile(k.optimizers.Adam(),loss="kullback_leibler_divergence",metrics=["accuracy"])

    y = pd.read_csv("sketches.csv", usecols=[784])
    x = pd.read_csv("sketches.csv", usecols=[i for i in range(0, 784)])
    x = x / 255
    y = pd.Categorical(y["word"]).codes

    y = np_utils.to_categorical(y, num_classes=6)


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=78)

    h = m.fit(x=X_train, y=y_train, epochs=5, batch_size=70)
    x_t = pd.read_csv("sketches_test.csv", usecols=[i for i in range(1, 785)])
    x_t = x_t /255
    predictions = m.predict(x=X_test, batch_size=200)

    m2 = k.models.Sequential()
    m2.add(k.layers.Dense(32, activation="relu", input_shape=(6,)))

    m2.add(k.layers.Dense(6, activation="softmax"))
    m2.compile(k.optimizers.SGD(), loss="categorical_crossentropy", metrics=["accuracy"])
    #predictions = m2.fit(x=predictions, y=y_test, validation_split=0.3, epochs=15,batch_size=50)

    predicted = categorise_answers(predictions)
    actual = test_answers(y_test)
    correct = 0
    confusion = [[0 for _ in range(6)] for _ in range(6)]

    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            correct += 1


        confusion[actual[i]][predicted[i]] += 1

    for i in range(len(confusion)):
        errors = 0
        total = 0
        for j in range(len(confusion)):
            if j != i:
                errors += confusion[i][j]
            total += confusion[i][j]
        print(i, confusion[i], errors, int(errors * 100 / total))

    print(correct/ len(predictions))
    m.save("{}_{}.h5".format(correct/ len(predictions), datetime.date.today()))

    publishresults(predicted)

def test3():
    m = k.models.Sequential()
    m.add(k.layers.Reshape((28, 28, 1), input_shape=(784,)))

    m.add(k.layers.Conv2D(32, 3, activation="relu"))
    m.add(k.layers.MaxPooling2D(2, 2))
    m.add(k.layers.Dropout(0.25))
    # m.add(k.layers.Conv2D(64, (2,2), activation="tanh"))
    # m.add(k.layers.AveragePooling2D( 2, 2))

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

    y = pd.read_csv("sketches.csv", usecols=[784])
    x = pd.read_csv("sketches.csv", usecols=[i for i in range(0, 784)])
    x = x / 255
    y = pd.Categorical(y["word"]).codes

    y = np_utils.to_categorical(y, num_classes=6)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=78)

    h = m.fit(x=X_train, y=y_train, epochs=30, batch_size=70)
    x_t = pd.read_csv("sketches_test.csv", usecols=[i for i in range(1, 785)])
    x_t = x_t / 255
    predictions = m.predict(x=X_test, batch_size=200)

    m2 = k.models.Sequential()
    m2.add(k.layers.Dense(32, activation="relu", input_shape=(6,)))

    m2.add(k.layers.Dense(6, activation="softmax"))
    m2.compile(k.optimizers.SGD(), loss="categorical_crossentropy", metrics=["accuracy"])
    # predictions = m2.fit(x=predictions, y=y_test, validation_split=0.3, epochs=15,batch_size=50)

    predicted = categorise_answers(predictions)
    actual = test_answers(y_test)
    correct = 0
    confusion = [[0 for _ in range(6)] for _ in range(6)]

    difficulties = []
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            correct += 1
        elif predicted[i] == 0 and actual[i] == 1:
            difficulties.append(X_test.index[i])

        confusion[actual[i]][predicted[i]] += 1



    for i in range(len(confusion)):
        errors = 0
        total = 0
        for j in range(len(confusion)):
            if j != i:
                errors += confusion[i][j]
            total += confusion[i][j]
        print(i, confusion[i], errors, int(errors * 100 / total))

    print(correct / len(predictions))
    #m.save("{}_{}.h5".format(correct / len(predictions), datetime.date.today()))


    for difficulty in difficulties:
        X_train = X_train.append([x.iloc[difficulty, :]]*2, ignore_index=True)

        y_train = np.vstack([y_train, y[difficulty, :]])
        y_train = np.vstack([y_train, y[difficulty, :]])


    m = k.models.Sequential()
    m.add(k.layers.Reshape((28, 28, 1), input_shape=(784,)))

    m.add(k.layers.Conv2D(32, 3, activation="relu"))
    m.add(k.layers.MaxPooling2D(2, 2))
    m.add(k.layers.Dropout(0.25))
    # m.add(k.layers.Conv2D(64, (2,2), activation="tanh"))
    # m.add(k.layers.AveragePooling2D( 2, 2))

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
    h = m.fit(x=X_train, y=y_train, epochs=30, batch_size=70)

    predictions = m.predict(x=X_test, batch_size=200)


    predicted = categorise_answers(predictions)
    actual = test_answers(y_test)
    correct = 0
    confusion = [[0 for _ in range(6)] for _ in range(6)]

    difficulties = []
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            correct += 1
        elif predicted[i] == 0 and actual[i] == 1:
            difficulties.append(X_test.index[i])

        confusion[actual[i]][predicted[i]] += 1

    for i in range(len(confusion)):
        errors = 0
        total = 0
        for j in range(len(confusion)):
            if j != i:
                errors += confusion[i][j]
            total += confusion[i][j]
        print(i, confusion[i], errors, int(errors * 100 / total))

    print(correct / len(predictions))

test3()