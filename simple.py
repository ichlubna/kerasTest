import sys
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, model_from_json

np.random.seed(5)

dataset = np.loadtxt('simple.csv', delimiter=',')
inputsCount = 8
inputs = dataset[:,0:inputsCount]
outputs = dataset[:,inputsCount]

model = None
if len(sys.argv) == 1:
    #create model and train
    model = Sequential()
    #fully connected layers - Dense
    model.add(Dense(12, input_shape=(inputsCount,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #batch - how many samples (rows, feature vectors) are processed before weights update
    #epoch - how many times iterate over dataset
    model.fit(inputs, outputs, epochs=150, batch_size=10)
else:
    #load existing model
    file = open(sys.argv[1]+".json", 'r')
    json = file.read()
    file.close()
    model = model_from_json(json)
    model.load_weights(sys.argv[1]+".h5")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

accuracy = model.evaluate(inputs, outputs)
print('Training set accuracy')
print(accuracy)

testSet = np.loadtxt('simpleTest.csv', delimiter=',')
inputs = testSet[:,0:inputsCount]
outputs = testSet[:,inputsCount]
accuracy = model.evaluate(inputs, outputs)
print('Test set accuracy')
print(accuracy)

singleTestSet = np.loadtxt('simpleTestPredictions.csv', delimiter=',')
inputs = singleTestSet[:,0:inputsCount]
outputs = singleTestSet[:,inputsCount]
predictions = model.predict(inputs)
print("Predictions")
for i in range(len(predictions)):
    print(inputs[i])
    print(outputs[i])
    print(predictions[i])
    print("")

#store model if trained
if len(sys.argv) == 1:
    modelJson = model.to_json()
    with open("model.json", "w") as file:
        file.write(modelJson)
    model.save_weights("model.h5")
    print("Model stored to model.json and weights to model.h5")
