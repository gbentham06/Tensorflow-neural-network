from tensorflow import keras
import numpy as np
import os

MODEL_SAVE = 'mnist_model.keras'

# Unpacking Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# Preprocessing data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train_hot = keras.utils.to_categorical(y_train, 10)
y_test_hot = keras.utils.to_categorical(y_test, 10)

if os.path.exists(MODEL_SAVE):
    model = keras.models.load_model(MODEL_SAVE)
else:
    # build neural net
    model = keras.Sequential([
        # flatten our input layer
        keras.layers.Flatten(input_shape=(28,28), name='flat_input'),
        # hidden layers
        keras.layers.Dense(128, activation='relu', name='hidden_layer_1'),
        keras.layers.Dense(64, activation='relu', name='hidden_layer_2'),
        #output layer
        keras.layers.Dense(10, activation='softmax', name='output_layer')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train_hot,
        epochs=10,
        batch_size=32,
        validation_split=0.1
    )

    model.save(MODEL_SAVE)

def testscript():
    model.evaluate(x_test, y_test_hot, verbose=2)
    predictions = model.predict(x_test[:5])
    predictions = [np.argmax(i) for i in predictions]

    for i in zip(predictions, y_test[:5]):
        print(f"predicted: {i[0]}, true: {i[1]}")

if __name__ == '__main__':
    testscript()