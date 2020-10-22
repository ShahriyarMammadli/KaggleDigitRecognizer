# Shahriyar Mammadli
# Import required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

# Building CNN model
def buildCNNModel(X_train, X_test, y_train, y_test, trainSize, testSize, predDf):
    # Normalize the data
    X_train = X_train/255.0
    X_test = X_test / 255.0
    predDf = predDf / 255.0
    # Shaping the input size accordingly
    X_train = X_train.values.reshape(trainSize, 28,28, 1)
    y_train = to_categorical(y_train.values, num_classes=10)
    X_test = X_test.values.reshape(testSize, 28, 28, 1)
    y_test = to_categorical(y_test.values, num_classes=10)
    # Building a model
    model = models.Sequential()
    # Stage 1
    model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Dropout(0.25))
    # Stage 2
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Dropout(0.25))
    # Stage 3
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256))
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(10, activation='softmax'))
    # Show the model details
    model.summary()
    # Using adamax optimizer, run the model
    model.compile(optimizer='adamax',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # Putting early stopping to avoid overtraining
    earlyStopping = EarlyStopping(monitor="val_accuracy",
                                  mode='auto', patience=30,
                                  restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_test, y_test), callbacks=[earlyStopping])
    # Plot train loss vs validation loss to observe fitting details
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
    print(f"Highest Validation Accuracy: {round(100 * max(history.history['val_accuracy']), 2)}")
    predictions = model.predict(predDf.values.reshape(predDf.shape[0], 28,28, 1))
    CNNPredictions = list(map(lambda i: np.argmax(i), predictions))
    return pd.DataFrame({'ImageId': range(1, predDf.shape[0] + 1), 'Label': CNNPredictions})