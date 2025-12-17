import tensorflow as tf
from tensorflow.keras import layers, models
from scikeras.wrappers import KerasRegressor, KerasClassifier


def build_tf_regressor(nb_features, layers_count=2, width=64, activation='relu', dropout_rate=0.0, learning_rate=1e-3, loss='huber'):

    model = models.Sequential()

    # Couche d'entrée
    model.add(layers.Input(shape=(nb_features,)))

    # Couches cachées
    for i in range(layers_count):
        model.add(layers.Dense(width, activation=activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Couche de sortie
    model.add(layers.Dense(2))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def build_tf_classifier(nb_features, layers_count=2, width=64, activation='relu', dropout_rate=0.0, learning_rate=1e-3, loss='binary_crossentropy'):

    model = models.Sequential()

    # Couche d'entrée
    model.add(layers.Input(shape=(nb_features,)))

    # Couches cachées
    for i in range(layers_count):
        model.add(layers.Dense(width, activation=activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Couche de sortie
    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    return model