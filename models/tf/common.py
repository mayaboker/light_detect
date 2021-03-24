import tensorflow as tf


def ConvBn(*args, **kwargs):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(*args, **kwargs),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ])


def DwConvBn(*args, **kwargs):
    return tf.keras.Sequential([
        tf.keras.layers.DepthwiseConv2D(*args, **kwargs),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ])

