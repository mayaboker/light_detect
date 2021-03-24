import tensorflow as tf


def ConvBn(*args, **kwargs):
    pad = kwargs['kernel_size'] // 2
    return tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(pad),
        tf.keras.layers.Conv2D(*args, **kwargs),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),
    ])


def DwConvBn(*args, **kwargs):
    pad = kwargs['kernel_size'] // 2
    return tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(pad),
        tf.keras.layers.DepthwiseConv2D(*args, **kwargs),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),
    ])


if __name__ == "__main__":
    input_shape = (2, 32, 32, 3)
    x = tf.random.normal(input_shape)

    
    conv = ConvBn(filters=64, kernel_size=3, strides=2)
    y = conv(x)
    print(y.shape)

