from LAYERS.common_layers import  *
from tensorflow.keras import Model
from tensorflow.keras.layers import Input


#Darknet53 -> feature extraction
def darknet53(inputs, activation, name=None, training=False, data_format=None):
    x = inputs
    #count = 0
    #x = inputs = Input([None, None, 3])
    inputs = conv2d_with_padding(inputs, filters=32, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #inputs = tf.nn.leaky_relu(inputs, alpha=leaky_relu)
    inputs = LeakyReLU(alpha=activation)(inputs)

    inputs = conv2d_with_padding(inputs, filters=64, kernel_size=3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)
    inputs = LeakyReLU(alpha=activation)(inputs)

    inputs = darknet_residual(inputs, filters=32, training=training, data_format=data_format, activation=activation)

    inputs = conv2d_with_padding(inputs, filters=128, kernel_size=3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)
    inputs = LeakyReLU(alpha=activation)(inputs)

    for i in range(2):
        inputs = darknet_residual(inputs, filters=64, training=training, data_format=data_format, activation=activation)

    inputs = conv2d_with_padding(inputs, filters=256, kernel_size=3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)
    inputs = LeakyReLU(alpha=activation)(inputs)

    for i in range(8):
        inputs = darknet_residual(inputs, filters=128, training=training, data_format=data_format, activation=activation)

    route1 = inputs

    inputs = conv2d_with_padding(inputs, filters=512, kernel_size=3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)
    inputs = LeakyReLU(alpha=activation)(inputs)

    for i in range(8):
        inputs = darknet_residual(inputs, filters=256, training=training, data_format=data_format, activation=activation)

    route2 = inputs

    inputs = conv2d_with_padding(inputs, filters=1024, kernel_size=3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)
    inputs = LeakyReLU(alpha=activation)(inputs)

    for i in range(4):
        inputs = darknet_residual(inputs, filters=512, training=training, data_format=data_format, activation=activation)

    aux = tf.keras.Model(x, (route1,route2,inputs), name=name)
    aux.summary()
    #return tf.keras.Model(x, (route1,route2,outputs), name=name)
    return route1, route2, outputs #RESTITUIRE QUESTO O UN MODELLO?

