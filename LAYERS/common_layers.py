import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D

#Batch Normalization with parameters taken from the cfg file
def batch_norm(inputs, training, data_format, momentum, epsilon):
    if data_format == 'channels_first': 
        #In channel first tensor is considered as (Number_Of_Channels, Height , Width)
        axis = 1
    else:
        axis = 3
    return tf.layers.batch_normalization(
        inputs = inputs, axis = axis, momentum = momentum, epsilon = epsilon,
        scale = True, training = training)

#Convolution with 2-D stride and padding
def conv2d_with_padding(inputs, filters, kernel_size, strides=1, data_format):
    if strides == 1:
        padding = 'same'
    else:
        inputs = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding, use_bias= False, data_format=data_format, kernel_regularizer=l2(0.0005))

#Residual block for Darknet 
def darknet_residual(inputs, filters, training, data_format, strides=1, leaky_relu):
    shortcut = inputs

    inputs = conv2d_with_padding(inputs, filters, kernel_size=1, strides, data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)

    inputs = conv2d_with_padding(inputs, 2*filters, kernel_size=3, strides, data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)

    output += shortcut

    return output

#Layers to be used after Darknet53 -> yolo_convolution_block and yolo_layer

#Five convolutional blocks for the route, then another one for the output
def yolo_convolution_block(inputs, filters, training, data_format,leaky_relu):
    #1
    inputs = conv2d_with_padding(inputs, filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)
    #2
    inputs = conv2d_with_padding(inputs, 2*filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)
    #3
    inputs = conv2d_with_padding(inputs, filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)
    #4
    inputs = conv2d_with_padding(inputs, 2*filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)
    #5
    inputs = conv2d_with_padding(inputs, filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)
    #Route
    route = inputs
    #Output (MANTENERE QUA O TIRARE FUORI?)
    inputs = conv2d_with_padding(inputs, 2*filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    output = tf.nn.leaky_relu(inputs, alpha= leaky_relu)

    return route, output

#Final detection layer
def yolo_layer(inputs, num_classes, anchors, img_size, data_format):
    n_anchors = len(anchors)

    inputs = tf.layers.conv2d(inputs, filters=n_anchors * (5 + n_classes),
                              kernel_size=1, strides=1, use_bias=True,
                              data_format=data_format)
    

