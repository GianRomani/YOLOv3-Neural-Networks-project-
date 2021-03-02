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
o
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
    #Last Convolution
    output = tf.layers.conv2d(inputs, filters=n_anchors * (5 + n_classes),
                              kernel_size=1, strides=1, use_bias=True,
                              data_format=data_format)
    
    shape_output = output.get_shape().as_list()

    if data_format == 'channels_first':
        grid_shape = shape_output[2:4]
        output = tf.transpose(output, [0,2,3,1])
    else:
        grid_shape = shape_output[1:3]

    output_reshaped = tf.reshape(output, [-1, n_anchors*grid_shape[0]*grid_shape[1], 5+n_classes])
    #For example from (1,13,13,255) to (1,3*13*13,85)

    #Now we can get the values of the boxes
    box_center, box_shape, confidence, classes = tf.split(output_reshaped, [2,2,1,n_classes], axis=-1)
    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])



    
