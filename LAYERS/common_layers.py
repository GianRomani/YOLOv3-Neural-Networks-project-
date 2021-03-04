import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, LeakyReLU
from tensorflow.keras.regularizers import l2

#Batch Normalization with parameters taken from the cfg file
def batch_norm(inputs, training, data_format, momentum=0.9, epsilon=1e-05):
    if data_format == 'channels_first': 
        #In channel first tensor is considered as (Number_Of_Channels, Height , Width)
        axis = 1
    else:
        axis = 3
    return tf.keras.layers.BatchNormalization(
        axis = axis, momentum = momentum, epsilon = epsilon,
        scale = True, trainable = training)(inputs)

#Convolution with 2-D stride and padding
def conv2d_with_padding(inputs, filters, kernel_size, data_format, strides=1):
    if strides == 1:
        padding = 'same'
    else:
        inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)  # top left half-padding
        padding = 'valid'
    temp = strides

    output = Conv2D(filters=filters, kernel_size=kernel_size,
                strides=temp, padding=padding, use_bias= False, 
                data_format=data_format, kernel_regularizer=l2(0.0005))(inputs)

    return output

#Residual block for Darknet 
def darknet_residual(inputs, filters, training, data_format, activation, strides=1):
    shortcut = inputs

    inputs = conv2d_with_padding(inputs, filters, 1, data_format, strides)
    inputs = batch_norm(inputs, training, data_format)
    #inputs = tf.nn.leaky_relu(inputs, alpha= activation)
    inputs = LeakyReLU(alpha=activation)(inputs)

    inputs = conv2d_with_padding(inputs, 2*filters, 3, data_format, strides)
    inputs = batch_norm(inputs, training, data_format)
    #output = tf.nn.leaky_relu(inputs, alpha= activation)
    output = LeakyReLU(alpha=activation)(inputs)

    output += shortcut

    return output

#Layers to be used after Darknet53 -> yolo_convolution_block and yolo_layer

#Five convolutional blocks for the route, then another one for the output
def yolo_convolution_block(inputs, filters, training, data_format,activation):
    #1
    inputs = conv2d_with_padding(inputs, filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #inputs = tf.nn.leaky_relu(inputs, alpha= activation)
    inputs = LeakyReLU(alpha=activation)(inputs)
    #2
    inputs = conv2d_with_padding(inputs, 2*filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #inputs = tf.nn.leaky_relu(inputs, alpha= activation)
    inputs = LeakyReLU(alpha=activation)(inputs)
    #3
    inputs = conv2d_with_padding(inputs, filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #inputs = tf.nn.leaky_relu(inputs, alpha= activation)
    inputs = LeakyReLU(alpha=activation)(inputs)
    #4
    inputs = conv2d_with_padding(inputs, 2*filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #inputs = tf.nn.leaky_relu(inputs, alpha= activation)
    inputs = LeakyReLU(alpha=activation)(inputs)
    #5
    inputs = conv2d_with_padding(inputs, filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #inputs = tf.nn.leaky_relu(inputs, alpha= activation)
    inputs = LeakyReLU(alpha=activation)(inputs)
    #Route
    route = inputs
    #Output (MANTENERE QUA O TIRARE FUORI?)
    inputs = conv2d_with_padding(inputs, 2*filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #output = tf.nn.leaky_relu(inputs, alpha= activation)
    output = LeakyReLU(alpha=activation)(inputs)

    return route, output

#Final detection layer
def yolo_layer(inputs, n_classes, anchors, img_size, data_format):
    n_anchors = len(anchors)
    #Last Convolution
    output = Conv2D(filters=n_anchors * (5 + n_classes),
                    kernel_size=1, strides=1, use_bias=True,
                    data_format=data_format)(inputs)
    
    shape_output = output.get_shape().as_list()

    #grid_shapes stores the number of cells for each dimension
    if data_format == 'channels_first':
        grid_shape = shape_output[2:4]
        output = tf.transpose(output, [0,2,3,1])
    else:
        grid_shape = shape_output[1:3]

    print(n_anchors, shape_output, grid_shape[0], grid_shape[1])
    output_reshaped = tf.reshape(output, [-1, n_anchors*grid_shape[0]*grid_shape[1], 5+n_classes])
    #For example from (1,13,13,255) to (1,3*13*13,85)

    #Now we can get the values of the bounding boxes
    box_centers, box_shapes, confidence, classes = tf.split(output_reshaped, [2,2,1,n_classes], axis=-1)
    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])
    '''
    Compute the top-left coordinates for the cells, x_y_offset.
    x_y_offset is an array of couples that has to be added to box_centers.
    Since we have three boxes for each cell, each couple of values that represents the coordinates 
    of a cell has to be repeated for three times. 
    x_y_offset = [[[0. 0.],[0. 0.],[0. 0.],[1. 0.],[1. 0.],[1. 0.],[2. 0.],...,[12. 12.],[12. 12.],[12. 12.]]]
    '''
    x = tf.range(grid_shape[0], dtype=tf.float32) 
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2]) #now x_y_offset has the right shape (as explained above)
    '''
    Sigmoid squashes the predicted values in box_centers in a range from 0 to 1,
    because we want to avoid that the center lies in a cell that is not the one considered, 
    after the sum between the predicted x and y coordinates and the top-left coordinates of the cell.
    '''
    box_centers = tf.nn.sigmoid(box_centers)
    #Adding the offset
    box_centers = (box_centers + x_y_offset) * strides

    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    #Computing the actual width and height on the feature map
    box_shapes = tf.exp(box_shapes) * tf.to_float(anchors)

    confidence = tf.nn.sigmoid(confidence)
    #Softmaxing class scores assume that the classes are mutually exclusive
    classes = tf.nn.sigmoid(classes)

    boxes = tf.concat([box_centers, box_shapes,
                        confidence, classes], axis=-1)

    return boxes

    
