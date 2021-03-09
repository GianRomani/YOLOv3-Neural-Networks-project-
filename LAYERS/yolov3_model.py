import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from LAYERS.darknet53 import *
from LAYERS.common_layers import *
from UTILS.boxes import *

#upsample, out_shape is obtained from the shape of route1 or route2
#Nearest neighbor interpolation is used to unsample inputs to out_shape
def upsample(inputs, out_shape, data_format):

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width = out_shape[1]

    inputs = tf.image.resize(inputs, [new_height, new_width], method='nearest')

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs

#YOLOv3 model
def yolov3(n_classes, model_size, anchors, iou_threshold, confidence_threshold, data_format, activation, channels=3, training=False):
    
    #input = Input([None,None,3])
    x = inputs = Input([model_size[0],model_size[1],3]) #Per ora ho messo 416x416 anche per darknet53
    #Backbone
    route1, route2, inputs = darknet53(inputs, activation, name='yolo_darknet')#(inputs)
    
    #Detect1
    route, inputs = yolo_convolution_block(inputs, 512, training, data_format, activation, name='yolo_conv0')

    detect1 = yolo_layer(inputs, n_classes, anchors[6:9], model_size, data_format, name='yolo_layer0')

    inputs = convolutional_block(route, 256, 1, training, data_format, activation, name='conv_block0')
    #inputs = batch_norm(inputs, training, data_format) 
    #inputs = tf.nn.leaky_relu(inputs, alpha = activation)
    #inputs = LeakyReLU(alpha=activation)(inputs)
    upsample_size = route2.get_shape().as_list()
    inputs = upsample(inputs, upsample_size, data_format)
    inputs = tf.concat([inputs,route2], axis=3)
    #Detect2
    route, inputs = yolo_convolution_block(inputs, 256, training, data_format, activation, name='yolo_conv1')
    
    detect2 = yolo_layer(inputs, n_classes, anchors[3:6], model_size, data_format, name='yolo_layer1')

    inputs = convolutional_block(route, 128, 1, training, data_format, activation, name='conv_block1')
    #inputs = batch_norm(inputs, training, data_format) 
    #inputs = tf.nn.leaky_relu(inputs, alpha = activation)
    #inputs = LeakyReLU(alpha=activation)(inputs)
    upsample_size = route1.get_shape().as_list()
    inputs = upsample(inputs, upsample_size, data_format)
    inputs = tf.concat([inputs,route1], axis=3)
    #Detect3
    route, inputs = yolo_convolution_block(inputs, 128, training, data_format, activation, name='yolo_conv2')
    
    detect3 = yolo_layer(inputs, n_classes, anchors[0:3], model_size, data_format, name='yolo_layer2')

    inputs = tf.concat([detect1, detect2, detect3], axis=1)

    #inputs = build_boxes(inputs)

    #inputs = nms(inputs, n_classes, iou_threshold, confidence_threshold)

    aux = Model(x, inputs, name='yolov3')

    #aux.summary()

    return aux