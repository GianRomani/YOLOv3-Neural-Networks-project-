import tensorflow as tf
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

    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs

#YOLOv3 
#def Yolov3(channels=3):