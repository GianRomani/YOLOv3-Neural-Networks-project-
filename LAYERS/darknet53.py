import common_layers

#Darknet53 -> feature extraction
def darknet53(inputs, training, data_format):
    inputs = conv2d_with_padding(inputs, filters=32, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=leaky_relu)

    inputs = conv2d_with_padding(inputs, filters=64, kernel_size=3, strides=2, data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)

    inputs = darknet_residual(inputs, filters=32, training=training, data_format=data_format)

    inputs = conv2d_with_padding(inputs, filters=128, kernel_size=3, strides=2, data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)

    for i in range(2):
        inputs = darknet_residual(inputs, filters=64, training=training, data_format=data_format)

    inputs = conv2d_with_padding(inputs, filters=256, kernel_size=3, strides=2, data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)

    for i in range(8):
        inputs = darknet_residual(inputs, filters=128, training=training, data_format=data_format)

    route1 = inputs

    inputs = conv2d_with_padding(inputs, filters=512, kernel_size=3, strides=2, data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)

    for i in range(8):
        inputs = darknet_residual(inputs, filters=256, training=training, data_format=data_format)

    route2 = inputs

    inputs = conv2d_with_padding(inputs, filters=1024, kernel_size=3, strides=2, data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= leaky_relu)

    for i in range(4):
        inputs = darknet_residual(inputs, filters=512, training=training, data_format=data_format)

    return route1, route2, inputs

