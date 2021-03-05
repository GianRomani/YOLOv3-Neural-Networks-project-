import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

YOLO_V3_LAYERS = [
  'yolo_darknet',
  'yolo_conv0',
  'yolo_layer0',
  'conv_block0',
  'yolo_conv1',
  'yolo_layer1',
  'conv_block1',
  'yolo_conv2',
  'yolo_layer2'
]

def load_darknet_weights(model, weights_file):
  wf = open(weights_file, 'rb')
  major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
  layers = YOLO_V3_LAYERS

  for layer_name in layers:
    sub_model = model.get_layer(layer_name)
    for i, layer in enumerate(sub_model.layers):
      if not layer.name.startswith('conv2d'):
        continue
      batch_norm = None
      if i + 1 < len(sub_model.layers) and \
            sub_model.layers[i + 1].name.startswith('batch_norm'):
        batch_norm = sub_model.layers[i + 1]

      #logging.info("{}/{} {}".format(
        #sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

      filters = layer.filters
      #print(filters)
      size = layer.kernel_size[0]
      #print("here")
      #print(layer.get_input_shape_at(0))
      #print(layer.get_input_shape_at(1)[1])
      #in_dim = layer.input_shape[-1]
      in_dim = (layer.get_input_shape_at(1)[3])

      if batch_norm is None:
        conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
      else:
        bn_weights = np.fromfile(
          wf, dtype=np.float32, count=4 * filters)

        bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

      conv_shape = (filters, in_dim, size, size)
      conv_weights = np.fromfile(
        wf, dtype=np.float32, count=np.product(conv_shape))

      conv_weights = conv_weights.reshape(
        conv_shape).transpose([2, 3, 1, 0])

      if batch_norm is None:
        layer.set_weights([conv_weights, conv_bias])
      else:
        layer.set_weights([conv_weights])
        batch_norm.set_weights(bn_weights)

  assert len(wf.read()) == 0, 'failed to read all data'
  wf.close()
