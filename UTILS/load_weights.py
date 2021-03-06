import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

import struct

'''
#EXTRA
import os

import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL

import cv2
from numpy import expand_dims
import tensorflow as tf

from keras import backend as K
from keras.layers import Input, Lambda, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.models import load_model, Model
from keras.layers.merge import add, concatenate
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.patches import Rectangle
'''

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

#END


def custom_load_dkw(model, weights_file):
  """
  I agree that this code is very ugly, but I don’t know any better way of doing it.
  """
  print("Within LoadW f")

  print("Open weights file...")
  wf = open(weights_file, 'rb')
  print(wf)
  print("Done")
  major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

  j = 0
  for i in range(54,75):
      conv_layer_name = 'conv2d_%d' %i #if i > 0 else 'conv2d'
      bn_layer_name = 'batch_normalization_%d' %j #if j > 0 else 'batch_normalization'

      conv_layer = model.get_layer(conv_layer_name)
      filters = conv_layer.filters
      k_size = conv_layer.kernel_size[0]
      in_dim = conv_layer.input_shape[-1]

      if i not in [58, 66, 74]:
          # darknet weights: [beta, gamma, mean, variance]
          bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
          # tf weights: [gamma, beta, mean, variance]
          bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
          bn_layer = model.get_layer(bn_layer_name)
          j += 1
      else:
          conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

      # darknet shape (out_dim, in_dim, height, width)
      conv_shape = (filters, in_dim, k_size, k_size)
      conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
      # tf shape (height, width, in_dim, out_dim)
      conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

      if i not in [58, 66, 74]:
          print("!!!")
          print(conv_weights)
          conv_layer.set_weights([conv_weights])
          bn_layer.set_weights(bn_weights)
      else:
          print("???")
          print(conv_weights)
          conv_layer.set_weights([conv_weights, conv_bias])

  assert len(wf.read()) == 0, 'failed to read all data'
  wf.close()


def load_darknet_weights(model, weights_file):
      

  wf = open(weights_file, 'rb')
  major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
  layers = YOLO_V3_LAYERS

  for layer_name in layers:
    print(layer_name)
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

def load_yolo_weights_2(model, weights_file):
    tf.keras.backend.clear_session() # used to reset layer names
    
    # load Darknet original weights to TensorFlow model
    range1 = 75 
    range2 = [58, 66, 74]
    
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(range1):
            if i > 0:
                conv_layer_name = 'conv2d_%d' %i
            else:
                conv_layer_name = 'conv2d'
                
            if j > 0:
                bn_layer_name = 'batch_normalization_%d' %j
            else:
                bn_layer_name = 'batch_normalization'
            
            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in range2:
                # darknet weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in range2:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'

class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,    = struct.unpack('i', w_f.read(4))
            minor,    = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)
            
            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def load_weights(self, model):
        for i in range(106):
            try:
                conv_layer = model.get_layer('conv2d_' + str(i))
                print("loading weights of convolution #" + str(i))

                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer('batch_normalization_' + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    beta  = self.read_bytes(size) # bias
                    gamma = self.read_bytes(size) # scale
                    mean  = self.read_bytes(size) # mean
                    var   = self.read_bytes(size) # variance            

                    weights = norm_layer.set_weights([gamma, beta, mean, var])  

                if len(conv_layer.get_weights()) > 1:
                    print("1")
                    bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    print("2")
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                print("no convolution #" + str(i))     
    
    def reset(self):
        self.offset = 0    