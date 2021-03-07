import numpy as np
import tensorflow as tf
from tensorflow.keras import Model


def load_yolo_weights(model, weights_file):
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

            '''
            print("-------------------------------------------------")
            print("Investigate values of conv_layer load weights...")
            conv_layer = model.get_layer(conv_layer_name)
            print("filters -->", end=' ')
            filters = conv_layer.filters
            print(filters)
            print("k_size -->", end=' ')
            k_size = conv_layer.kernel_size[0]
            print(k_size)
            print("in_dim -->", end=' ')
            in_dim = conv_layer.input_shape[-1]
            print(in_dim)
            print("Done!...")
            print("-------------------------------------------------")
            '''
            
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
            
            '''
            print("number of elements read fromfile")
            print(np.product(conv_shape))

            aux = np.shape(conv_weights)
            print("Dimension of Array from np.fromfile with count prod(conv_shape)")
            print(aux)
            '''
            
            '''
            print("Test value of conv_weights")
            #print(conv_weights)
            print(np.shape(conv_weights))
            '''
          

            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
            
            '''
            print("Test value of conv_weights")
            print(conv_weights)
            print(np.shape(conv_weights))

            print("Iteration number: "+str(i))
            '''
            

            if i not in range2:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'