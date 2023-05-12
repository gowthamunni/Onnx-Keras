from tensorflow import keras
import numpy as np
import logging


def convert_upsample(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert upsample.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:upsample')
    logger.warning('!!! EXPERIMENTAL SUPPORT (upsample) !!!')


    # added from here.
    # import pdb; pdb.set_trace()
    # if params["coordinate_transformation_mode"].decode('utf-8') == "pytorch_half_pixel":
    #     if len(node.input) > 2:
    #         assert AttributeError('More than 3 input for upsampling layer.')
        
    #     N , H, W, C = layers[node.input[1]]
    #     inN , inC, inH, inW = layers[node.input[0]].shape
    #     sizes = H / inH , W/ inW 
    #     if params['mode'].decode('utf-8') == 'linear':
    #         upsampling = keras.layers.UpSampling2D(size=sizes,data_format="channels_first", interpolation="bilinear",name=keras_name)
    #         layers[node_name] = upsampling(layers[node.input[0]])

    #     elif params['mode'].decode('utf-8') == 'nearest':
    #         upsampling = keras.layers.UpSampling2D(size=sizes,data_format="channels_first", interpolation='nearest',name=keras_name)
    #         layers[node_name] = upsampling(layers[node.input[0]])
    #     else:
    #         logger.error('Cannot convert non-nearest and non linear upsampling/Resize.')
    #         raise AssertionError('Cannot convert non-nearest and non linear upsampling/Resize.')

    # to here.
    
    # else:
    if "scales" in params:
        # for opset version - 7
        if len(node.input) != 1:
            raise AttributeError('Unsupported number of inputs')
        scale = np.uint8(params['scales'][-2:])

    else:
        # for opset version - 9+
        # Upsample since opset version 9 uses input[1] as 'scales' instead of attributes.
        scale = np.uint8(layers[node.input[1]][-2:])

    if params['mode'].decode('utf-8') != 'nearest':
        logger.error('Cannot convert non-nearest upsampling.')
        raise AssertionError('Cannot convert non-nearest upsampling')

    upsampling = keras.layers.UpSampling2D(
            size=scale, name=keras_name
        )

    layers[node_name] = upsampling(layers[node.input[0]])
