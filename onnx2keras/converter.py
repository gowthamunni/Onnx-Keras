"""
The ONNX to keras converter module
"""

from tensorflow import keras
import logging
import inspect
from onnx import numpy_helper
from .layers import AVAILABLE_CONVERTERS
import onnx
import pdb



def onnx_node_attributes_to_dict(args):
    """
    Parse ONNX attributes to Python dictionary
    :param args: ONNX attributes object
    :return: Python dictionary
    """
    def onnx_attribute_to_dict(onnx_attr):
        """
        Parse ONNX attribute
        :param onnx_attr: ONNX attribute
        :return: Python data type
        """
        if onnx_attr.HasField('t'):
            return numpy_helper.to_array(getattr(onnx_attr, 't'))

        for attr_type in ['f', 'i', 's']:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ['floats', 'ints', 'strings']:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))
    return {arg.name: onnx_attribute_to_dict(arg) for arg in args}


def onnx_to_keras(onnx_model, input_names,
                  input_shapes=None, name_policy=None, verbose=True, change_ordering=False):
    """
    Convert ONNX graph to Keras model format
    :param onnx_model: loaded ONNX model
    :param input_names: list with input names
    :param input_shapes: override input shapes (experimental)
    :param name_policy: override layer names. None, "short" or "renumerate" (experimental)
    :param verbose: verbose output
    :param change_ordering: change ordering to HWC (experimental)
    :return: Keras model
    """
    
    # Use channels first format by default.
    keras_fmt = keras.backend.image_data_format()
    keras.backend.set_image_data_format('channels_last')

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger('onnx2keras')

    logger.info('Converter is called.')

    onnx_weights = onnx_model.graph.initializer
    onnx_inputs = onnx_model.graph.input
    onnx_outputs = [i.name for i in onnx_model.graph.output]
    onnx_nodes = onnx_model.graph.node

    logger.debug('List input shapes:')
    logger.debug(input_shapes)

    logger.debug('List inputs:')

    for i, input in enumerate(onnx_inputs):
        logger.debug('Input {0} -> {1}.'.format(i, input.name))

    logger.debug('List outputs:')
    for i, output in enumerate(onnx_outputs):
        logger.debug('Output {0} -> {1}.'.format(i, output))

    logger.debug('Gathering weights to dictionary.')
    weights = {}
    for onnx_w in onnx_weights:
        try:
            if len(onnx_w.ListFields()) < 4:
                onnx_extracted_weights_name = onnx_w.ListFields()[1][1]
            else:
                onnx_extracted_weights_name = onnx_w.ListFields()[2][1]
            weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)
        except:
            onnx_extracted_weights_name = onnx_w.ListFields()[3][1]
            weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)

        logger.debug('Found weight {0} with shape {1}.'.format(
                     onnx_extracted_weights_name,
                     weights[onnx_extracted_weights_name].shape))

    layers = dict()
    lambda_funcs = dict()
    keras_outputs = []
    keras_inputs = []

    for i, input_name in enumerate(input_names):
        for onnx_i in onnx_inputs:
            if onnx_i.name == input_name:
                if input_shapes:
                    input_shape = input_shapes[i]
                else:
                    onnx_input_shape = [i.dim_value for i in onnx_i.type.tensor_type.shape.dim]
                    if len(onnx_input_shape) == 4 :
                        N, C, H, W = onnx_input_shape
                        input_shape = [H, W, C]

                layers[input_name] = keras.layers.InputLayer(
                    input_shape=input_shape, name=input_name
                ).output

                keras_inputs.append(layers[input_name])

                logger.debug('Found input {0} with shape {1}'.format(input_name, input_shape))

    # Convert every operation separable
    node_names = []
    for node_index, node in enumerate(onnx_nodes):
       
        node_type = node.op_type
        node_params = onnx_node_attributes_to_dict(node.attribute)

        # Add global converter info:
        node_params['name_policy'] = name_policy

        node_name = str(node.output[0])
        
        keras_names = []
        for output_index, output in enumerate(node.output):
            
            if name_policy == 'short':
                keras_name = keras_name_i = str(output)[:8]
                suffix = 1
                while keras_name_i in node_names:
                    keras_name_i = keras_name + '_' + str(suffix)
                    suffix += 1
                keras_names.append(keras_name_i)
            elif name_policy == 'renumerate':
                postfix = node_index if len(node.output) == 1 else "%s_%s" % (node_index, output_index)
                keras_names.append('LAYER_%s' % postfix)
            else:
                keras_names.append(output)

        if len(node.output) != 1:
            logger.warning('Trying to convert multi-output node')
            node_params['_outputs'] = list(node.output)
            node_names.extend(keras_names)
        else:
            keras_names = keras_names[0]
            node_names.append(keras_names)

        logger.debug('######')
        logger.debug('...')
        logger.debug('Converting ONNX operation')
        logger.debug('type: %s', node_type)
        logger.debug('node_name: %s', node_name)
        logger.debug('node_params: %s', node_params)
        logger.debug('...')

        logger.debug('Check if all inputs are available:')
        if len(node.input) == 0 and node_type != 'Constant':
            raise AttributeError('Operation doesn\'t have an input. Aborting.')

        for i, node_input in enumerate(node.input):
            logger.debug('Check input %i (name %s).', i, node_input)
            if node_input not in layers:
                logger.debug('The input not found in layers / model inputs.')
                if node_input in weights:
                    logger.debug('Found in weights, add as a numpy constant.')
                    layers[node_input] = weights[node_input]
                # added the condition to remove blank inputs 
                elif node_input == "":
                    while node_input in node.input:
                        node.input.remove(node_input)
                
                else:
                    raise AttributeError('Current node is not in weights / model inputs / layers.')
        else:
            logger.debug('... found all, continue')
        keras.backend.set_image_data_format('channels_last')
        
        AVAILABLE_CONVERTERS[node_type](
            node,
            node_params,
            layers,
            lambda_funcs,
            node_name,
            keras_names
        )
        if isinstance(keras_names, list):
            keras_names = keras_names[0]

        try:
            logger.debug('Output TF Layer -> ' + str(layers[keras_names]))
        except KeyError:
            pass

    # Check for terminal nodes
    
    for layer in onnx_outputs:
        if layer in layers:
            keras_outputs.append(layers[layer])


    # Create model
    model = keras.models.Model(inputs=keras_inputs, outputs=keras_outputs)
    
    # maybe instead of returning the model, saving it as SavedModel instead.
    return model
