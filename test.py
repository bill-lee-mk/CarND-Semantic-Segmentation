#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 12:06:47 2019

@author: lilei
"""

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_layer, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


weights_initializer_stddev = 0.01
def custom_init(stddev=weights_initializer_stddev):
    return tf.random_normal_initializer(stddev=stddev)



weights_regularized_l2 = 1e-3
def custom_regu(scale=weights_regularized_l2):
    return tf.contrib.layers.l2_regularizer(scale=scale)

# extracrt features
def conv_1x1(x, num_output):
    kernel_size = 1 #  1 by 1 convolution
    strides = 1
    return tf.layers.conv2d(x, 
                            num_output, 
                            kernel_size, 
                            strides, 
                            padding = 'same', 
                            kernel_initializer=tf.random_normal_initializer(stddev=weights_initializer_stddev), 
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weights_regularized_l2))

# increase the height and width dimensions of the 4D input tensor    
def upsample(x, num_output, kernel_size, strides):
    return tf.layers.conv2d_transpose(x,
                                      num_output,
                                      kernel_size= kernel_size,
                                      strides= strides,
                                      padding= 'same',
                                      kernel_initializer=tf.random_normal_initializer(stddev=weights_initializer_stddev), 
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weights_regularized_l2))

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
 
    # print the shape
    tf.Print(vgg_layer7_out, [tf.shape(vgg_layer7_out)])
    
    # replace FC layer 7 to 1x1 Convolution to preserve spatial information
    conv_1x1_layer7 = conv_1x1(vgg_layer7_out, num_classes)
    
    # print the shape
    tf.Print(conv_1x1_layer7, [tf.shape(conv_1x1_layer7)])
    

    # upsample deconvolution x 2
    first_upsamplex2 = upsample(conv_1x1_layer7, num_classes, 4, 2)
    
    # print the shape
    tf.Print(first_upsamplex2, [tf.shape(first_upsamplex2)])
    
    # replace FC layer 4 to 1x1 Convolution to preserve spatial information
    conv_1x1_layer4 = conv_1x1(vgg_layer4_out, num_classes)
    
    # print the shape
    tf.Print(conv_1x1_layer4, [tf.shape(conv_1x1_layer4)])    
    
    # skip connection: elementwise addition of 2 layers 
    first_skip = tf.add(first_upsamplex2, conv_1x1_layer4)
    
    # print the shape
    tf.Print(first_skip, [tf.shape(first_skip)])
    

    # upsample deconvolution x 2
    second_upsamplex2 = upsample(first_skip, num_classes, 4, 2)
    
    # print the shape
    tf.Print(second_upsamplex2, [tf.shape(second_upsamplex2)])
    
    
    # replace FC layer 3 to 1x1 Convolution to preserve spatial information
    conv_1x1_layer3 = conv_1x1(vgg_layer3_out, num_classes)

    # print the shape
    tf.Print(conv_1x1_layer3, [tf.shape(conv_1x1_layer3)])    
    
    # skip connection: elementwise addition of 2 layers
    second_skip = tf.add(second_upsamplex2, conv_1x1_layer3)
    
    # print the shape
    tf.Print(second_skip, [tf.shape(second_skip)]) 
    
    
    third_upsamplex8 = upsample(second_skip, num_classes, 16, 8)   
    
    # print the shape
    tf.Print(third_upsamplex8, [tf.shape(third_upsamplex8)]) 
    
    return third_upsamplex8

tests.test_layers(layers)


sess = tf.Session()    
input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, './data/vgg')
  
tf.Print(layer7_out, [tf.shape(layer7_out)])
tf.Print(layer7_out, [tf.shape(layer7_out)[1:3]])

tf.Print(layer4_out, [tf.shape(layer4_out)])
tf.Print(layer3_out, [tf.shape(layer3_out)])



conv_1x1_layer7 = conv_1x1(layer7_out, 2)












with tf.Session() as sess:
#    # Path to vgg model
#    vgg_path = os.path.join(data_dir, 'vgg')
#    # Create function to get batches
#    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    # OPTIONAL: Augment Images for better results
    #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

    # TODO: Build NN using load_vgg, layers, and optimize function

    # Placeholders
#    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
#    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # Getting layers from vgg.
    input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, './data/vgg')
    
    tf.Print(layer7_out, [tf.shape(layer7_out)])
    
    tf.Print(layer7_out, [tf.shape(layer7_out)[1:3]])
































