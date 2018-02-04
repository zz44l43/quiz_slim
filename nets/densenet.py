"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block', isTraining=False):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    if isTraining:
        current = slim.dropout(current, scope=scope + '_dropout')
    current = slim.avg_pool2d(current, [2,2])
    return current

def transition_block(net, num_outputs, kernel_size = [1,1], scope='transition', is_trainning=False):
    current = slim.batch_norm(net, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_transition')
    if is_trainning:
        current = slim.dropout(current, scope=scope + '_dropout')
    return current

def transition_to_classes(net, n_classes =10):
    current = slim.batch_norm(net, scope='_transition_to_classes')
    current = tf.nn.relu(current)
    pool_kernal = int(current.get_shape()[-2])
    current = slim.avg_pool2d(net, [pool_kernal, pool_kernal])
    logits = slim.fully_connected(current, n_classes)
    return logits

def block(net, layers, growth, scope='block', isTraining=False):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx), isTraining=False)
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx),isTraining = isTraining)
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5
    layers_per_block = 250
    n_channels =16
    first_conv_output_number = 16

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]) as sc:
        end_points
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:

            # From the paper: Before enterting the first dense block, a convolution with 16 output channels is performed on the input images.
            with tf.variable_scope("first_conv_layer"):
                print("first_conv_layer")
                net = slim.conv2d(images, first_conv_output_number, [3,3])

            #From the paper: 1st Desity block follow by a transition blcok
            with tf.variable_scope("block_1"):
                print("first_block")
                net = block(net, layers_per_block,growth,isTraining=is_training)
                n_channels += growth*layers_per_block
                with tf.variable_scope("transition_1"):
                    net = transition_block(net, n_channels, is_trainning=is_training)

            #From the paper: 2nd Desity block follow by a transition blcok
            with tf.variable_scope("block_2"):
                print("2nd_block")
                net = block(net, layers_per_block,growth,isTraining=is_training)
                n_channels += growth*layers_per_block
                with tf.variable_scope("transition_2"):
                    net = transition_block(net, n_channels, is_trainning=is_training)

            with tf.variable_scope("block_3"):
                print("thrid_block")
                net = block(net, layers_per_block,growth,isTraining=is_training)
                n_channels += growth*layers_per_block
                with tf.variable_scope("transition_layer_to_classes"):
                    net = transition_to_classes(net, num_classes)
            logits = tf.reshape(net, [-1,num_classes])
            end_points = slim.utils.convert_collection_to_dict(end_points)
            return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
