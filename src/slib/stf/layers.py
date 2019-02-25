from slib.stf import initializers
import numpy as np
import tensorflow as tf

from slib.stf import initializers


def flat_glorot(fan_in, shape):
    stddev = np.math.sqrt(2 / fan_in)
    return tf.truncated_normal(shape, mean=0, stddev=stddev)


def variable_summaries(var, var_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(var_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def dense_log(name,
          inputs,
          fan_in,
          fan_out,
          activation=None,
          verbose=False):
    with tf.name_scope(name):
        with tf.name_scope('declare_weights'):
            stddev = np.math.sqrt(2 / fan_in)
            initial_weights = tf.truncated_normal(
                [fan_in, fan_out],
                mean=0, stddev=stddev, name='initial_weights')
            weights = tf.Variable(initial_weights,
                                  trainable=True, name='weights')
            if verbose:
                with tf.name_scope('summaries'):
                    variable_summaries(weights, 'weights')

        with tf.name_scope('declare_biases'):
            initial_biases = tf.constant(0.01,
                                         shape=[fan_out], name='initial_biases')
            biases = tf.Variable(initial_biases, name='biases')
            if verbose:
                with tf.name_scope('summaries'):
                    variable_summaries(biases, 'biases')

        with tf.name_scope('compute_Wx_plus_b'):
            pre_activations = tf.matmul(inputs, weights) + biases
            if verbose:
                with tf.name_scope('summaries'):
                    with tf.name_scope('pre_activations'):
                        tf.summary.histogram('histogram', pre_activations)

        activations = pre_activations
        if activation is not None:
            activations = activation(pre_activations)
            if verbose:
                with tf.name_scope('summaries'):
                    with tf.name_scope('activations'):
                        tf.summary.histogram('histogram', activations)

    return activations


def dense(input_tensor,
          fan_in,
          fan_out,
          activation=None,
          kernel_initializer=None,
          bias_initializer=None,
          name=None):
    if kernel_initializer is None:
        stddev = np.math.sqrt(2 / fan_in)
        kernel_initializer = tf.truncated_normal_initializer(
            mean=0, stddev=stddev)
    if bias_initializer is None:
        bias_initializer = tf.constant_initializer(0.01)

    layer = tf.layers.dense(input_tensor, fan_out,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            name=name)
    if activation is not None:
        layer = activation(layer)
    return layer

