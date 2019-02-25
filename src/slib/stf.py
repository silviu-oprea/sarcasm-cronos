import functools

import tensorflow as tf


def define_scope(f):
    attribute = '_cache_' + f.__name__

    @property
    @functools.wraps(f)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(f.__name__):
                setattr(self, attribute, f(self))
        return getattr(self, attribute)

    return decorator


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def get_init(fan_in):
    stddev = np.math.sqrt(2 / fan_in)
    kernel_init_normal = tf.constant_initializer(
        rng.normal(0, stddev, [fan_in, 1])
    )
    bias_init_constant = tf.constant_initializer(np.full([1], 0.01).astype(np.float32))

    return kernel_init_normal, bias_init_constant


def prelu(_x, name):
    alphas = tf.get_variable(name, _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.25),
                             dtype=tf.float32,
                             trainable=True)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def dense(input_tensor, fan_in, fan_out, activation=None, regularize=False, name=None):
    kernel_init, bias_init = stf.get_init(fan_in)

    kernel_reg = None
    bias_reg = None
    if regularize:
        kernel_reg = tf.contrib.layers.l2_regularizer(1.)
        bias_reg = tf.contrib.layers.l2_regularizer(1.)

    layer = tf.layers.dense(input_tensor, fan_out,
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            kernel_regularizer=kernel_reg,
                            bias_regularizer=bias_reg)
    layer = activation(layer, name=name + '_activation')
    return layer
