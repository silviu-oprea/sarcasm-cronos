import numpy as np
import tensorflow as tf

np.random.seed(9876)
rng = np.random.RandomState(9876)


def flat_glorot(fan_in, shape, given_rng=None):
    if given_rng is None:
        given_rng = rng
    stddev = np.math.sqrt(2 / fan_in)
    initializer = tf.constant_initializer(given_rng.normal(0, stddev, shape))
    return initializer


def constant(shape, value):
    return tf.constant_initializer(np.full(shape, value).astype(np.float32))
