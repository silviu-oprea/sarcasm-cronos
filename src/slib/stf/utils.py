import functools

import tensorflow as tf

def define_scope(f):
    attribute = '_cache_' + f.__name__

    @property
    @functools.wraps(f)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope('declare_' + f.__name__):
                setattr(self, attribute, f(self))
        return getattr(self, attribute)

    return decorator


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
