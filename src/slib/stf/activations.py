import tensorflow as tf

call_count = 0

def prelu(inputs, name=None):
    global call_count
    if name is None:
        name = 'prelu_{}'.format(call_count)
    with tf.name_scope('prelu'):
        alphas = tf.get_variable(name + '_alphas',
                                 inputs.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.25),
                                 dtype=tf.float32,
                                 trainable=True)
        pos = tf.nn.relu(inputs, name=name + '_relu')
        neg = alphas * (inputs - abs(inputs)) * 0.5
        output = pos + neg

    return output
