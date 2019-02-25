import numpy as np
import tensorflow as tf

from slib import stf
np.random.seed(9876)

class Model:
    def __init__(self,
                 word_idx_mat,
                 uid_idx_list,
                 targets,
                 doc_len_list,
                 we_cart_mask,
                 lstm_init_h,
                 lstm_init_c,
                 is_training,
                 # constants
                 data_stats,
                 resources,
                 FLAGS,
                 rng=None):
        """
        :param word_idx_mat: Indexes of words in word embedding matrix
        - Each row corresponds to a document.
        - Each entry in a row corresponds to a word in the document. It is an
        integer representing the index of the respective word in the word word
        embedding matrix
        [
           [1 3 2 4], # doc 1
           [2 1 6 9]  # doc 2
        ]
        :param uid_idx_list: Indexes of users of each document in user embedding
        matrix
        [7, 3, 4] # document 0 is written by user whose embedding is at index 7
        in the user embedding matrix
        :param targets: True labels, one for each document
        - [1, 0, 0, 1] # document 0 is sarcastic, document 1 is not
        :param doc_len_list: List of document lengths, one for each document
        - [12, 14, 7] # document 0 has 12 words, document 1 has 14
        :param we_cart_mask: Mask for word embedding cartesian (intra-attention)

        :param max_doc_len: Maximum document length
        :param word_embeds: The word embedding matrix. There is a row for each
        word in the vocabulary, of dimension word_embeds_dim.
        :param usr_embeds: The user embedding matrix. There is a row for each
        user, of dimension usr_embeds_dim.
        :param FLAGS:
        """
        self.word_idx_mat = word_idx_mat
        self.uid_idx_list = uid_idx_list
        self.targets = targets
        self.doc_len_list = doc_len_list
        self.we_cart_mask = we_cart_mask
        self.lstm_init_h = lstm_init_h
        self.lstm_init_c = lstm_init_c
        self.is_training = is_training

        # constants
        self.data_stats = data_stats
        self.resources = resources

        self.FLAGS = FLAGS

        if rng is None:
            rng = np.random.RandomState(9876)
        self.rng = rng

        self.embedded_docs, self.embedded_users
        self.conv_layer, self.ia_layer, self.lstm_layer, self.concat_layer
        self.normalization_layer,
        self.dropout
        self.logits, self.outputs, self.loss, self.optimizer, self.train

    @stf.utils.define_scope
    def embedded_docs(self):
        """
        Say we have a batch of 2 documents.

        Transforms

        word_idx_mat = [
           [19 4],    # doc 1
           [2 1 6 9]  # doc 2
        ]

        into

        embedded_docs = [
          [ # doc 1
            [3.2, 1.4, 2.6, 8.7] # embedding of word 19,
            [3.2, 1.4, 2.6, 8.7] # embedding of word 4
          ],
          [ ... # doc 2]
        ]
        """
        word_embeds_init = tf.constant(
            self.resources.word_embeds, dtype=tf.float32)
        word_embeds_var = tf.get_variable('word_embeds_var',
                                          initializer=word_embeds_init,
                                          trainable=True)
        return tf.nn.embedding_lookup(
            word_embeds_var, self.word_idx_mat, name='embedded_docs')

    @stf.utils.define_scope
    def embedded_users(self):
        """
        Say we have a batch of 2 documents.

        Transforms

        uid_idx_list = [12, 3] # document 0 is written by the user whose
                               # embedding is on index 12 in the user embedding
                               # matrix
        into

        embedded_users = [
          [1.2, 3.2, 5.4], # user 12
          [0.7, 9.2, 5.7]  # user 3
        ]
        """
        usr_embeds_init = tf.constant(
            self.resources.usr_embeds, dtype=tf.float32)
        usr_embeds_var = tf.get_variable('usr_embeds_var',
                                         initializer=usr_embeds_init,
                                         trainable=True)
        embedded_users = tf.nn.embedding_lookup(usr_embeds_var,
                                                self.uid_idx_list,
                                                name='embedded_users')
        return embedded_users

    def _conv_factory(self, inputs, idx, k_height, padded_doc_len):
        with tf.name_scope('conv_{}'.format(idx)):
            filter_shape = [k_height,
                            self.resources.word_embed_dim,
                            1,
                            self.FLAGS.conv_num_feat_maps]
            maxpool_shape = [1, padded_doc_len - k_height + 1, 1, 1]

            fan_in = k_height * self.resources.word_embed_dim
            stddev = np.math.sqrt(2 / fan_in)
            kernel_init_normal = self.rng.normal(0, stddev, filter_shape)\
                .astype(np.float32)
            conv_kernel = tf.get_variable('conv{}_kernel'.format(idx),
                                          initializer=kernel_init_normal)

            bias_init_constant = np.full([self.FLAGS.conv_num_feat_maps], 0.01).astype(np.float32)
            conv_bias = tf.get_variable('conv{}_bias'.format(idx),
                                        initializer=bias_init_constant)

            conv = tf.nn.conv2d(
                input=inputs,
                filter=conv_kernel,
                strides=[1, 1, 1, 1],
                padding='VALID')
            conv = tf.nn.bias_add(conv, conv_bias)
            conv = stf.activations.prelu(conv, 'conv{}_prelu'.format(idx))
            pool = tf.nn.max_pool(conv,
                                  ksize=maxpool_shape,
                                  strides=[1, 1, 1, 1],
                                  padding='VALID')
            pool_out = tf.reshape(pool, [-1, self.FLAGS.conv_num_feat_maps],
                                  name='conv{}'.format(idx))
            return pool_out

    @stf.utils.define_scope
    def conv_layer(self):
        with tf.name_scope('pad_and_reshape'):
            pad_len = max(self.FLAGS.conv_k1_height,
                          self.FLAGS.conv_k2_height,
                          self.FLAGS.conv_k3_height) - 1
            padded_doc_len = self.data_stats.max_doc_len + 2*pad_len
            padded_word_embeds = tf.pad(self.embedded_docs,
                                        [[0, 0], [pad_len, pad_len], [0, 0]])
            conv_in = tf.expand_dims(padded_word_embeds, 3)
        with tf.name_scope('build_conv_units'):
            conv1 = self._conv_factory(conv_in, 1,
                                       self.FLAGS.conv_k1_height, padded_doc_len)
            conv2 = self._conv_factory(conv_in, 2,
                                       self.FLAGS.conv_k2_height, padded_doc_len)
            conv3 = self._conv_factory(conv_in, 3,
                                       self.FLAGS.conv_k3_height, padded_doc_len)
        with tf.name_scope('concat_conv_units'):
            all_conv = tf.concat([conv1, conv2, conv3], axis=1)
            self.conv_layer_dim = 3 * self.FLAGS.conv_num_feat_maps
        return all_conv

    @stf.utils.define_scope
    def ia_layer(self):
        cart = tf.tile(self.embedded_docs, [1, 1, self.data_stats.max_doc_len])
        cart_shape = [-1,
                      self.data_stats.max_doc_len * self.data_stats.max_doc_len,
                      self.resources.word_embed_dim]
        t = tf.reshape(cart, cart_shape)
        t1 = tf.tile(self.embedded_docs, [1, self.data_stats.max_doc_len, 1])
        embed_cart = tf.concat([t, t1], axis=2)
        embed_cart = self.we_cart_mask * embed_cart

        # s_polysemy = stf.dense(embed_cart,
        #                        fan_in=2 * self.word_embed_dim,
        #                        fan_out=self.FLAGS.s_polysemy_dim,
        #                        activation=stf.prelu)
        # s_polysemy = self.we_cart_mask * s_polysemy
        # s = stf.dense(s_polysemy, fan_in=self.FLAGS.s_polysemy_dim, fan_out=1)

        s = stf.layers.dense(embed_cart,
                             fan_in=2 * self.resources.word_embed_dim,
                             fan_out=1)
        s = self.we_cart_mask * s
        s_rows = tf.reshape(
            s, [-1, self.data_stats.max_doc_len, self.data_stats.max_doc_len])
        s_rows_max = tf.expand_dims(tf.reduce_max(s_rows, axis=2), 2)
        v_at = s_rows_max * self.embedded_docs
        v_at = tf.reduce_sum(v_at, axis=1, name='v_at')
        return v_at

    @stf.utils.define_scope
    def lstm_layer(self):
        lstm = tf.nn.rnn_cell.LSTMCell(self.FLAGS.lstm_dim)
        init_state = tf.nn.rnn_cell.LSTMStateTuple(self.lstm_init_c,
                                                self.lstm_init_h)
        output, state = tf.nn.dynamic_rnn(lstm, self.embedded_docs,
                                           dtype=tf.float32,
                                           sequence_length=self.doc_len_list,
                                           initial_state=init_state)
        v_lstm = state.h
        return v_lstm

    @stf.utils.define_scope
    def concat_layer(self):
        layer_lst = []
        concat_layer_dim = 0
        if 'ia' in self.FLAGS.layers:
            ia_drop = tf.layers.dropout(self.ia_layer,
                                        rate=self.FLAGS.dropout_rate,
                                        training=self.is_training)
            layer_lst.append(ia_drop)
            concat_layer_dim += self.resources.word_embed_dim
        if 'lstm' in self.FLAGS.layers:
            lstm_drop = tf.layers.dropout(self.lstm_layer,
                                          rate=self.FLAGS.dropout_rate,
                                          training=self.is_training)
            layer_lst.append(lstm_drop)
            concat_layer_dim += self.FLAGS.lstm_dim
        if 'conv' in self.FLAGS.layers:
            conv_drop = tf.layers.dropout(self.conv_layer,
                                          rate=self.FLAGS.dropout_rate,
                                          training=self.is_training)
            layer_lst.append(conv_drop)
            concat_layer_dim += self.conv_layer_dim
        if 'usr_embed' in self.FLAGS.layers:
            usr_embed_drop = tf.layers.dropout(self.embedded_users,
                                               rate=self.FLAGS.dropout_rate,
                                               training=self.is_training)
            layer_lst.append(usr_embed_drop)
            concat_layer_dim += self.resources.usr_embed_dim
        self.concat_layer_dim = concat_layer_dim
        concat_layer = tf.concat(layer_lst, axis=1)
        return concat_layer

    @stf.utils.define_scope
    def normalization_layer(self):
        batch_norm = tf.layers.batch_normalization(self.concat_layer)
        dense = stf.layers.dense(batch_norm,
                                 fan_in=self.concat_layer_dim,
                                 fan_out=self.FLAGS.dense_dim,
                                 activation=stf.activations.prelu)
        return dense

    @stf.utils.define_scope
    def dropout(self):
        dropout = tf.layers.dropout(self.normalization_layer,
                                    rate=self.FLAGS.dropout_rate,
                                    training=self.is_training)
        return dropout

    @stf.utils.define_scope
    def logits(self):
        logits = stf.layers.dense(self.dropout,
                                  fan_in=self.concat_layer_dim,
                                  fan_out=2)
        return logits

    @stf.utils.define_scope
    def outputs(self):
        return tf.nn.softmax(self.logits)

    @stf.utils.define_scope
    def loss(self):
        targets_nograd = tf.stop_gradient(self.targets)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets_nograd, logits=self.logits)
        loss = tf.reduce_mean(loss)
        return loss

    @stf.utils.define_scope
    def optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate)
        return optimizer

    @stf.utils.define_scope
    def train(self):
        trainable_vars = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_vars)
        grads, _ = tf.clip_by_global_norm(
            grads, clip_norm=self.FLAGS.grad_clip_norm)
        train = self.optimizer.apply_gradients(
            zip(grads, trainable_vars),
            global_step=tf.train.get_or_create_global_step()
        )
        return train
