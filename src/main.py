import sys
import numpy as np
import os

import tensorflow as tf

from slib import stf
from model import Model
from data_and_resources_acl import get_data_and_resources
import metrics
np.random.seed(9876)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tweets_col', None, """Tweets collection""")
tf.app.flags.DEFINE_string('buckets_col', None, """Buckets collection""")
tf.app.flags.DEFINE_integer('batch_size', None, """Batch size""")
tf.app.flags.DEFINE_list('train_buckets', None, """Batch size""")
tf.app.flags.DEFINE_list('valid_buckets', None, """Batch size""")
tf.app.flags.DEFINE_list('test_buckets', None, """Batch size""")

tf.app.flags.DEFINE_string(
    'word_embeds_col', 'embeddings_glove', """Word embeddings collection""")
tf.app.flags.DEFINE_integer(
    'word_embed_dim', 100, """Dimension of word embeddings""")
tf.app.flags.DEFINE_list(
    'usr_embeds_files', [], """Files containing user embeddings""")

tf.app.flags.DEFINE_integer(
    'min_word_freq', 1,
    """Minimum frequency of a word in corpus.
    If it occurs less than this, it will be omitted or replaced with <unk>.""")
tf.app.flags.DEFINE_integer(
    'min_doc_len', 5, """Minimum number of words in document""")
tf.app.flags.DEFINE_integer(
    'max_doc_len', 40, """Maximum number of words in document""")
tf.app.flags.DEFINE_boolean(
    'upsample', False,
    """Minimum frequency of a word in corpus.
    If it occurs less than this, it will be omitted or replaced with <unk>.""")

tf.app.flags.DEFINE_string('mongo_host', 'localhost', """Mongo host""")
tf.app.flags.DEFINE_integer('mongo_port', 27017, """Mongo port""")


tf.app.flags.DEFINE_string(
    'summaries_dir',
    os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                 'tensorflow/sarcasm_cronos/summaries'),
    """Directory where to write summaries for tensorboard""")

tf.app.flags.DEFINE_float('learning_rate', 0.001, """Learning rate""")
tf.app.flags.DEFINE_float('grad_clip_norm', 1, """Gradient clip norm""")

tf.app.flags.DEFINE_list(
    'layers', ['ia', 'lstm', 'usr_embed'],
    """Layers. One or more of ia, lstm and usr_embed""")
tf.app.flags.DEFINE_integer(
    'lstm_dim', 100,
    """Dimension of the LSTM layer""")
tf.app.flags.DEFINE_integer(
    'polysemy_dim', 4,
    """Dimension of the polysemy ia layer""")
tf.app.flags.DEFINE_integer(
    'dense_dim', 100,
    """Dimension of the dense normalization layer""")
tf.app.flags.DEFINE_integer(
    'conv_k1_height', 3, """Height of the 1st convolutional kernel""")
tf.app.flags.DEFINE_integer(
    'conv_k2_height', 5, """Height of the 2st convolutional kernel""")
tf.app.flags.DEFINE_integer(
    'conv_k3_height', 7, """Height of the 3st convolutional kernel""")
tf.app.flags.DEFINE_integer(
    'conv_num_feat_maps', 400, """Number of convolutional feature maps""")

tf.app.flags.DEFINE_float(
    'dropout_rate', 0,
    """Dropout rate""")
tf.app.flags.DEFINE_integer(
    'num_epochs', 100,
    """Number of epochs""")
tf.app.flags.DEFINE_integer(
    'summary_record_freq', 10,
    """Record summary every <summary_record_freq> epochs.""")
rng = np.random.RandomState(9876)
tf.set_random_seed(9876)


def define_graph(data_stats, resources):
    # ======================================================================== #
    # Data placeholders
    word_idx_mat = tf.placeholder(tf.int32, shape=[None, data_stats.max_doc_len])
    uid_idx_list = tf.placeholder(tf.int32, shape=[None])
    targets = tf.placeholder(tf.int32,
                             shape=[None],
                             name='targets')
    doc_len_list = tf.placeholder(tf.int32,
                                  shape=[None],
                                  name='doc_len_list')
    we_cart_mask = tf.placeholder(
        tf.float32,
        shape=[None, data_stats.max_doc_len * data_stats.max_doc_len, 1],
        name='we_cart_mask')

    lstm_init_h = tf.placeholder(tf.float32,
                                 shape=[None, FLAGS.lstm_dim],
                                 name='lstm_init_h')
    lstm_init_c = tf.placeholder(tf.float32,
                                 shape=[None, FLAGS.lstm_dim],
                                 name='lstm_init_c')

    is_training = tf.placeholder_with_default(True, shape=())
    # ======================================================================== #
    # Model
    model = Model(word_idx_mat,
                  uid_idx_list,
                  targets,
                  doc_len_list,
                  we_cart_mask,
                  lstm_init_h,
                  lstm_init_c,
                  is_training,
                  data_stats,
                  resources,
                  FLAGS,
                  rng)

    # tf.summary.scalar('loss', model.loss)
    # summary = tf.summary.merge_all()
    return stf.utils.AttrDict(locals())

# ============================================================================ #
# Train and valid
def get_feed_dict(graph, batch, is_training):
    lstm_init_h = rng.uniform(
        -0.01, 0.01, [FLAGS.batch_size, FLAGS.lstm_dim])
    lstm_init_c = rng.uniform(
        -0.01, 0.01, [FLAGS.batch_size, FLAGS.lstm_dim])

    feed_dict = {
        graph.word_idx_mat: batch.word_idx_mat,
        graph.uid_idx_list: batch.uid_idx_list,
        graph.targets: batch.targets,
        graph.doc_len_list: batch.doc_len_list,
        graph.we_cart_mask: batch.we_cart_mask,
        graph.lstm_init_h: lstm_init_h,
        graph.lstm_init_c: lstm_init_c,
        graph.is_training: is_training
    }
    return feed_dict


def run_train_epoch(sess, graph, data):
    losses = []

    for batch_num, batch in enumerate(data.train):
        batch_loss, _ = sess.run([graph.model.loss, graph.model.train],
                                 feed_dict=get_feed_dict(graph, batch, True))
        losses.append(batch_loss)
    loss = np.mean(losses)
    return loss


def run_valid_epoch(sess, graph, data):
    losses = []
    roc_acc = metrics.RocAccumulator()

    for batch_num, batch in enumerate(data):
        batch_loss, batch_outputs = sess.run(
            [graph.model.loss, graph.model.outputs],
            feed_dict=get_feed_dict(graph, batch, False))
        losses.append(batch_loss)
        roc_acc.add_batch(batch.targets, batch_outputs)
    loss = np.mean(losses)
    roc = roc_acc.get_prf()
    return loss, roc


global_val_fscore = 0
global_val_fscore_thresh = 0
global_test_fscore = 0
def print_epoch_stats(epoch,
                      train_loss, valid_loss, test_loss,
                      valid_roc, test_roc):
    global global_val_fscore
    global global_val_fscore_thresh
    global global_test_fscore

    val_fscore_thresh, val_fscore = max(valid_roc.fscore.items(),
                                        key=lambda t: t[1])
    val_prec = valid_roc.precision[val_fscore_thresh]
    val_recall = valid_roc.recall[val_fscore_thresh]

    test_prec = test_roc.precision[val_fscore_thresh]
    test_recall = test_roc.recall[val_fscore_thresh]
    test_fscore = test_roc.fscore[val_fscore_thresh]

    if val_fscore > global_val_fscore:
        global_val_fscore = val_fscore
        global_val_fscore_thresh = val_fscore_thresh
        global_test_fscore = test_fscore
    print('epoch {}: '
          'train loss {:.4f} - val loss {:.4f}'
          '\n\tval prec:\t{:.4f} [{}]\t-\ttest prec:\t{:.4f}'
          '\n\tval recall:\t{:.4f} [{}]\t-\ttest recall:\t{:.4f} '
          '\n\tval fscore:\t{:.4f} [{}]\t-\ttest fscore:\t{:.4f} '
          '<- global: val {:.4f} [{}] - test {:.4f}'
          '\n'
          .format(epoch,
                  train_loss, valid_loss,
                  val_prec, val_fscore_thresh, test_prec,
                  val_recall, val_fscore_thresh, test_recall,
                  val_fscore, val_fscore_thresh, test_fscore,
                  global_val_fscore, global_val_fscore_thresh,
                  global_test_fscore))


def run(graph, data):
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    valid_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/valid',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test',
                                        sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for epoch in range(FLAGS.num_epochs):
        train_loss = run_train_epoch(sess, graph, data)
        train_writer.add_summary(
            metrics.simple_scalar_summary('epoch_loss', train_loss), epoch)

        valid_loss, valid_roc = run_valid_epoch(sess, graph, data.valid)
        valid_writer.add_summary(
            metrics.simple_scalar_summary('epoch_loss', valid_loss), epoch)

        test_loss, test_roc = run_valid_epoch(sess, graph, data.test)
        valid_writer.add_summary(
            metrics.simple_scalar_summary('epoch_loss', test_loss), epoch)

        print_epoch_stats(epoch,
                          train_loss, valid_loss, test_loss,
                          valid_roc, test_roc)
    sess.close()

def main(argv=None):
    train_buckets = [int(b) for b in FLAGS.train_buckets]
    valid_buckets = [int(b) for b in FLAGS.valid_buckets]
    test_buckets = [int(b) for b in FLAGS.test_buckets]
    data, data_stats, resources = get_data_and_resources(FLAGS.tweets_col,
                                                         FLAGS.buckets_col,
                                                         FLAGS.batch_size,
                                                         train_buckets,
                                                         valid_buckets,
                                                         test_buckets,
                                                         FLAGS.word_embeds_col,
                                                         FLAGS.word_embed_dim,
                                                         FLAGS.usr_embeds_files,
                                                         FLAGS.min_word_freq,
                                                         FLAGS.min_doc_len,
                                                         FLAGS.max_doc_len,
                                                         FLAGS.upsample,
                                                         FLAGS.mongo_host,
                                                         FLAGS.mongo_port,
                                                         rng)
    graph = define_graph(data_stats, resources)
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    run(graph, data)

if __name__ == '__main__':
    tf.app.run(main=main, argv=sys.argv)
