import collections
import functools
from collections import defaultdict

from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import pymongo

from slib import tokenizers
from slib import sitools
from slib import stf
np.random.seed(9876)

# riloff_sarcasm_markers = {'#sarcasm', '#sarcastic'}
riloff_sarcasm_markers = {}
ptacek_sarcasm_markers = {'#satire', '#satira', '#sarcazm',
                          '#sarkazm', '#sarcasm', '#sarkasm',
                          '#sarkasmus', '#sarcastic', '#sarkastik',
                          '#ironie', '#irony'}

DataBatch = collections.namedtuple(
    'DataBatch',
    ['word_idx_mat', 'uid_idx_list', 'targets', 'doc_len_list', 'we_cart_mask'])
Data = collections.namedtuple('Data', ['train', 'valid', 'test'])
DataStats = collections.namedtuple('DataStats', ['max_doc_len'])
Resources = collections.namedtuple(
    'Resources',
    ['word_embeds', 'word_embed_dim', 'usr_embeds', 'usr_embed_dim'])


def get_data_gen(tweets_col,
                 buckets_col,
                 batch_size,
                 buckets=None,
                 min_doc_len=0,
                 mongo_host='localhost',
                 mongo_port=27017,
                 stopwords=None):
    mc = pymongo.MongoClient(mongo_host, mongo_port)
    db = mc['sarcasm_icwsm']
    tweets_col = db[tweets_col]
    buckets_col = db[buckets_col]

    base_query = {'num_words': {'$gte': min_doc_len}}
    if buckets is not None: base_query['bucket'] = {'$in': buckets}
    pos_query = {'label': 1}
    for k, v in base_query.items():
        pos_query[k] = v
    neg_query = {'label': 0}
    for k, v in base_query.items():
        neg_query[k] = v
    pos_bs = list(buckets_col.find(pos_query).sort('num_words', pymongo.ASCENDING))
    neg_bs = list(buckets_col.find(neg_query).sort('num_words', pymongo.ASCENDING))
    num_pos = buckets_col.count(pos_query)
    num_neg = buckets_col.count(neg_query)
    total = num_pos + num_neg

    # Compute number of positive and negative examples per batch,
    # such that the overall class ratio is maintained is each batch.
    class_ratio = len(pos_bs) / len(neg_bs)
    nr_neg_per_batch = batch_size / (class_ratio + 1)
    nr_pos_per_batch = nr_neg_per_batch * class_ratio
    nr_neg_per_batch = int(nr_neg_per_batch)
    nr_pos_per_batch = int(nr_pos_per_batch)
    if nr_pos_per_batch + nr_neg_per_batch < batch_size:
        nr_neg_per_batch += 1
    assert nr_pos_per_batch + nr_neg_per_batch == batch_size

    print('[get_data_gen] {} positive and {} negative examples per batch'
          .format(nr_pos_per_batch, nr_neg_per_batch))
    pos_idx = neg_idx = 0
    batch_pos_left = batch_neg_left = 0
    it_num = 1

    while True:
        if batch_pos_left == 0 and batch_neg_left == 0:
            batch_pos_left = nr_pos_per_batch
            batch_neg_left = nr_neg_per_batch
        else:
            if batch_pos_left > 0 and pos_idx < len(pos_bs):
                bs = pos_bs[pos_idx]
                pos_idx += 1
                batch_pos_left -= 1
            elif batch_neg_left > 0 and neg_idx < len(neg_bs):
                bs = neg_bs[neg_idx]
                neg_idx += 1
                batch_neg_left -= 1
            else:
                if pos_idx < len(pos_bs):
                    bs = pos_bs[pos_idx]
                    pos_idx += 1
                elif neg_idx < len(neg_bs):
                    bs = neg_bs[neg_idx]
                    neg_idx += 1
                else:
                    mc.close()
                    raise StopIteration
            if it_num % 500 == 0:
                print('Loading example {} / {}'.format(it_num, total))
            tweet = tweets_col.find_one({'id': bs['tid']})
            text = tweet['text']
            tokens = tokenizers.tokenize_tweet(text,
                                               stopwords=stopwords,
                                               allowed_punctuation={'.', '?', '!'})

            yield bs['tid'], bs['uid'], bs['label'], tokens
            it_num += 1


def get_word_embeds(word2freq,
                    min_word_freq=1,
                    dim=100,
                    mongo_host='localhost',
                    mongo_port=27017,
                    mongo_col='embeddings_glove'):
    word2vec = {}
    unk_words = []

    mc = pymongo.MongoClient(mongo_host, mongo_port)
    glove_col = mc['resources'][mongo_col]
    for word, freq in word2freq.items():
        if freq >= min_word_freq:
            rec = glove_col.find_one({'word': word, 'dim': dim})
            if rec is not None:
                word2vec[word] = rec['vec']
            else:
                word2vec[word] = np.random.uniform(-0.1, 0.1, dim)
                unk_words.append(word)
    mc.close()

    known_percentage = len(unk_words) / len(word2vec) * 100
    print('[get_word_embeds] {} out of {} vocab words ({:.1f} %) were not found in glove {}'
          .format(len(unk_words), len(word2vec), known_percentage, unk_words[:10]))

    word2idx = {}
    embeds = np.zeros(shape=[len(word2vec) + 1, dim], dtype='float32')
    for idx, (word, vec) in enumerate(word2vec.items(), start=1):
        embeds[idx] = vec
        word2idx[word] = idx

    return embeds, word2idx


def get_static_word_embeds(word_embeds_file,
                           word2freq,
                           min_word_freq=1):
    word2vec = {}
    unk_words = []
    # print('[get_static_word_embeds] Reading word embeddings')
    # glove =

    for word, freq in word2freq.items():
        if freq >= min_word_freq:
            rec = glove_col.find_one({'word': word, 'dim': dim})
            if rec is not None:
                word2vec[word] = rec['vec']
            else:
                word2vec[word] = np.random.uniform(-0.1, 0.1, dim)
                unk_words.append(word)
    mc.close()

    known_percentage = len(unk_words) / len(word2vec) * 100
    print('[get_word_embeds] {} out of {} vocab words ({:.1f} %) were not found in glove {}'
          .format(len(unk_words), len(word2vec), known_percentage, unk_words[:10]))

    word2idx = {}
    embeds = np.zeros(shape=[len(word2vec) + 1, dim], dtype='float32')
    for idx, (word, vec) in enumerate(word2vec.items(), start=1):
        embeds[idx] = vec
        word2idx[word] = idx

    return embeds, word2idx


def get_usr_embeds(uids, path_and_dim_list):
    # assert len(path_and_dim_list) > 0, 'Please provide usr embed source and dim'
    total_dim = functools.reduce(lambda sum, t: sum + t[1], path_and_dim_list, 0)
    embeds = np.zeros([len(uids), total_dim], dtype='float32')

    uid2idx = dict(sitools.zip_with_index(uids))
    # Start index in array
    start_vec_idx = 0

    print('[get_usr_embeds] Loading usr embeds with total dim {}'.format(total_dim))
    for path, dim in path_and_dim_list:
        print('[get_usr_embeds] Loading usr embeds from {} with dim {}'.format(path, dim))

        end_vec_idx = start_vec_idx + dim

        with open(path, 'r') as file:
            for line in file:
                # Read uid and vector
                fields = line.split()
                file_uid = int(fields[0][1:])
                file_vec = [float(x) for x in fields[1:]]

                # Continue if we don't need this uid
                if file_uid not in uids:
                    print('u{} not needed'.format(file_uid))
                    continue

                usr_idx = uid2idx[file_uid]
                embeds[usr_idx, start_vec_idx: end_vec_idx] = file_vec

        start_vec_idx = end_vec_idx

    return embeds, total_dim, uid2idx


def vectorize_data(data, word2idx, uid2idx, max_doc_len):
    # One row for each document
    X = [] # [<index of word in word embedding matrix>]
    U = [] # [<index of user in user embedding matrix>]
    Y = [] # [<list of labels>]
    DL = [] # [<list of doc lens>]
    for tid, uid, label, doc in data:
        word_idx_doc = []
        for word in doc[:max_doc_len]:
            if word in word2idx:
                word_idx_doc.append(word2idx[word])
            # else:
            #     word_idx_doc.append(0)
        DL.append(len(word_idx_doc))
        while len(word_idx_doc) < max_doc_len:
            word_idx_doc.append(0)
        X.append(word_idx_doc)
        U.append(uid2idx[uid])
        Y.append(label)
    X, U, Y, DL = np.array(X), np.array(U), np.array(Y), np.array(DL)

    return X, U, Y, DL


def do_upsample(X, Y, DL, U):
    x_width = X.shape[1]
    DL = np.expand_dims(DL, 1)
    U = np.expand_dims(U, 1)

    all_X = np.concatenate((X, U, DL), axis=1)
    try:
        # sm = SMOTE(random_state=797, k_neighbors=2, kind='svm')
        num_pos = sum(Y)
        num_neg = len(Y) - sum(Y)
        num_pos = min(num_neg, max(num_pos, len(Y) // 2))
        ros = RandomOverSampler()
        all_X, Y = ros.fit_sample(all_X, Y)
        # all_X, Y = sm.fit_sample(all_X, Y)
    except:
        pass
    X = all_X[:, :x_width]
    U = all_X[:, x_width:x_width + 1]
    DL = all_X[:, x_width + 1:]

    # sanity checks
    assert X.shape[0] == Y.shape[0] == U.shape[0] == DL.shape[0]
    assert DL.shape[1] == 1
    assert U.shape[1] == 1

    DL = DL[:, 0]
    U = U[:, 0]
    return X, Y, DL, U


def create_mask(doc_lens, max_doc_len):
    r = np.arange(max_doc_len)
    cart_idx = np.transpose([np.repeat(r, len(r)), np.tile(r, len(r))])
    masks = []
    for dl in doc_lens:
        mask = np.empty_like(cart_idx)
        mask[:] = cart_idx
        mask[np.where(np.diff(mask, axis=1) == 0)[0]] = 0
        mask[np.where(mask >= dl)[0]] = 0
        mask[np.where(np.sum(mask, axis=1) > 0)[0]] = 1
        mask = mask[:, :1]
        # mask = np.concatenate([np.zeros([pad_len, 1]), mask, np.zeros([pad_len, 1])])
        masks.append(mask)

    return np.array(masks)


def batch_data(X, Y, U, DL, batch_size, max_doc_len, upsample=False, rng=None):
    if rng is None:
        rng = np.random.RandomState(9876)

    fill_space = batch_size - len(Y) % batch_size
    if fill_space > 0:
        fill_idxs = rng.permutation(len(Y))[:fill_space]
        X = np.append(X, X[fill_idxs], axis=0)
        Y = np.append(Y, Y[fill_idxs], axis=0)
        U = np.append(U, U[fill_idxs], axis=0)
        DL = np.append(DL, DL[fill_idxs], axis=0)
    assert len(Y) % batch_size == 0

    batches = []
    num_batches = len(Y) // batch_size

    for i in range(num_batches):
        bX = X[i * batch_size: (i + 1) * batch_size]
        bU = U[i * batch_size: (i + 1) * batch_size]
        bY = Y[i * batch_size: (i + 1) * batch_size]
        bDL = DL[i * batch_size: (i + 1) * batch_size]
        if upsample:
            bX, bY, bDL, bU = do_upsample(bX, bY, bDL, bU)
        bM = create_mask(bDL, max_doc_len)
        perm = rng.permutation(batch_size)

        batches.append(DataBatch(bX[perm], bU[perm], bY[perm], bDL[perm], bM[perm]))
    return batches


def get_data_and_resources(tweets_col,
                           buckets_col,
                           batch_size,
                           train_buckets,
                           valid_buckets,
                           test_buckets,
                           word_embeds_col,
                           word_embed_dim,
                           usr_embeds_files,
                           min_word_freq=1,
                           min_doc_len=0,
                           max_doc_len=None,
                           upsample=False,
                           mongo_host='localhost',
                           mongo_port=27107,
                           rng=None):
    if rng is None:
        rng = np.random.RandomState(9876)
    if tweets_col == 'riloff_tweets':
        stopwords = riloff_sarcasm_markers
    else:
        stopwords = ptacek_sarcasm_markers

    print('[get_data] Load train data')
    train_data = list(
        get_data_gen(tweets_col, buckets_col, batch_size, train_buckets, min_doc_len,
                     mongo_host, mongo_port, stopwords))
    print('[get_data] Load valid data')
    valid_data = list(
        get_data_gen(tweets_col, buckets_col, batch_size, valid_buckets, min_doc_len,
                     mongo_host, mongo_port, stopwords))
    print('[get_data] Load test data')
    test_data = list(
        get_data_gen(tweets_col, buckets_col, batch_size, test_buckets, min_doc_len,
                     mongo_host, mongo_port, stopwords))

    print('[get_data] Compute vocab')
    word2freq = defaultdict(int)
    comp_max_doc_len = 0
    uids = set()

    # tid, uid, label
    for tid, uid, label, doc in train_data:
        uids.add(uid)
        comp_max_doc_len = max(comp_max_doc_len, len(doc))
        for word in doc:
            word2freq[word] += 1
    for tid, uid, label, doc in valid_data:
        uids.add(uid)
        comp_max_doc_len = max(comp_max_doc_len, len(doc))
        for word in doc:
            word2freq[word] += 1
    for tid, uid, label, doc in test_data:
        uids.add(uid)
        comp_max_doc_len = max(comp_max_doc_len, len(doc))
        for word in doc:
            word2freq[word] += 1
    if max_doc_len is not None:
        max_doc_len = min(max_doc_len, comp_max_doc_len)
    else:
        max_doc_len = comp_max_doc_len
    print('[get_data] Max doc len is {}'.format(max_doc_len))

    word_embeds, word2idx = get_word_embeds(
        word2freq, min_word_freq, word_embed_dim,
        mongo_host, mongo_port, word_embeds_col)
    usr_embeds, usr_embed_dim, uid2idx = get_usr_embeds(
        uids, [(f, 100) for f in usr_embeds_files])

    tX, tU, tY, tDL = vectorize_data(train_data, word2idx, uid2idx, max_doc_len)
    vX, vU, vY, vDL = vectorize_data(valid_data, word2idx, uid2idx, max_doc_len)
    tstX, tstU, tstY, tstDL = vectorize_data(
        test_data, word2idx, uid2idx, max_doc_len)

    train = batch_data(tX, tY, tU, tDL, batch_size, max_doc_len,
                       upsample=upsample, rng=rng)
    valid = batch_data(vX, vY, vU, vDL, batch_size, max_doc_len, rng=rng)
    test = batch_data(tstX, tstY, tstU, tstDL, batch_size, max_doc_len, rng=rng)

    return Data(train, valid, test), \
           DataStats(max_doc_len), \
           Resources(word_embeds, word_embed_dim, usr_embeds, usr_embed_dim)


def get_static_data_and_resources(tweets_file,
                                  word_embeds_file,
                                  usr_embeds_files,
                                  batch_size,
                                  train_buckets,
                                  valid_buckets,
                                  test_buckets,
                                  min_word_freq=1,
                                  min_doc_len=0,
                                  max_doc_len=None,
                                  upsample=False,
                                  rng=None):
    if rng is None:
        rng = np.random.RandomState(9876)

    print('[get_data] Load tweets')
    tweets = pd.read_csv(tweets_file, sep='\t')
    train_data = tweets.loc[tweets['bucket'].isin(train_buckets),
                        ['tid', 'uid', 'label', 'tokens']].values
    valid_data = tweets.loc[tweets['bucket'].isin(valid_buckets),
                        ['tid', 'uid', 'label', 'tokens']].values
    test_data = tweets.loc[tweets['bucket'].isin(test_buckets),
                        ['tid', 'uid', 'label', 'tokens']].values

    print('[get_data] Compute vocab')
    word2freq = defaultdict(int)
    comp_max_doc_len = 0
    uids = set()

    # tid, uid, label
    for tid, uid, label, doc in train_data:
        uids.add(uid)
        comp_max_doc_len = max(comp_max_doc_len, len(doc))
        for word in doc:
            word2freq[word] += 1
    for tid, uid, label, doc in valid_data:
        uids.add(uid)
        comp_max_doc_len = max(comp_max_doc_len, len(doc))
        for word in doc:
            word2freq[word] += 1
    for tid, uid, label, doc in test_data:
        uids.add(uid)
        comp_max_doc_len = max(comp_max_doc_len, len(doc))
        for word in doc:
            word2freq[word] += 1
    if max_doc_len is not None:
        max_doc_len = min(max_doc_len, comp_max_doc_len)
    else:
        max_doc_len = comp_max_doc_len
    print('[get_data] Max doc len is {}'.format(max_doc_len))

    word_embeds, word2idx = get_static_word_embeds(
        word_embeds_file, word2freq, min_word_freq)
    usr_embeds, usr_embed_dim, uid2idx = get_usr_embeds(
        uids, [(f, 100) for f in usr_embeds_files])

    tX, tU, tY, tDL = vectorize_data(train_data, word2idx, uid2idx, max_doc_len)
    vX, vU, vY, vDL = vectorize_data(valid_data, word2idx, uid2idx, max_doc_len)
    tstX, tstU, tstY, tstDL = vectorize_data(
        test_data, word2idx, uid2idx, max_doc_len)

    train = batch_data(tX, tY, tU, tDL, batch_size, max_doc_len,
                       upsample=upsample, rng=rng)
    valid = batch_data(vX, vY, vU, vDL, batch_size, max_doc_len, rng=rng)
    test = batch_data(tstX, tstY, tstU, tstDL, batch_size, max_doc_len, rng=rng)

    return Data(train, valid, test), \
           DataStats(max_doc_len), \
           Resources(word_embeds, word_embed_dim, usr_embeds, usr_embed_dim)

if __name__ == '__main__':
    data = get_static_data_and_resources('data/tweets/riloff.txt',
                                         '',
                                         '',
                                         16,
                                         [0, 1, 2, 3, 4, 5, 6, 7],
                                         [8],
                                         [9])
