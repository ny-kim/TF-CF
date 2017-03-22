import time
from optparse import OptionParser
from collections import deque

import pandas as pd
import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2

import dataio
import ops

np.random.seed(13575)

BATCH_SIZE = 1000
USER_NUM = 772
ITEM_NUM = 17548
DIM = 15
EPOCH_MAX = 100
DEVICE = "/cpu:0"


def clip(x):
    return np.floor(x)


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def _read_csv(path):
    return pd.read_csv(path,
                       index_col='id',
                       delimiter=',',
                       quotechar='|',
                       quoting=csv.QUOTE_NONE,
                       escapechar='\\')

def _indexing(train_items, test_items):
    items = pd.Series(train_items + test_items).unique().tolist()
    new_ids = map(str, list(range(len(items))))
    return dict(zip(items, new_ids))


def get_data(train_path, test_path):
    raw_train_set = _read_csv(train_path)
    raw_test_set = _read_csv(test_path)

    train_set = raw_train_set[raw_train_set.type == 'explicit']
    test_set = raw_test_set[raw_test_set.type == 'explicit']

    users = _indexing(train_set.source.tolist(), test_set.source.tolist())
    tracks = _indexing(train_set.target.tolist(), test_set.target.tolist())

    train_set.replace({'source': users, 'target': tracks}, inplace=True)
    test_set.replace({'source': users, 'target': tracks}, inplace=True)

    return (train_set, test_set)


def svd(train, test):
    samples_per_batch = len(train) // BATCH_SIZE

    iter_train = dataio.ShuffleIterator([train['source'],
                                         train['target'],
                                         train['weight']],
                                        batch_size=BATCH_SIZE)

    iter_test = dataio.OneEpochIterator([test['source'],
                                         test['target'],
                                         test['weight']],
                                        batch_size=-1)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items})
                    pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,
                                                       end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-tr', '--train',
                      dest='train_path',
                      help='training dataset file path',
                      metavar='FILE')
    parser.add_option('-te', '--test',
                      dest='test_path',
                      help='testing dataset file path',
                      metavar='FILE')
    (option, _) = parser.parse_args()

    (trains, tests) = get_data(option.train_path, option.test_path)
    svd(trains, tests)
    print("Done!")
