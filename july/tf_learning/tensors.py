#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

CLASS_NUM = 10

def labels2OneHot(labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense( concated, tf.pack([batch_size, CLASS_NUM]), 1.0, 0.0)

    return onehot_labels


labels_placeholders = tf.placeholder(tf.int32, shape=(5))
oneHostLabels = labels2OneHot(labels_placeholders)

sess = tf.Session()
feed_dict = { labels_placeholders: [1, 3, 7, 9, 0] }

print(sess.run(oneHostLabels, feed_dict=feed_dict))


