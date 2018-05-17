from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import sys
import numpy as np

#Construct 2D frequency vs phase, then flatten one dimesion 4*4*64

def construct_fvp_2d(X_train, is_training):
    with tf.variable_scope('con_fvp'):
        conv1 = tf.layers.conv2d(X_train, 16, 3, strides=(2, 2), padding='same', activation=tf.nn.relu, name='conv1')
        conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1_pl = tf.layers.max_pooling2d(conv1_bn, 2, 2, name='pool1')
        conv2 = tf.layers.conv2d(conv1_pl, 32, 3, padding='same', activation=tf.nn.relu, name='conv2')
        conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2_pl = tf.layers.max_pooling2d(conv2_bn, 2, 2, name='pool2')
        conv3 = tf.layers.conv2d(conv2_pl, 64, 3, padding='same', activation=tf.nn.relu, name='conv3')
        conv3_bn = tf.layers.batch_normalization(conv3, training=is_training, name='conv3_bn')
        conv3_pl = tf.layers.max_pooling2d(conv3_bn, 2, 2, name='pool3')
        out = tf.contrib.layers.flatten(conv3_pl)
    return out
#Construct 2D time vs phase, then flatten one dimesion 4*4*64
def construct_tvp_2d(X_trian, is_training):
    with tf.variable_scope('con_tvp'):
        conv1 = tf.layers.conv2d(X_trian, 16, 3, strides=(2, 2), padding='same', activation=tf.nn.relu, name='conv1')
        conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1_pl = tf.layers.max_pooling2d(conv1_bn, 2, 2, name='poo11')
        conv2 = tf.layers.conv2d(conv1_pl, 32, 3, padding='same', activation=tf.nn.relu, name='conv2')
        conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2_pl = tf.layers.max_pooling2d(conv2_bn, 2, 2, name='pool2')
        conv3 = tf.layers.conv2d(conv2_pl, 64, 3, padding='same', activation=tf.nn.relu, name='conv3')
        conv3_bn = tf.layers.batch_normalization(conv3, training=is_training, name='conv3_bn')
        conv3_pl = tf.layers.max_pooling2d(conv3_bn, 2, 2, name='pool3')
        out = tf.contrib.layers.flatten(conv3_pl)
    return out

#Construct 1D DM 16*64
def construct_dm_1d(X_train, is_training):
    with tf.variable_scope('con_dm'):
        conv1 = tf.layers.conv1d(X_train, 32, 3, padding='same', activation=tf.nn.relu, name='conv1')
        conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1_pl = tf.layers.max_pooling1d(conv1_bn, 2, 2, name='pool1')
        conv2 = tf.layers.conv1d(conv1_pl, 64, 3, padding='same', activation=tf.nn.relu, name='conv2')
        conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2_pl = tf.layers.max_pooling1d(conv2_bn, 2, 2, name='pool2')
        out = tf.contrib.layers.flatten(conv2_pl)
    return out

#Construct 1D profiles 16*64
def construct_prof_1d(X_train, is_training):
    with tf.variable_scope('con_profi'):
        conv1 = tf.layers.conv1d(X_train, 32, 3, padding='same', activation=tf.nn.relu, name='conv1')
        conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1_pl = tf.layers.max_pooling1d(conv1_bn, 2, 2, name='pool1')
        conv2 = tf.layers.conv1d(conv1_pl, 64, 3, padding='same', activation=tf.nn.relu, name='conv2')
        conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2_pl = tf.layers.max_pooling1d(conv2_bn, 2, 2, name='pool2')
        out = tf.contrib.layers.flatten(conv2_pl)

    return out
# Merge model array shape [batch_size, 1024, 1]
def merge_model(model1, model2, model3, model4, rate=0.):
    with tf.variable_scope('merge_model'):
        model = tf.concat([model1, model2, model3, model4], axis=1)
        model_drop_1 = tf.layers.dropout(model, rate=rate)
        model_fc_1 = tf.layers.dense(model_drop_1, 512)
        mode_drop_2 = tf.layers.dropout(model_fc_1, rate=rate)
        logits = tf.layers.dense(mode_drop_2, 2)
        # output = tf.nn.softmax(logits)

    return logits


# image_size = 64
#
# x_1 = tf.placeholder(tf.float32, [None, image_size, image_size, 1], name='input1')
# x_2 = tf.placeholder(tf.float32, [None, image_size, image_size, 1], name='input2')
# x_3 = tf.placeholder(tf.float32, [None, image_size, 1], name='input3')
# x_4 = tf.placeholder(tf.float32, [None, image_size, 1], name='input4')
# y = tf.placeholder(tf.float32, [None, 2])
# model_fvp = construct_fvp_2d(x_1)
# model_tvp = construct_tvp_2d(x_2)
# model_dm = construct_dm_1d(x_3)
# model_prof = construct_prof_1d(x_4)
# model_merge = merge_model(model_fvp, model_tvp, model_dm, model_prof)
# print(model_merge)

