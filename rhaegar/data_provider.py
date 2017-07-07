# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np

import tensorflow as tf


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()


def build_vocab(data):
    #data = _read_words(filename)

    counter = collections.Counter(data.split(' '))
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(1, len(words)+1)))
    print("vocab size is {0}".format(len(word_to_id)))
    return word_to_id


def text_to_word_ids(data, word_to_id):
    #data = _read_words(filename)
    return [word_to_id[word] for word in data.split(' ') if word in word_to_id]

def pad_word_ids(word_ids, max_length):
    data_len = len(word_ids)
         
    if data_len < max_length:
        word_ids = np.lib.pad(word_ids, (max_length - data_len, 0), 'constant').tolist()

    return word_ids[:max_length]

def batch_produce(raw_data, y_raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "PTBProducer", [raw_data, y_raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        y_raw_data = tf.convert_to_tensor(y_raw_data, name="y_raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)

        data = tf.reshape(raw_data, [1, -1])
        y_data = tf.reshape(y_raw_data, [1, -1])
       

        i = tf.train.range_input_producer(batch_size, shuffle=False).dequeue()

        x = tf.strided_slice(data, [0, i * num_steps], [0, (i + 1) * num_steps])
        x.set_shape([1,num_steps])

        y = tf.strided_slice(y_data, [0,i * 3], [0 , (i+1) * 3])
        y.set_shape([1, 3])

        print(['-----x------', x])
        print(['-----y------', y])

        return x, y


def batch_producer(raw_data, y_raw_data, batch_size, num_steps, number_of_class, name=None):
    
    with tf.name_scope(name, "PTBProducer", [raw_data, y_raw_data, batch_size, num_steps]):

        len_data = len(raw_data)

        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        y_raw_data = tf.convert_to_tensor(y_raw_data, name="y_raw_data", dtype=tf.int32)

        i = tf.train.range_input_producer(len_data, shuffle=False).dequeue()

        x = raw_data[i, :] #tf.slice(raw_data, [i, 0], [i, num_steps])
        x = tf.reshape(x, [1, num_steps])

        y = y_raw_data[i, :] #tf.slice(y_raw_data, [i,0], [i,3])
        y = tf.reshape(y, [1 ,number_of_class])


        return x, y, i
