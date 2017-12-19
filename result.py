import tensorflow as tf
import numpy as np
import konlpy as knp
import random as rand
import codecs
import os
import string
import sys
from collections import Counter

from flask import Flask ,request

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def data():
    query = request.form['query']
    number = request.form['num']

    result = get_result(query, int(number))

    if result == 'Keyword Value not Found':
        return str('[]'), 200

    else :
        return str(result), 200


def get_dict():
    vocab_to_int_path = './saves/vti.txt'
    int_to_vocab_path = './saves/itv.txt'

    vti_file = codecs.open(vocab_to_int_path, 'r', encoding='utf-8', errors='ignore')
    itv_file = codecs.open(int_to_vocab_path, 'r', encoding='utf-8', errors='ignore')

    vocab_to_int = {line.split(',')[0]: int(line.split(',')[1].replace('\n', '')) for line in vti_file.readlines()}
    int_to_vocab = {int(line.split(',')[0]): line.split(',')[1].replace('\n', '') for line in itv_file.readlines()}
    
    return vocab_to_int, int_to_vocab


def get_result(keyword, k):
    vocab_to_int, int_to_vocab = get_dict()
    
    invader_net = tf.Graph()

    with invader_net.as_default():
        inputs_ = tf.placeholder(tf.int32, shape=[None], name='inputs')
        targets_ = tf.placeholder(tf.int32, shape=[None, None], name='targets')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

        batch_array_ = tf.placeholder(tf.float32, [None, None], name='batch_array')
        k_ = tf.placeholder(tf.int32, name='k')

    n_vocab = len(int_to_vocab)
    n_embedding = 250
    with invader_net.as_default():
        embedding = tf.Variable(tf.random_uniform([n_vocab, n_embedding], -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)

    n_sampled = 150
    with invader_net.as_default():
        softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(n_vocab))

        loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, targets_, embed, n_sampled, n_vocab)

        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_).minimize(cost)

        saver = tf.train.Saver()

    with invader_net.as_default():
        normed_embedding = tf.nn.l2_normalize(embedding, dim=1)  # embed
        normed_array = tf.nn.l2_normalize(batch_array_, dim=1)

        cosine_similarity = tf.matmul(normed_array, tf.transpose(normed_embedding, [1, 0]))
        # closest_words = tf.argmax(cosine_similarity, 1)
        closest_words = tf.nn.top_k(cosine_similarity, k_)
        
    save_path = './checkpoints/invader_net.ckpt'

    with tf.Session(graph=invader_net) as sess:
        saver.restore(sess, save_path)
        
        try:
            test_word = vocab_to_int[keyword]
        except KeyError:
            print('Keyword Value not Found')
            return 'Keyword Value not Found'
            
        test_embed = sess.run(embed, feed_dict={inputs_: [test_word]})

        feed = {
            batch_array_: test_embed,
            k_: k
        }

        output = sess.run(closest_words, feed_dict=feed)
        vocab_outputs = [int_to_vocab[word] for word in output[1][0]]
        
        print(vocab_outputs)
        return vocab_outputs

if __name__ == '__main__' :
    app.run('0.0.0.0', '2000')