# encoding=utf8

import tensorflow as tf
from tensorflow.contrib import rnn
from attention import attention


class TextRNN(object):
    """
    A RNN for text classification.
    Uses an embedding layer, followed by a lstm/gru and attention layer.
    """
    def __init__(self, sequence_length, num_classes,keep_prob, l2_reg_lambda,
                 vocab_size, embedding_size, num_hidden, attention_size,
                 network='lstm', bi_drection=True):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")  # 初始化权重
            # print('>>>', self.W.shape,self.input_x.shape) #(18758, 128) (?, 56)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        # Define a lstm cell with tensorflow
        cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0) if network=='lstm' else rnn.GRUCell(num_hidden)

        if bi_drection:
            hiddens, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                             cell_bw=cell,
                                                             inputs=self.embedded_chars, dtype=tf.float32)
            outputs = tf.concat(hiddens, axis=2)
        else:
            outputs, states = tf.nn.dynamic_rnn(cell, self.embedded_chars, dtype=tf.float32)

        '''
        if network=='lstm':
            if single:# 单层 单向 动态lstm
                lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
                outputs, states = tf.nn.dynamic_rnn(lstm_cell, self.embedded_chars, dtype=tf.float32)
            else:# 单层 双向 动态lstm
                lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
                lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
                hiddens, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 inputs=self.embedded_chars,dtype=tf.float32)
                outputs = tf.concat(hiddens, axis=2)
            
        elif network=='gru':
            if single:
                gru = rnn.GRUCell(num_hidden)
                outputs, grulast_states = tf.nn.dynamic_rnn(gru, self.embedded_chars, dtype=tf.float32)
            else:
                gru_fw = rnn.GRUCell(num_hidden)
                gru_bw = rnn.GRUCell(num_hidden)
                hiddens, state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=gru_fw, cell_bw=gru_bw,
                    inputs=self.embedded_chars, dtype=tf.float32)
                outputs = tf.concat(hiddens, axis=2)'''

        # Attention layer
        with tf.name_scope("attention"):
            attention_output = attention(outputs, attention_size)
            drop = tf.nn.dropout(attention_output, keep_prob)
            print('drop', drop.shape)  # shape=(56, 256)

        # linear activation, using rnn inner loop last output
        with tf.name_scope("output"):
            num_hiddens = num_hidden*2 if bi_drection else num_hidden
            W = tf.get_variable('W', shape=[num_hiddens, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([num_classes]))

            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

            self.logits = tf.nn.xw_plus_b(drop, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, axis=1, name='predictions')

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            # 交叉熵损失函数参考：https://blog.csdn.net/mao_xiao_feng/article/details/53382790
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            ) + self.l2_reg_lambda * self.l2_loss#使用交叉熵损失函数

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


