# encoding=utf8

from __future__ import print_function
import tensorflow as tf
import numpy as np
import datetime
import data_helpers
from text_rnn import TextRNN

from config import Config

config = Config()


def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    rnn = TextRNN(
        sequence_length=config.timesteps,
        num_classes=config.num_classes,
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=config.embedding_dim,
        num_hidden=config.num_hidden,
        l2_reg_lambda=config.l2_reg_lambda,
        keep_prob=config.dropout_keep_prob,
        attention_size=config.attention_size
    )
    # Define training procedure
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    train_op = optimizer.minimize(rnn.loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(init)
        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), config.batch_size, config.training_steps)

        eval_min_loss = 100
        early_stop_steps = 0
        # Train loop. For each batch ...
        for epoch, batch in enumerate(batches, 1):
            x_batch, y_batch = zip(*batch)
            feed_dict_train = {
                rnn.input_x: x_batch,
                rnn.input_y: y_batch,
            }
            _, loss_, accuracy_ = sess.run([train_op, rnn.loss, rnn.accuracy], feed_dict_train)
            if epoch % 50 == 0:
                print("epoch:{} loss {:g}, acc {:g}".format(epoch, loss_, accuracy_))

            if epoch % 100 == 0:
                print("\nEvaluation:")
                feed_dict_eval = {
                    rnn.input_x: x_dev,
                    rnn.input_y: y_dev,
                }
                loss_, accuracy_ = sess.run([rnn.loss, rnn.accuracy], feed_dict_eval)
                time_str = datetime.datetime.now().isoformat()
                print("{}: epoch {}, loss {:g}, acc {:g}\n".format(time_str, epoch, loss_, accuracy_))
                if loss_ < eval_min_loss:
                    eval_min_loss = loss_
                else:
                    early_stop_steps += 1

                if early_stop_steps == config.early_stop_steps:
                    print('eval loss no improvment, early stopped!!')
                    break



def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = data_helpers.preprocess()
    # print(y_train)
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()