# encoding=utf8

import os


class Config():
    def __init__(self):
        # general config
        self.learning_rate = 0.01
        self.training_steps = 200
        self.batch_size = 64
        self.display_step = 150
        self.evaluate_every = 150
        self.checkpoint_every = 150
        self.num_checkpoints = 3
        self.early_stop_steps = 5


        # Network Parameters
        self.embedding_size = 128
        self.num_hidden = 128  # hidden layer num of features
        self.num_classes = 2  # MNIST total classes (0-9 digits)
        self.dropout_keep_prob = 0.5
        self.l2_reg_lambda = 0.1

        # CNN parameters
        self.filter_sizes = '3,4,5'

        # RNN parameters
        self.network = 'lstm'
        self.bi_drection = True
        self.timesteps = 56  # timesteps
        self.attention_size = 100


        # Misc Parameters
        self.allow_soft_placement = True
        self.log_device_placement = False


        # Data loading params
        self.dev_sample_percentage = 0.1
        self.positive_data_file = "./data/rt-polaritydata/rt-polarity.pos"
        self.negative_data_file = "./data/rt-polaritydata/rt-polarity.neg"

        
