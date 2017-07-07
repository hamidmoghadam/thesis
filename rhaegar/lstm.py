import tensorflow as tf
import time
import numpy as np
import data_provider as dp
import inspect


class LSTMInput(object):
    """The input data."""

    def __init__(self, config, data, y_data, name=None):
        self.batch_size = batch_size = 1 #config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = len(data)#((len(data) // batch_size) - 1) // num_steps
        self.number_of_class = 3
        self.input_data, self.targets, self.i = dp.batch_producer(data,y_data,self.batch_size, num_steps, name=name)


class LSTMNetwork(object):
    def __init__(self, config, is_training, input):

        self._config = config
        self._is_training = is_training
        self._input = input

        batch_size = self._input.batch_size
        num_steps = self._input.num_steps
        number_of_class = self._input.number_of_class
        vocab_size = self._config.vocab_size
        size = self._config.hidden_size
        

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell

        if self._is_training and self._config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self._config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(self._config.num_layers)], state_is_tuple=True)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self._input.input_data)

        if self._is_training and self._config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self._config.keep_prob)

        outputs = []
        
        self._initial_state = cell.zero_state(1, tf.float32)#batch_size
        state = self._initial_state
        self.temp = []
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                (cell_output, state) = cell(inputs[:, time_step, :], state)
                #outputs.append(cell_output)
                outputs = cell_output

        output = outputs #tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])


        softmax_w = tf.get_variable("softmax_w", [size, number_of_class], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [number_of_class], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        logits = tf.reshape(logits,[1,-1])
        self._cost = tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels=self._input.targets)
        self._final_state = state

        if is_training:
            self._lr = tf.Variable(0.0, trainable=False)
            self._tvars = tf.trainable_variables()
            self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
            self._lr_update = tf.assign(self._lr, self._new_lr)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(self._cost)
            

            # Evaluate model
            correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(self._input.targets,1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    def set_lr(self, lr, session):
        session.run(self._lr_update, feed_dict={self._new_lr: lr})

    def run_epoch(self, session):
        """Runs the model on the given data."""
        start_time = time.time()
        accuracies = 0.0
        costs = 0.0
        iters = 0

        fetches = {
            "cost": self._cost,
            "final_state": self._final_state,
            "accuracy":self._accuracy
        }

        if self._is_training:
            fetches["optimizer"] = self.optimizer
            #fetches["train_op"] = self._train_op

        state = session.run(self._initial_state)

        for step in range(self._input.epoch_size):
            feed_dict = {}
            for i, (c, h) in enumerate(self._initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]
            accuracy = vals["accuracy"]
            accuracies += accuracy
            costs += cost
            iters += self._input.num_steps

        return costs/ self._input.epoch_size, accuracy / self._input.epoch_size


class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 3
    hidden_size = 50
    max_epoch = 4
    max_max_epoch = 5
    keep_prob = 1.0
    lr_decay = 0.95
    batch_size = 5
    vocab_size = 8000


class BestConfig(object):
    init_scale = 0.1
    learning_rate = 0.5
    max_grad_norm = 5
    num_layers = 1
    num_steps = 40
    hidden_size = 700
    max_epoch = 1
    max_max_epoch = 5
    keep_prob = 1.0
    lr_decay = 0.98
    batch_size = 0
    vocab_size = 8000#49432#10000
