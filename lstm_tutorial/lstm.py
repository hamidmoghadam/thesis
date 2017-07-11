import tensorflow as tf
import time
import numpy as np
import reader
import inspect


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)

class LSTMNetwork(object):
    def __init__(self, config, is_training, input):

        self._config = config
        self._is_training = is_training
        self._input = input

        batch_size = self._input.batch_size
        num_steps = self._input.num_steps
        size = self._config.hidden_size
        vocab_size = self._config.vocab_size

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell

        if self._is_training and self._config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self._config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self._config.num_layers)], state_is_tuple=True)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self._input.input_data)

        if self._is_training and self._config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self._config.keep_prob)

        outputs = []

        self._initial_state = cell.zero_state(batch_size, tf.float32)
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._input.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])

        self._cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if is_training:
            self._lr = tf.Variable(0.0, trainable=False)
            self._tvars = tf.trainable_variables()
            self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
            self._lr_update = tf.assign(self._lr, self._new_lr)

            grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, self._tvars), self._config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(zip(grads, self._tvars),
                                                       global_step=tf.contrib.framework.get_or_create_global_step())


    def set_lr(self, lr, session):
        session.run(self._lr_update, feed_dict={self._new_lr: lr})

    def run_epoch(self, session):
        """Runs the model on the given data."""
        start_time = time.time()
        costs = 0.0
        iters = 0

        fetches = {
            "cost": self._cost,
            "final_state": self._final_state,
        }

        if(self._is_training):
            fetches["train_op"] = self._train_op

        state = session.run(self._initial_state)

        for step in range(self._input.epoch_size):
            feed_dict = {}
            for i, (c, h) in enumerate(self._initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]

            costs += cost
            iters += self._input.num_steps


        return np.exp(costs / iters)

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 25
  hidden_size = 100
  max_epoch = 4
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def main():
  raw_data = reader.ptb_raw_data(r'simple-examples/data/')
  train_data, valid_data, test_data, _ = raw_data

  config = SmallConfig()
  eval_config = SmallConfig()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = LSTMNetwork(is_training=True, config=config, input=train_input)
        #tf.summary.scalar("Training Loss", m.cost)
        #tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = LSTMNetwork(is_training=False, config=config, input=valid_input)

        #tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = LSTMNetwork(is_training=False, config=config, input=test_input)

    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        lr = config.learning_rate * lr_decay
        m.set_lr(lr, session)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, lr))

        train_perplexity = m.run_epoch(session)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

        valid_perplexity = mvalid.run_epoch(session)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = mtest.run_epoch(session)
      print("Test Perplexity: %.3f" % test_perplexity)


main()








