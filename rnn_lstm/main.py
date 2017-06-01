import data_provider as dp
import lstm
import tensorflow as tf

def main():
  raw_data = dp.ptb_raw_data(r'simple-examples/data/')
  train_data, valid_data, test_data, _ = raw_data

  config = lstm.SmallConfig()
  eval_config = lstm.SmallConfig()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        train_input = lstm.LSTMInput(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = lstm.LSTMNetwork(is_training=True, config=config, input=train_input)
        #tf.summary.scalar("Training Loss", m.cost)
        #tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
        valid_input = lstm.LSTMInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = lstm.LSTMNetwork(is_training=False, config=config, input=valid_input)

        #tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
        test_input = lstm.LSTMInput(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = lstm.LSTMNetwork(is_training=False, config=config, input=test_input)

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