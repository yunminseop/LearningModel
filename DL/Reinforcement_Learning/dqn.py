import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=10, l_rate=1e-1):
        with tf.compat.v1.variable_scope(self.net_name):
            self._X = tf.compat.v1.placeholder(
                tf.float32, [None, self.input_size], name="input_x")
            W1 = tf.compat.v1.get_variable("W1", shape=[self.input_size, h_size], initializer=tf.keras.initializers.GlorotUniform())
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))

            W2 = tf.compat.v1.get_variable("W2", shape=[h_size, self.output_size], initializer=tf.keras.initializers.GlorotUniform())

            self._Qpred = tf.matmul(layer1, W2)

            self._Y = tf.compat.v1.placeholder(shape=[None, self.output_size], dtype=tf.float32)

            self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

            self._train = tf.compat.v1.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})
    
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
    