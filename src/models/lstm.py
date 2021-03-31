import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Layer, Dense, Conv1D, BatchNormalization, Dropout, LayerNormalization, LSTM, Embedding, Bidirectional
import numpy as np

class SharedBlock(Layer):

    def __init__(self, hidden_dim=1024, num_filters=128, lstm_units=10, num_blocks=2, vocabulary_size=184, embedding_out = 8,
                 text_lenght=5, in_lstm_units = 32, rate=0.1, name="shared block", **kwargs):
        super(SharedBlock, self).__init__(name=name, **kwargs)
        self.embedding = Embedding(vocabulary_size, embedding_out, input_length=text_lenght)
        self.bidirectional_lstm = Bidirectional(LSTM(in_lstm_units))
        self.lstm_layer = LSTM(lstm_units)
        self.dropout = Dropout(rate)
        self.cnn_layer = Conv1D(
            filters=num_filters,
            kernel_size=2,
            # Use 'same' padding so outputs have the same shape as inputs.
            padding='same')
        self.activation_layer = Activation('relu')
        self.dense = Dense(hidden_dim)
        self.layernorm = LayerNormalization(epsilon=1e-6)

    def call(self, x, z, training, mask=None):
        
        z = self.embedding(z)
        z = self.bidirectional_lstm(z)
        z = tf.reshape(tf.tile(z, [x.shape[1], 1]),
                       [x.shape[0], x.shape[1], z.shape[1]])
        x = tf.concat([x, z], -1)
        x = self.activation_layer(self.cnn_layer(x))
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.layernorm(x)

        return self.lstm_layer(x)

class Actor(Layer):

    def __init__(self, num_policies, hidden_dim=1024, num_filters=128, lstm_units=10, num_blocks=2, vocabulary_size=184,
                 embedding_out = 8, in_lstm_units = 32, text_lenght=5, name="actor", **kwargs):
        super(Actor, self).__init__(name=name, **kwargs)
        self.shared_block = SharedBlock(hidden_dim=hidden_dim, num_filters=num_filters, lstm_units=lstm_units, 
                                        num_blocks=num_blocks, vocabulary_size=vocabulary_size, text_lenght=5, 
                                        in_lstm_units = in_lstm_units, embedding_out = embedding_out)
        self.dense_layer = Dense(num_policies, name='actor_output')

    def call(self, x, z):
        x = self.shared_block(x, z)
        
        return self.dense_layer(x)

class Critic(Layer):

    def __init__(self, hidden_dim=1024, num_filters=128, lstm_units=10, num_blocks=2, vocabulary_size=184, embedding_out = 8,
                 text_lenght=5, in_lstm_units = 32, name="critic", **kwargs):
        super(Critic, self).__init__(name=name, **kwargs)
        self.shared_block = SharedBlock(hidden_dim=hidden_dim, num_filters=num_filters, lstm_units=lstm_units,
                                        num_blocks=num_blocks, vocabulary_size=vocabulary_size, text_lenght=5,
                                        in_lstm_units = in_lstm_units, embedding_out = embedding_out)
        self.dense_layer = Dense(1, name='critic_output')

    def call(self, x, z):
        x = self.shared_block(x, z)

        return self.dense_layer(x)

class ActorCritic(Model):

    def __init__(self, num_policies, hidden_dim=1024, num_filters=128, lstm_units=10, num_blocks=2, 
                 vocabulary_size=184, text_lenght=5, in_lstm_units = 32, embedding_out = 8):
        super(ActorCritic, self).__init__()
        self.actor = Actor(num_policies = num_policies, hidden_dim=hidden_dim, text_lenght=5, 
                             num_filters=num_filters, lstm_units=lstm_units, num_blocks=num_blocks, 
                             in_lstm_units = in_lstm_units, vocabulary_size=vocabulary_size, embedding_out = embedding_out)
        self.critic = Critic(hidden_dim=hidden_dim, num_filters=num_filters, lstm_units=lstm_units, num_blocks=num_blocks, 
                           in_lstm_units = in_lstm_units, text_lenght=5, embedding_out = embedding_out,
                           vocabulary_size=vocabulary_size)
        self.logstd = tf.Variable(np.zeros([1, num_policies]),  dtype=tf.float32 ,name='logstd')

    def call(self, x, z):
        # Critic
        value = self.critic(x, z)

        # Actor
        actor_output = self.actor(x, z)
        std = tf.zeros_like(actor_output) + tf.exp(self.logstd)
        dist = tfp.distributions.Normal(loc=actor_output, scale=std)

        return value, dist    