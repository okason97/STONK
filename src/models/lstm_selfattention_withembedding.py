import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Layer, Dense, Conv1D, BatchNormalization, Dropout, LayerNormalization, LSTM, Embedding
import numpy as np

def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class AttentionBlock(Layer):

    def __init__(self, hidden_dim=1024, num_filters=128,
                 name="attention block", rate=0.1, residual=True, last=False, **kwargs):
        super(AttentionBlock, self).__init__(name=name, **kwargs)
        self.attention_layer = MultiHeadAttention(d_model=hidden_dim, num_heads=8)
        self.dropout1 = Dropout(rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.cnn_layer = Conv1D(
            filters=num_filters,
            kernel_size=7,
            # Use 'same' padding so outputs have the same shape as inputs.
            padding='same')
        self.activation_layer = Activation('relu')
        self.dense1 = Dense(hidden_dim/2)
        self.dense2 = Dense(hidden_dim)
        self.dropout2 = Dropout(rate)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.residual = residual
        self.last = last

    def call(self, x, training, mask=None):

        cl_output = self.activation_layer(self.cnn_layer(x))
        attn_output, _ = self.attention_layer(cl_output,cl_output,cl_output,mask)
        attn_output = self.dropout1(attn_output, training=training)
        if self.residual:
            x = self.layernorm1(x + attn_output)
        else:
            x = self.layernorm1(attn_output)

        if not self.last:
            ff_output = self.dense1(x)
            ff_output = self.dense2(ff_output)
            ff_output = self.dropout2(ff_output, training=training)
            x = self.layernorm2(x + ff_output)
                
        return x

class SharedBlock(Layer):

    def __init__(self, hidden_dim=1024, num_filters=128, lstm_units=10, num_blocks=2, unique_words=124, embedding_out = 8,
                 name="shared block", **kwargs):
        super(SharedBlock, self).__init__(name=name, **kwargs)
        self.embedding = Embedding(unique_words, embedding_out)
        self.attention_blocks = [AttentionBlock(hidden_dim=hidden_dim, num_filters=num_filters, residual=i!=0,
                                                last=i==num_blocks-1) 
                                 for i in range(num_blocks)]
        self.lstm_layer = LSTM(lstm_units)

    def call(self, x, training, mask=None):
        
        for attention_block in self.attention_blocks:
            x = attention_block(x)
        
        return self.lstm_layer(x)

class Critic(Layer):

    def __init__(self, num_policies, hidden_dim=1024, num_filters=128, lstm_units=10, num_blocks=2,
                 name="critic", **kwargs):
        super(Critic, self).__init__(name=name, **kwargs)
        self.shared_block = SharedBlock(hidden_dim=hidden_dim, num_filters=num_filters, lstm_units=lstm_units, 
                                        num_blocks=num_blocks)
        self.dense_layer = Dense(num_policies, name='critic_output')

    def call(self, x):
        x = self.shared_block(x)
        
        return self.dense_layer(x)

class Actor(Layer):

    def __init__(self, hidden_dim=1024, num_filters=128, lstm_units=10, num_blocks=2,
                 name="actor", **kwargs):
        super(Actor, self).__init__(name=name, **kwargs)
        self.shared_block = SharedBlock(hidden_dim=hidden_dim, num_filters=num_filters, lstm_units=lstm_units,
                                        num_blocks=num_blocks)
        self.dense_layer = Dense(1, name='actor_output')

    def call(self, x):
        x = self.shared_block(x)

        return self.dense_layer(x)

class ActorCritic(Model):

    def __init__(self, num_policies, hidden_dim=1024, num_filters=128, lstm_units=10, num_blocks=2):
        super(ActorCritic, self).__init__()
        self.critic = Critic(num_policies = num_policies, hidden_dim=hidden_dim, 
                             num_filters=num_filters, lstm_units=lstm_units, num_blocks=num_blocks)
        self.actor = Actor(hidden_dim=hidden_dim, num_filters=num_filters, lstm_units=lstm_units, num_blocks=num_blocks)
        self.logstd = tf.Variable(np.zeros([1, num_policies]),  dtype=tf.float32 ,name='logstd')

    def call(self, x):
        # Actor
        value = self.actor(x)

        # Critic
        critic_output = self.critic(x)
        std = tf.zeros_like(critic_output) + tf.exp(self.logstd)
        dist = tfp.distributions.Normal(loc=critic_output, scale=std)

        return value, dist    