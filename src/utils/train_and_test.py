import sys
if not '../' in sys.path:
    sys.path.append('../')

import numpy as np
import tensorflow as tf
import io
from IPython.display import clear_output
import matplotlib.pyplot as plt

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states_daily, states_industry, actions, log_probs, returns, advantage):
    batch_size = tf.shape(states_daily)[0]
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states_daily.numpy()[rand_ids, :], np.array(states_industry)[rand_ids, :], actions.numpy()[rand_ids, :], log_probs.numpy()[rand_ids, :], returns.numpy()[rand_ids, :], advantage.numpy()[rand_ids, :]

def ppo_update(ppo_epochs, mini_batch_size, states_daily, states_industry, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state_daily, state_industry, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states_daily, states_industry, actions, log_probs, returns, advantages):
            with tf.GradientTape() as tape:
                # value, dist = model(state, np.reshape(state[:,-1,:],(batch_size, 1, num_inputs[1])))
                value, dist = model(state_daily, state_industry)

                entropy = tf.math.reduce_mean(dist.entropy())
                new_log_probs = dist.log_prob(action)

                ratio = tf.math.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, clip_value_min=1.0 - clip_param, 
                                         clip_value_max=1.0 + clip_param) * advantage

                actor_loss  = - tf.math.reduce_mean(tf.math.minimum(surr1, surr2))
                critic_loss = tf.math.reduce_mean(tf.math.pow(return_ - value, 2))

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
def plot(epochs, rewards_train, rewards_test):
    clear_output(True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
    fig.suptitle('frame %s. ' % epochs[-1])
    ax1.set_title('train. reward: %s' % rewards_train[-1])
    ax1.plot(epochs, rewards_train)
    ax2.set_title('test. reward: %s' % rewards_test[-1])
    ax2.plot(epochs, rewards_test)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=600)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    plt.show()
    return image

def test_env(batch_size, run_lenght, initial_money, record_days=False, random_reset=True):
    state = env.reset(training=False, batch_size=batch_size, run_lenght=run_lenght, 
                      initial_money=initial_money, random_reset=random_reset, mean_test_reward=None)
    done = False
    operation_array = []
    days_array = []
    rewards_array = []
    total_profit = np.zeros(batch_size)
    while not done:
        _, dist = model(state[0], state[1])
        next_state, reward, done, operations, day, profit = env.step(dist.sample())
        state = next_state
        if record_days:
            operation_array.append(np.array(operations))
            days_array.append(np.array(day))
            rewards_array.append(np.array(reward))
        mean_test_reward(np.array(reward))
        total_profit += profit
    total_profit = total_profit/test_initial_money
    return operation_array, days_array, rewards_array, total_profit