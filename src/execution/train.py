import sys
if not '../' in sys.path:
    sys.path.append('../')
    
import tensorflow as tf

import pandas as pd
import numpy as np
import os
from datetime import datetime
import time

from envs.stocks_env_multiaction import Stocks_env
from utils.train_and_test import compute_gae, ppo_iter, ppo_update, plot, test_env
from models.lstm_selfattention_embedding import ActorCritic
            
    
def train_model(lr=1e-4, rs=1, run_lenght=10, batch_size=256, window_size=5, ppo_epochs=4, num_epochs=400,
                test_iterations=5, hidden_dim=256, kernel_size=7, num_filters=128, lstm_units=512, num_blocks=1,
                embedding_out=6, in_lstm_units=16, initial_money=100, test_seed=None, model_filename=None,
                with_hold=False, actor_activation=False, log_freq=10, save_freq=40, data=None,
                tokenized_industry=None, vocabulary_size=None, identifier=None):

    # log
    models_directory = 'results/models/'
    save_directory = 'results/saved-timesteps/'
    if identifier is None:
        date = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
        identifier = "stonks-" + date
    test_summary_writer = tf.summary.create_file_writer('results/summaries/test/' + identifier)
    mean_test_reward = tf.keras.metrics.Mean(name='mean_test_reward')
    mean_train_reward = tf.keras.metrics.Mean(name='mean_train_reward')


    # initialize env
    env = Stocks_env(data, window_size, run_lenght, batch_size=batch_size, with_hold=with_hold,
                     tokenized_industry=tokenized_industry, test_seed=test_seed, initial_money=initial_money)

    # test parameters
    test_batch_size  = len(env.get_test_symbols())
    unique_symbols = data.symbol.unique()
    mini = 999
    for symbol in unique_symbols:
        if len(data[data.symbol==symbol]) < mini:
            mini = len(data[data.symbol==symbol])
    test_run_lenght = mini-window_size-1
    test_initial_money = initial_money

    # inputs and policies
    num_inputs  = env.get_observation_space()
    num_policies = env.get_action_space()

    # initialize the model
    model = ActorCritic(num_policies = num_policies, hidden_dim=hidden_dim, num_filters=num_filters,
                        actor_activation=actor_activation, lstm_units=lstm_units, text_lenght=tokenized_industry.shape[1],
                        kernel_size=kernel_size, vocabulary_size=vocabulary_size, embedding_out=embedding_out, 
                        in_lstm_units = in_lstm_units)
    optimizer = tf.keras.optimizers.Adam(lr)
    if model_filename:
        state = env.reset()
        model(state[0], state[1])
        model.load_weights(models_directory + model_filename + '.h5')

    train_rewards = []
    test_rewards = []
    total_profits = []
    test_total_profits = []
    best_test = 0
    epoch  = 0
    epochs = []

    start = time.time()
    while epoch < num_epochs:

        log_probs = []
        values    = []
        states_daily = []
        states_industry = []
        actions   = []
        rewards   = []
        masks     = []
        total_profit = np.zeros(batch_size)

        state = env.reset(training=True, batch_size=batch_size, run_lenght=run_lenght, initial_money=initial_money)

        done = False
        while not done:
            value, dist = model(state[0], state[1])

            action = dist.sample()
            next_state, reward, done, _, _, profit = env.step(action)
            mean_train_reward(reward)
            reward = reward*rs

            log_prob = dist.log_prob(action)

            log_probs.append(log_prob)
            values.append(np.array(value))
            rewards.append(tf.dtypes.cast(tf.expand_dims((reward,), axis=1), tf.float32))
            masks.append(tf.expand_dims((float(1 - done),), axis=1))

            total_profit += profit

            states_daily.append(state[0])
            states_industry.append(state[1])
            actions.append(action)

            state = next_state

        # next_value, _ = model(next_state, np.reshape(next_state[:,-1,:],(batch_size, 1, num_inputs[1])))
        next_value, _ = model(next_state[0], next_state[1])
        returns = compute_gae(next_value, rewards, masks, values)

        returns_save = returns

        returns   = tf.reshape(tf.concat(returns, axis=1),(run_lenght*batch_size,-1))
        log_probs = tf.concat(log_probs, axis=0)
        values    = tf.concat(values, axis=0)
        states_daily    = tf.concat(states_daily, axis=0)    
        states_industry = tf.reshape(states_industry,(run_lenght*batch_size,-1))
        actions   = tf.reshape(tf.concat(actions, axis=0),(run_lenght*batch_size,-1))

        advantage = returns - values
        advantage = tf.reshape(advantage,(run_lenght*batch_size**2,1))

        ppo_update(ppo_epochs, batch_size, states_daily, states_industry, actions, log_probs, returns, advantage)

        if epoch % log_freq == 0:
            for i in range(test_iterations):
                operation_array, days_array, rewards_array, test_total_profit = test_env(test_batch_size, test_run_lenght, initial_money, record_days=(i==test_iterations-1), random_reset=False, run_lenght=test_run_lenght ,mean_test_reward=mean_test_reward)
            total_profits.append(total_profit/initial_money)
            test_total_profits.append(test_total_profit)
            train_rewards.append(mean_train_reward.result().numpy())
            test_rewards.append(mean_test_reward.result().numpy())
            epochs.append(epoch)
            with test_summary_writer.as_default():
                tf.summary.scalar('mean_test_reward', mean_test_reward.result(), step=epoch)
                tf.summary.scalar('mean_train_reward', mean_train_reward.result(), step=epoch)
                tf.summary.image('Plot', plot(epochs, train_rewards, test_rewards), step=epoch)
            # serialize weights to HDF5
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            if not os.path.exists(save_directory+'/operation/'):
                os.makedirs(save_directory+'/operation/')
            if not os.path.exists(save_directory+'/endingday/'):
                os.makedirs(save_directory+'/endingday/')
            if not os.path.exists(save_directory+'/rewards/'):
                os.makedirs(save_directory+'/rewards/')
            if not os.path.exists(save_directory+'/profits/'):
                os.makedirs(save_directory+'/profits/')
            if not os.path.exists(save_directory+'/test-profits/'):
                os.makedirs(save_directory+'/test-profits/')
            pd.DataFrame(operation_array).to_csv(save_directory+"/operation/{}-epoch{}.csv".format(identifier, epoch), 
                                                 header=env.get_current_symbols(), index=None)
            pd.DataFrame(days_array).to_csv(save_directory+"/endingday/{}-epoch{}.csv".format(identifier, epoch), 
                                                 header=env.get_current_symbols(), index=None)
            pd.DataFrame(rewards_array).to_csv(save_directory+"/rewards/{}-epoch{}.csv".format(identifier, epoch), 
                                                 header=env.get_current_symbols(), index=None)
            pd.DataFrame(total_profits).to_csv(save_directory+"/profits/{}.csv".format(identifier),
                                               index=None)
            pd.DataFrame(test_total_profits).to_csv(save_directory+"/test-profits/{}.csv".format(identifier),
                                                    index=None)

            if epoch % save_freq == 0:
                if not os.path.exists(models_directory):
                    os.makedirs(models_directory)
                if best_test<mean_test_reward.result():
                    model.save_weights(models_directory + "best-{}.h5".format(identifier))
                model.save_weights(models_directory + "model-{}.h5".format(identifier))

            mean_test_reward.reset_states()
            end = time.time()
            print(end - start)
            start = time.time()

        mean_train_reward.reset_states()
        epoch += 1