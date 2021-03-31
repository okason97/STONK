import numpy as np
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
import random

class EmptyScaler(preprocessing.StandardScaler):    
    def transform(self, X):
        return np.array(X)

class Stocks_env:

    def __init__(self, data, window_size, run_lenght, batch_size=1, random_reset=False,
                 boundary = 0.5, tokenized_industry=[], train_test_ratio=0.2,
                 test_seed = None, initial_money=1000, with_hold=False, normalize=True):
        self.data = data
        self.batch_size = batch_size
        self.run_lenght = run_lenght
        self.window_size = window_size
        self.initial_money = initial_money
        self.money = np.full(self.batch_size, self.initial_money, dtype=float)
        self.boundary = boundary
        self.owned = np.zeros(self.batch_size, dtype=float)
        self.current_symbols = None
        self.current_ending_day = []
        self.tokenized_industry = tokenized_industry
        self.with_hold = with_hold
        self.train_test_ratio = train_test_ratio
        self.random_reset = random_reset
        if len(self.tokenized_industry):
            self.tokenized_industry_state = np.empty((self.batch_size, self.tokenized_industry.shape[1]-1), dtype=float)

        self.operations = np.zeros(self.batch_size, dtype=float)
        self.rewards = np.zeros(self.batch_size, dtype=float)
        self.profits = np.zeros(self.batch_size, dtype=float)
        
        if test_seed:
            self.test_seed = test_seed
        else:
            self.test_seed = None

        if normalize:
            # fit normalizer
            fit_data = self.data.drop(["symbol"], axis=1)
            self.scaler = preprocessing.StandardScaler().fit(fit_data)
        else:
            self.scaler = EmptyScaler()

        unique_symbols = pd.unique(self.data.symbol)
        if self.test_seed:
            r = np.random.RandomState(self.test_seed)
        else:
            r = np.random
        self.test_symbols = r.choice(unique_symbols, int(len(unique_symbols)*self.train_test_ratio), replace=False)
        self.train_symbols = np.setdiff1d(unique_symbols, self.test_symbols)
        
    def get_observation_space(self):
        return (self.window_size, 80)

    def get_action_space(self):
        if self.with_hold:
            return 3
        else:
            return 2

    def get_scaler(self):
        return self.scaler

    def get_current_symbols(self):
        return self.current_symbols

    def get_test_symbols(self):
        return self.test_symbols

    def reset(self, training=True, batch_size=None, run_lenght=None, initial_money=None):
        self.state_index = 0
        if batch_size:
            self.batch_size = batch_size
            self.operations = np.zeros(self.batch_size, dtype=float)
            self.rewards = np.zeros(self.batch_size, dtype=float)
            self.profits = np.zeros(self.batch_size, dtype=float)
        if run_lenght:
            self.run_lenght = run_lenght
        if initial_money:
            self.initial_money = initial_money
        self.money = np.full(self.batch_size, self.initial_money, dtype=float)
        self.owned = np.zeros(self.batch_size, dtype=float)
        self.current_ending_day = []
        self.batch_data = []
        if len(self.tokenized_industry):
            self.tokenized_industry_state = np.empty((self.batch_size, self.tokenized_industry.shape[1]-1), dtype=float)
        if training:
            r = np.random
            sampled_symbols = r.choice(self.train_symbols, self.batch_size, replace=False)
        else:
            if self.test_seed and not self.random_reset:
                r = np.random.RandomState(self.test_seed)
                sampled_symbols = r.choice(self.test_symbols, self.batch_size, replace=False)            
                r = np.random
            else:
                r = np.random
                sampled_symbols = r.choice(self.test_symbols, self.batch_size, replace=False)            
        self.current_symbols = sampled_symbols
        self.current_end_index = []
        if len(self.tokenized_industry):
            current_index = 0
        for symbol in sampled_symbols:
            end_index = r.randint(self.window_size+self.run_lenght+1, len(self.data[self.data.symbol==symbol])+1)
            self.current_ending_day.append(end_index)
            selected_data = self.data[self.data.symbol==symbol][end_index-(self.window_size+self.run_lenght+1):end_index]
            selected_data = selected_data.drop(["symbol"], axis=1)
            self.batch_data.append(selected_data.astype('float32'))

            if len(self.tokenized_industry):
                i = np.argwhere(self.tokenized_industry[:,0]==symbol)[0,0]
                self.tokenized_industry_state[current_index] = self.tokenized_industry[i][1:]
                current_index += 1
                
        state = np.array(list(map(lambda x: self.scaler.transform(x[0:self.window_size]), self.batch_data)))
        if len(self.tokenized_industry):
            state = [state, np.array(self.tokenized_industry_state)]
        return state

    def step(self, action):
        current_day = self.state_index+self.window_size
        
        # buy/sell/hold following short or long strategies

        for i in range(self.batch_size):
            close = float(self.batch_data[i].iloc[[current_day]].close)
            next_day_close = float(self.batch_data[i].iloc[[current_day+1]].close)
            selected_action = tf.math.argmax(action[i])
            if selected_action == 0 and action[i][0]>0:
                # buy
                investment = np.clip(action[i][0],0,1)*self.money[i]
            elif selected_action == 1 and action[i][1]>0:
                # sell
                if self.owned[i]>0:
                    investment = -np.clip(action[i][0],0,1)*self.owned[i]*close
                else:
                # short selling
                    investment = -np.clip(action[i][0],0,1)*self.initial_money
            else:
                # hold
                investment = 0

            self.owned[i] += investment/close
            self.operations[i] = investment
            self.money[i] = self.money[i] - investment

            # calculate reward
            previous_money = self.owned[i]*close + self.money[i]
            next_day_money = self.owned[i]*next_day_close + self.money[i]
            self.profits[i] = next_day_money-previous_money
            self.rewards[i] = self.profits[i]

        # get next state
        self.state_index += 1
        next_state = np.array(list(map(lambda x: self.scaler.transform(x[self.state_index:current_day+1]), self.batch_data)))

        if len(self.tokenized_industry):
            next_state = [next_state, np.array(self.tokenized_industry_state)]

        # finished the run?
        if self.state_index >= self.run_lenght:
            done = True
        else:
            done = False
                    
        return next_state, self.rewards, done, self.operations, np.array(self.current_ending_day)+self.state_index-self.run_lenght, self.profits