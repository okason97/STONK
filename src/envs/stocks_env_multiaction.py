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
        self.grouped_data = {}
        self.standarized_grouped_data = {}
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

        unique_symbols = pd.unique(self.data.symbol)

        if normalize:
            # fit normalizer
            self.data = self.data.drop(["symbol"], axis=1)
            self.scaler = preprocessing.StandardScaler().fit(self.data)
        else:
            self.scaler = EmptyScaler()

        for symbol in unique_symbols:
            self.grouped_data[symbol] = data[data.symbol==symbol].drop(["symbol"], axis=1)
            self.standarized_grouped_data[symbol] = self.scaler.transform(self.grouped_data[symbol])
         
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

    def reset(self, training=True, batch_size=None, run_lenght=None, initial_money=None, random_reset=True):
        self.state_index = 0
        if random_reset:
            self.random_reset = random_reset
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
        self.standarized_batch_data = np.zeros((self.batch_size,self.window_size+self.run_lenght+1,80), dtype=np.float32)
        self.batch_data = np.zeros((self.batch_size,self.window_size+self.run_lenght+1,80))
        if len(self.tokenized_industry):
            self.tokenized_industry_state = np.empty((self.batch_size, self.tokenized_industry.shape[1]-1), dtype=float)
        if training:
            r = np.random
            sampled_symbols = r.choice(self.train_symbols, self.batch_size, replace=False)
            r = np.random
        else:
            if self.test_seed and not self.random_reset:
                r = np.random.RandomState(self.test_seed)
                sampled_symbols = r.choice(self.test_symbols, self.batch_size, replace=False)            
            else:
                r = np.random
                sampled_symbols = r.choice(self.test_symbols, self.batch_size, replace=False)            
        self.current_symbols = sampled_symbols
        self.current_end_index = []
        current_index = 0
        for symbol in sampled_symbols:
            end_index = r.randint(self.window_size+self.run_lenght+1, len(self.grouped_data[symbol])+1)
            self.current_ending_day.append(end_index)
            self.batch_data[current_index] = self.grouped_data[symbol][end_index-(self.window_size+self.run_lenght+1):end_index]
            self.standarized_batch_data[current_index] = self.standarized_grouped_data[symbol][end_index-(self.window_size+self.run_lenght+1):end_index]

            if len(self.tokenized_industry):
                i = np.argwhere(self.tokenized_industry[:,0]==symbol)[0,0]
                self.tokenized_industry_state[current_index] = self.tokenized_industry[i][1:]
            current_index += 1
                
        state = self.standarized_batch_data[:,0:self.window_size]
        if len(self.tokenized_industry):
            state = [state, np.array(self.tokenized_industry_state)]
        return state

    def step(self, action):
        current_day = self.state_index+self.window_size
        
        # buy/sell/hold following short or long strategies

        for i in range(self.batch_size):
            close = self.batch_data[i,current_day,self.data.columns.get_loc("close")]
            next_day_close = self.batch_data[i,current_day+1,self.data.columns.get_loc("close")]
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
            self.rewards[i] = next_day_money-previous_money
            self.profits[i] = self.rewards[i]

        # get next state
        self.state_index += 1
        next_state = self.standarized_batch_data[:,self.state_index:current_day+1]

        if len(self.tokenized_industry):
            next_state = [next_state, np.array(self.tokenized_industry_state)]

        # finished the run?
        if self.state_index >= self.run_lenght:
            done = True
        else:
            done = False
                    
        return next_state, self.rewards, done, self.operations, np.array(self.current_ending_day)+self.state_index-self.run_lenght, self.profits