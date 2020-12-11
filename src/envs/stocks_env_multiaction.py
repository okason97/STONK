import numpy as np
from sklearn import preprocessing
import pandas as pd
import random

class Stocks_env:

    def __init__(self, data, batch_size, window_size, run_lenght, 
                 boundary = 0.5, clip=True, tokenized_industry=None, 
                 test_seed = None, initial_money=1000):
        self.data = data
        self.batch_size = batch_size
        self.run_lenght = run_lenght
        self.window_size = window_size
        self.initial_money = initial_money
        self.money = np.full(self.batch_size, self.initial_money, dtype=float)
        self.previous_money = np.zeros(self.batch_size, dtype=float)
        self.boundary = boundary
        self.owned = np.zeros(self.batch_size, dtype=float)
        self.current_symbols = None
        self.current_ending_day = []
        self.clip = clip
        self.tokenized_industry = tokenized_industry
        if len(self.tokenized_industry):
            self.tokenized_industry_state = np.empty((self.batch_size, self.tokenized_industry.shape[1]-1), dtype=float)

        
        if test_seed:
            self.test_seed = test_seed
        else:
            self.test_seed = None

        # fit normalizer
        fit_data = self.data.drop(["symbol"], axis=1)
        self.scaler = preprocessing.StandardScaler().fit(fit_data)

        unique_symbols = pd.unique(self.data.symbol)
        if self.test_seed:
            r = np.random.RandomState(self.test_seed)
        else:
            r = np.random
        self.test_symbols = r.choice(unique_symbols, int(len(unique_symbols)*0.2), replace=False)
        self.train_symbols = np.setdiff1d(unique_symbols, self.test_symbols)
        
        
    def get_observation_space(self):
        return (self.window_size, 80)

    def get_action_space(self):
        return (2)

    def get_scaler(self):
        return self.scaler

    def get_current_symbols(self):
        return self.current_symbols

    def get_test_symbols(self):
        return self.test_symbols

    def reset(self, training=True, batch_size=None):
        self.state_index = 0
        if batch_size:
            self.batch_size = batch_size
        self.money = np.full(self.batch_size, self.initial_money, dtype=float)
        self.previous_money = np.zeros(self.batch_size, dtype=float)
        self.owned = np.zeros(self.batch_size, dtype=float)
        self.current_ending_day = []
        self.batch_data = []
        if len(self.tokenized_industry):
            self.tokenized_industry_state = np.empty((self.batch_size, self.tokenized_industry.shape[1]-1), dtype=float)
        if training:
            r = np.random
            sampled_symbols = r.choice(self.train_symbols, self.batch_size, replace=False)
        else:
            if self.test_seed:
                r = np.random.RandomState(self.test_seed)
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
        daily_change = 0

        operations = np.zeros(self.batch_size, dtype=float)
        profits = np.zeros(self.batch_size, dtype=float)
        
        # buy/sell following short or long strategies
        for i in range(self.batch_size):

            close = float(self.batch_data[i].iloc[[current_day]].close)
            next_day_close = float(self.batch_data[i].iloc[[current_day+1]].close)
            if action[i][0]>=action[i][1]:
                # positive buy/sell policy -> buy
                if self.money[i]<action[i][0]:
                    investment = self.money[i]
                else:
                    investment = action[i][0]
            else:
                # negative buy/sell policy -> sell
                investment = -action[i][1]
            self.owned[i] += investment/close
            operations[i] = investment
            self.money[i] = self.money[i] - investment

            # calculate profit (reward)
            daily_change += abs(next_day_close-close)/close
            self.previous_money[i] = self.owned[i]*close + self.money[i]
            next_day_money = self.owned[i]*next_day_close + self.money[i]
            profits[i] = next_day_money-self.previous_money[i]

        daily_change = daily_change/self.batch_size

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

        return next_state, profits, done, operations, np.array(self.current_ending_day)+self.state_index-self.run_lenght, daily_change