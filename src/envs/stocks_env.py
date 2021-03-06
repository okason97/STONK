import numpy as np
from sklearn import preprocessing
import pandas as pd
import random

class Stocks_env:

    def __init__(self, data, batch_size, window_size, run_lenght, 
                 boundary = 0.5, clip=True, tokenized_industry=None, 
                 test_seed = None):
        self.data = data
        self.batch_size = batch_size
        self.run_lenght = run_lenght
        self.window_size = window_size
        self.money = np.zeros(self.batch_size)
        self.previous_money = np.zeros(self.batch_size)
        self.boundary = boundary
        self.owned = np.zeros(self.batch_size)
        self.current_symbols = None
        self.current_ending_day = []
        self.clip = clip
        self.tokenized_industry = tokenized_industry
        self.profits = None
        self.operations = None
        if len(self.tokenized_industry):
            self.tokenized_industry_state = np.empty((self.batch_size, self.tokenized_industry.shape[1]-1))

        
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
        return (1)

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
        self.profits = np.zeros(self.batch_size)
        self.operations = np.zeros(self.batch_size)
        self.money = np.zeros(self.batch_size)
        self.previous_money = np.zeros(self.batch_size)
        self.owned = np.zeros(self.batch_size)
        self.current_ending_day = []
        self.batch_data = []
        if len(self.tokenized_industry):
            self.tokenized_industry_state = np.empty((self.batch_size, self.tokenized_industry.shape[1]-1))
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

        # get next state
        next_state = np.array(list(map(lambda x: self.scaler.transform(x[self.state_index:current_day]), self.batch_data)))

        # buy/sell following short or long strategies
        total_investment = 0
        for i in range(self.batch_size):

            if self.clip:
                investment = abs(np.clip(np.array(action[i]),-(1+self.boundary),1+self.boundary))-self.boundary
            else:
                investment = abs(action[i])-self.boundary
            close = float(self.batch_data[i].iloc[[current_day]].close)
            next_day_close = float(self.batch_data[i].iloc[[current_day+1]].close)
            traded = investment/close
            if action[i]>self.boundary:
                # positive buy/sell policy -> buy
                total_investment += investment
                self.owned[i] += traded
                self.operations[i] = investment
                self.money[i] = self.money[i] - investment
            elif action[i]<-self.boundary:
                # negative buy/sell policy -> sell
                # make traded positive for later usage on the reward calculation
                total_investment += investment
                self.owned[i] -= traded
                self.operations[i] = -investment
                self.money[i] = self.money[i] + investment
            else:
                self.operations[i] = 0                          

            daily_change += abs(1/close*next_day_close - 1)
            # calculate profit (reward)
            self.previous_money[i] = self.owned[i]*close + self.money[i]
            next_day_money = self.owned[i]*next_day_close + self.money[i]
            self.profits[i] = next_day_money-self.previous_money[i]

        daily_change = daily_change/self.batch_size
        reward = np.sum(self.profits)/self.batch_size

        self.state_index += 1

        # finished the run?
        if self.state_index >= self.run_lenght:
            done = True
        else:
            done = False

        if len(self.tokenized_industry):
            next_state = [next_state, np.array(self.tokenized_industry_state)]

        return next_state, reward, done, self.operations, np.array(self.current_ending_day)+self.state_index-self.run_lenght+1, daily_change