import sys
if not '../' in sys.path:
    sys.path.append('../')

import tensorflow as tf

import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
import time

from execution.train import Stocks_env
from datasets import nyse
from models.lstm_selfattention_embedding import ActorCritic

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
print(tf.config.experimental.list_logical_devices('GPU'))
tf.test.is_gpu_available()

data, tokenized_industry, vocabulary_size = nyse.load_data_with_industry('../data/', column='GICS Sector')

lr=1e-4
rs=1
run_lenght=10
batch_size=256
window_size=(5,10)
ppo_epochs=2
num_epochs=400
test_iterations=5
hidden_dim=(128,512)
kernel_size=7
num_filters=(64,128)
lstm_units=1024
num_blocks=(1,2)
embedding_out=6
in_lstm_units=16
initial_money=100
test_seed=None
model_filename=None
with_hold=False
actor_activation=False
log_freq=10
save_freq=40

for ws in window_size:
    for hd in hidden_dim:
        for nf in num_filters:
            for nb in num_blocks:
                date = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
                identifier = "stonks-ws{}-hd{}-nf{}-nb{}".format(ws,hd,nf,nb)
                print("Training "+identifier)
                train_model(lr=lr, rs=rs, run_lenght=run_lenght, batch_size=batch_size, window_size=ws, ppo_epochs=ppo_epochs,
                            num_epochs=num_epochs, test_iterations=test_iterations, hidden_dim=hidden_dim,
                            kernel_size=kernel_size, num_filters=nf, lstm_units=lstm_units, num_blocks=nb,
                            embedding_out=embedding_out, in_lstm_units=in_lstm_units, initial_money=initial_money,
                            test_seed=test_seed, model_filename=model_filename, with_hold=with_hold,
                            actor_activation=actor_activation, log_freq=log_freq, save_freq=save_freq, data=data,
                            tokenized_industry=tokenized_industry, vocabulary_size=vocabulary_size, identifier=identifier)