import pandas as pd
import json 
import tensorflow as tf
import numpy as np

def load_data(data_folder):
    # read data
    prices = pd.read_csv(data_folder+'854_1575_bundle_archive/prices-split-adjusted.csv')
    fundamentals = pd.read_csv(data_folder+'854_1575_bundle_archive/fundamentals.csv')

    # fix fundamentals and prices
    fundamentals = pd.read_csv(data_folder+'854_1575_bundle_archive/fundamentals.csv')
    fundamentals = fundamentals.drop(columns=fundamentals.columns[0]).dropna()
    fundamentals["Key"] = fundamentals["Ticker Symbol"]+fundamentals["For Year"].astype(int).astype(str)
    fundamentals = fundamentals.drop(columns="Ticker Symbol")
    fundamentals = fundamentals.drop(columns="Period Ending")
    fundamentals = fundamentals.drop(columns="For Year")

    prices["year"] = prices.date.str[:4]
    prices["Key"] = prices.symbol+prices.year

    # join price/fundamentals
    data = prices.merge(fundamentals, on='Key')
    data = data.drop(["date","year","Key"], axis=1)

    return data

def load_data_with_industry(data_folder, column="GICS Sub Industry"):
    data = load_data(data_folder)
    
    details = pd.read_csv(data_folder+'854_1575_bundle_archive/securities.csv')

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(details[column])

    tokenized_industry = tokenizer.texts_to_sequences(details[column])
    tokenized_industry = tf.keras.preprocessing.sequence.pad_sequences(tokenized_industry, padding='post')
    tokenized_industry = np.append(np.array(details["Ticker symbol"]).reshape((505,1)), tokenized_industry, axis=-1)

    vocabulary_size = len(json.loads(tokenizer.get_config()["word_index"]))+1

    return data, tokenized_industry, vocabulary_size