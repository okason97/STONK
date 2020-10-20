import pandas as pd

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