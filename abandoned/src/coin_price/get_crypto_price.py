# Documentation: https://alpaca.markets/sdks/python/api_reference/data/crypto.html

# import and initialization
from alpaca.data.historical import CryptoHistoricalDataClient
crypto_client = CryptoHistoricalDataClient()
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

import price_plot

# input
coin_name = input('Coin Name > ')

# get date
date = datetime.today().strftime('%Y-%m-%d')

# set request parameters
# TimeFrame could be set to Day or Minute
request_params = CryptoBarsRequest(
    symbol_or_symbols=f"{coin_name}/USD",
    timeframe=TimeFrame.Minute,
    start=datetime(2013, 12, 16),
    end=datetime(int(date[0:4]), int(date[5:7]), int(date[8:10]))
)

# fetch data
crypto_bars = crypto_client.get_crypto_bars(request_params)
crypto_bars.df.to_csv(f'outs/csvs/{coin_name}_price.csv', index=True)

price_plot.plot(coin_name)