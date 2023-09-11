import os
import configparser
import ccxt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from binance.client import Client

def get_active_futures_symbols(client):
    futures_symbols = []
    exchange_info = client.futures_exchange_info()
    ignored_pairs = ["ETHUSDT_230929", "ETHUSDT_231229", "BTCUSDT_230929", "BTCUSDT_231229"]

    for s in exchange_info['symbols']:
        if (
            s['status'] == 'TRADING' and
            s['quoteAsset'] == 'USDT' and
            s['symbol'] not in ignored_pairs
        ):
            futures_symbols.append(s['symbol'])
    return futures_symbols

config = configparser.ConfigParser()
config.read('binanceapi.ini')

apikey = config['DEFAULT']['API_KEY']
apisecret = config['DEFAULT']['API_SECRET']

# Create a Binance client instance
client = Client(apikey, apisecret)
# Create a Binance client instance
exchange = ccxt.binance({
    'rateLimit': 20,  # Adjust the rate limit as needed
    'apiKey': apikey,
    'secret': apisecret,
})

active_futures_symbols = get_active_futures_symbols(client)

# Set the figure size and resolution
fig_width = 1920 / 100
fig_height = 1080 / 100
dpi = 500  # Change this value for higher resolution

# Define the time intervals for the klines
time_intervals = ['5m', '15m', '30m', '1h', '4h', '6h','8h','12h','1d','3d','1w']

# Loop through each USDT trading pair and time interval, and plot the data
for symbol in tqdm(active_futures_symbols, desc="Symbols"):
    for interval in tqdm(time_intervals, desc="Intervals", leave=False):
        klines = exchange.fetch_ohlcv(symbol, interval)

        high_prices = [entry[2] for entry in klines]
        low_prices = [entry[3] for entry in klines]

        period_length = 20 * 24
        hh = max(high_prices[:period_length])
        ll = min(low_prices[:period_length])

        fib_levels = [-0.886, -0.786, -0.75, -0.706, -0.618, -0.382, -0.236, 0, 0.14, 0.236, 0.382, 0.5, 0.618, 0.706, 0.75, 0.79, 0.88, 1]
        fan_levels = [1.272, 1.618, 2]
        all_levels = fib_levels + fan_levels

        # Calculate linear regression channel
        x = np.arange(len(klines))
        y = (np.array(high_prices) + np.array(low_prices)) / 2  # Mean of high and low prices
        regression_coeffs = np.polyfit(x, y, 1)
        y_regression = np.polyval(regression_coeffs, x)

        channel_width = np.std(y - y_regression)
        upper_channel = y_regression + channel_width
        lower_channel = y_regression - channel_width

        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        ax = plt.gca()

        ax.plot(high_prices, label="High Prices", color="green")
        ax.plot(low_prices, label="Low Prices", color="red")
        ax.fill_between(range(len(klines)), high_prices, low_prices, color="lightgray", alpha=0.5)

        for level in all_levels:
            price = hh - (hh - ll) * level
            fib_line = ax.axhline(y=price, color="blue", linestyle="--")
            ax.annotate(f"Fib {level:.3f}", xy=(0, price), xytext=(5, 0), textcoords="offset points", color="blue", va="center", bbox=dict(facecolor='white', edgecolor='none', pad=1), fontsize=8)
            ax.annotate(f"{price:.4f}", xy=(40, price), xytext=(15, 0), textcoords="offset points", color="blue", va="center", fontsize=8, bbox=dict(facecolor='white', edgecolor='none', pad=1))

        ax.plot(upper_channel, label="Upper Channel", color="purple", linestyle="--")
        ax.plot(lower_channel, label="Lower Channel", color="purple", linestyle="--")

        # Calculate and display the last value of the linear regression channel
        last_upper_channel = upper_channel[-1]
        last_lower_channel = lower_channel[-1]

        ax.text(len(klines) - 1, last_upper_channel, f"Upper: {last_upper_channel:.4f}", ha="right", va="center", color="purple")
        ax.text(len(klines) - 1, last_lower_channel, f"Lower: {last_lower_channel:.4f}", ha="right", va="center", color="purple")

        ax.set_title(f"Price Chart with Fibonacci Levels and Linear Regression Channel ({symbol} - {interval})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

        # Ensure the coin_images directory exists
        if not os.path.exists("coin_images"):
            os.makedirs("coin_images")

        image_path = os.path.join("coin_images", f"{symbol.replace('/', '_')}_{interval}.png")
        plt.savefig(image_path, bbox_inches='tight')

        plt.close()

print("Images saved in 'coin_images' directory.")
