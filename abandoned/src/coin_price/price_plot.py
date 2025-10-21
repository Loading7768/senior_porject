import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

gruvbox = {
    'bg': '#282828',
    'fg': '#ebdbb2',
    'red': '#cc241d',
    'green': '#98971a',
    'yellow': '#d79921',
    'blue': '#458588',
    'purple': '#b16286',
    'aqua': '#689d6a',
}

def plot(coin_name):
    file_path = f'outs/csvs/{coin_name}_price.csv'
    dataframe = pd.read_csv(file_path, parse_dates=["timestamp"])
    window_size = 24 * 60
    smoothed_data = dataframe['close'].rolling(window=window_size, center=True).mean()

    # # Calculate slope
    # extra_smooth_data = dataframe['close'].rolling(window=window_size * 7, center=True).mean()
    # x = np.arange(len(extra_smooth_data))
    # y = extra_smooth_data.values
    # slopes = np.diff(y) / np.diff(x)

    # # Mark drastic changes
    # threshold = np.percentile(np.abs(slopes), 80)
    # drastic_indices = np.where(np.abs(slopes) > threshold)[0]

    # Set timestamp as index
    dataframe.set_index("timestamp", inplace=True)

    # Plot closing prices over time
    plt.figure(figsize=(12, 6), facecolor=gruvbox['bg'])
    plt.plot(dataframe.index, smoothed_data, label="Closing Price", color=gruvbox['blue'])
    # for idx in drastic_indices:
    #     plt.axvspan(smoothed_data.index[idx], smoothed_data.index[idx + 1], color=gruvbox['purple'])

    ax = plt.gca()
    ax.set_facecolor(gruvbox['bg'])

    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.xaxis.label.set_color(gruvbox['yellow'])
    ax.yaxis.label.set_color(gruvbox['yellow'])

    ax.tick_params(axis='both', colors=gruvbox['yellow'], which='both')

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # hide spines (boarder)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"{coin_name} / USD ({dataframe.index[0].strftime("%Y-%m-%d")} ~ {dataframe.index[-1].strftime("%Y-%m-%d")})", loc='left', color=gruvbox['red'])

    plt.grid(visible=True, color=gruvbox['fg'], linestyle='dotted', alpha=0.7)

    plt.savefig(f"outs/plots/{coin_name}_plot")
    plt.show(block=True)

plot("TRUMP")