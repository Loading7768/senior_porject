import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from datetime import datetime

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

coin_name = "TRUMP"

# Load the JSON data (replace 'your_data.json' with your actual file)
with open(f'assets/{coin_name}_price.json', 'r') as f:
    data = json.load(f)

points = data['data']['points']

# Prepare data for pandas DataFrame
dates = []
open_prices = []

for timestamp, values in points.items():
    # Convert Unix timestamp to datetime object
    dt_object = datetime.fromtimestamp(int(timestamp))
    # Format datetime object to YYYY-MM-DD string
    # date_str = dt_object.strftime('%Y-%m-%d')

    dates.append(dt_object)
    open_prices.append(values['v'][0])  # Assuming v[0] is the open price

# Create pandas DataFrame
df = pd.DataFrame({'Date': dates, 'Open Price': open_prices})

# Convert 'Date' column to datetime objects for proper plotting
df['Date'] = pd.to_datetime(df['Date'])

# Plotting
plt.figure(figsize=(12, 6), facecolor=gruvbox['bg'])  # Adjust figure size as needed
plt.plot(df['Date'], df['Open Price'], color=gruvbox['blue'])
# plt.plot(df_daily.index, df_daily.values, color=gruvbox['blue'])

ax = plt.gca()
ax.set_facecolor(gruvbox['bg'])
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.xaxis.label.set_color(gruvbox['yellow'])
ax.yaxis.label.set_color(gruvbox['yellow'])
ax.tick_params(axis='both', colors=gruvbox['yellow'], which='both')

date_format = mdates.DateFormatter('%Y-%m-%d')  # Format for year-month-day
ax.xaxis.set_major_formatter(date_format) 

# hide spines (boarder)
for spine in ax.spines.values():
    spine.set_visible(False)

start_date_str = df['Date'].iloc[0].strftime('%Y-%m-%d')
end_date_str = df['Date'].iloc[-1].strftime('%Y-%m-%d')
ax.set_title(f"{coin_name} / USD ({start_date_str} ~ {end_date_str})", loc='left', color=gruvbox['red'])

plt.grid(visible=True, color=gruvbox['fg'], linestyle='dotted', alpha=0.7)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping

plt.savefig(f"outs/plots/{coin_name}_plot")
plt.show()