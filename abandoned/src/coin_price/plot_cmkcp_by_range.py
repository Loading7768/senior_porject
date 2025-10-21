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

coin_name = "DOGE"

# Load the JSON data
with open(f'assets/{coin_name}_price.json', 'r') as f:
    data = json.load(f)

points = data['data']['points']

# Prepare data for pandas DataFrame
dates = []
open_prices = []

for timestamp, values in points.items():
    dt_object = datetime.fromtimestamp(int(timestamp))
    dates.append(dt_object)
    open_prices.append(values['v'][0])  # Assuming v[0] is the open price

# Create pandas DataFrame
df = pd.DataFrame({'Date': dates, 'Open Price': open_prices})

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Get user input for start and end dates
print(f"Available date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
start_date_input = input("Enter start date (YYYY-MM-DD): ")
end_date_input = input("Enter end date (YYYY-MM-DD): ")

# Convert input strings to datetime
try:
    start_date = pd.to_datetime(start_date_input)
    end_date = pd.to_datetime(end_date_input)
except ValueError:
    print("Invalid date format. Please use YYYY-MM-DD.")
    exit()

# Validate date range
if start_date < df['Date'].min() or end_date > df['Date'].max():
    print("Selected dates are outside the available data range.")
    exit()
if start_date > end_date:
    print("Start date cannot be after end date.")
    exit()

# Filter DataFrame based on date range
df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Check if filtered DataFrame is empty
if df_filtered.empty:
    print("No data available for the selected date range.")
    exit()

# Plotting
plt.figure(figsize=(12, 6), facecolor=gruvbox['bg'])
plt.plot(df_filtered['Date'], df_filtered['Open Price'], color=gruvbox['blue'])

ax = plt.gca()
ax.set_facecolor(gruvbox['bg'])
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.xaxis.label.set_color(gruvbox['yellow'])
ax.yaxis.label.set_color(gruvbox['yellow'])
ax.tick_params(axis='both', colors=gruvbox['yellow'], which='both')

date_format = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_format)

# Hide spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Use user-selected dates for title
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')
ax.set_title(f"{coin_name} / USD ({start_date_str} ~ {end_date_str})", loc='left', color=gruvbox['red'])

plt.grid(visible=True, color=gruvbox['fg'], linestyle='dotted', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(f"outs/plots/{coin_name}_plot_{start_date_str}_{end_date_str})")
plt.show()