'plotting figure_1'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read the data
cape_ratio = pd.read_csv("cape_ratio.csv")
tya_real_return = pd.read_csv("tya_real_return.csv")

#convert to datetime (should already be in right format however it works better this way)
cape_ratio['Date'] = pd.to_datetime(cape_ratio['Date'])
tya_real_return['Date'] = pd.to_datetime(tya_real_return['Date'])

#values for the return (for better readability)
cape_values = cape_ratio['cape_ratio'].values
dates = cape_ratio['Date'].values

#defining the regression coefficients
intercept = 0.270
slope = -0.177

#Calculate forecasted returns using the regression equation
#29.16 is the starting value I used

forecasted_returns = (intercept + slope * np.log(cape_values)) * 100 + 29.16

#Create figure and primary axis
fig, ax1 = plt.subplots()

#Plot ten year annulized real return on primary axis as well as the forecast
ax1.plot(dates, tya_real_return['tya_real_return'], color='#00008B', label='Real Returns')
ax1.plot(dates, forecasted_returns, color='grey', linestyle='dotted', label='Forecast returns')
ax1.set_xlabel('Date')
ax1.set_ylabel('Real Returns', color='#00008B')

ax1.set_ylim(-40, 40)

#Create secondary axis
ax2 = ax1.twinx()

#Plot cape ratio on secondary axis
ax2.plot(dates, cape_values, color='black', label='CAPE Ratio')
ax2.set_ylabel('CAPE Ratio', color='black')

#Show legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

ax2.set_ylim(0, 150)

# Show plot
plt.show()
