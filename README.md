### Developed By:   SABARI S
### Register NO: 212222240085
### Date: 

# Ex.No: 07                                       AUTO REGRESSIVE MODEL



### AIM:
To Implementat an Auto Regressive Model using Python for Air quality Dataset.

### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.

   
### PROGRAM
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'data_date.csv'  # Replace with the actual file path
data = pd.read_csv(file_path)

# Filter data for 'India' (you can change this to any other country)
country_data = data[data['Country'] == 'India'].copy()

# Convert 'Date' column to datetime format and set it as the index
country_data['Date'] = pd.to_datetime(country_data['Date'], infer_datetime_format=True)
country_data.set_index('Date', inplace=True)

# Assume 'AQI Value' column represents the data to analyze
aqi_values = country_data['AQI Value']

# Resample to weekly data by taking the mean
weekly_aqi_values = aqi_values.resample('W').mean()

# Perform Augmented Dickey-Fuller test for stationarity
result = adfuller(weekly_aqi_values.dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] < 0.05:
    print("The data is stationary.")
else:
    print("The data is non-stationary.")

# Split data into training and test sets (80% training, 20% testing)
train_size = int(len(weekly_aqi_values) * 0.8)
train, test = weekly_aqi_values[:train_size], weekly_aqi_values[train_size:]

# Plot ACF and PACF
fig, ax = plt.subplots(2, figsize=(8, 6))
plot_acf(train.dropna(), ax=ax[0], title='Autocorrelation Function (ACF)')
plot_pacf(train.dropna(), ax=ax[1], title='Partial Autocorrelation Function (PACF)')
plt.show()

# Fit AutoRegressive (AR) model with 13 lags
ar_model = AutoReg(train.dropna(), lags=13).fit()

# Make predictions on the test set
ar_pred = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Plot the predictions against the actual test data
plt.figure(figsize=(10, 4))
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('AR Model Prediction vs Test Data')
plt.xlabel('Time')
plt.ylabel('AQI Value')
plt.legend()
plt.show()

# Calculate Mean Squared Error
mse = mean_squared_error(test, ar_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Plot the full data: Train, Test, and Predictions
plt.figure(figsize=(10, 4))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('Train, Test, and AR Model Prediction')
plt.xlabel('Time')
plt.ylabel('AQI Value')
plt.legend()
plt.show()

```
### OUTPUT:

GIVEN DATA
![image](https://github.com/user-attachments/assets/c35095af-f392-4fd2-afca-cd202438ad78)

PACF - ACF
![image](https://github.com/user-attachments/assets/53dc58ee-97e6-425c-be94-1b20379c8ec0)
![image](https://github.com/user-attachments/assets/684d0671-27b8-4880-8387-3564918a8bc7)

PREDICTION
![image](https://github.com/user-attachments/assets/74218b34-db5a-4d56-a006-abd59c17d18a)

FINIAL PREDICTION
![image](https://github.com/user-attachments/assets/86600162-bf4a-4bbe-baea-798081b39f15)

### RESULT:
Thus we have successfully implemented the auto regression function using python.
