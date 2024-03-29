import matplotlib.pyplot as plt
import pandas as pd
from prediction import forecast_arima, forecast_LSTM, forecast_arimax
from pandas.plotting import register_matplotlib_converters
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from pykalman import KalmanFilter
import datetime
import pytz
register_matplotlib_converters()

# Input variables
arima_parameters = (6, 1, 4)  # p, d and q for ARIMA
forecast_steps=30;  # Forecast in the future
time_step = 15  # Past time steps considered for LSTM
iter_LSTM=10
run_arima = 0  # Set to 1 if you want to run ARIMA prediction, else 0
run_lstm=0  # Set to 1 if you want to run the LSTM prediction, else 0
run_arimax=0 
lag_plot=2

############ RANDOM DATA GENERATION 
years = range(2015, 2021)  # chosen duration of years for which random data is generated.

ev_sales = []
initial_sales = 100  # Set initial sales value outside the loop
for year in years:
    monthly_growth = np.random.uniform(0.01, 0.02, 12)  
    monthly_sales = [initial_sales]  # Start from last year's ending value
    for growth in monthly_growth:
        monthly_sales.append(monthly_sales[-1] * (1 + growth))
    ev_sales.extend(monthly_sales[1:])  # Append without the initial value
    initial_sales = monthly_sales[-1]

base_gdp = 2000  # starting GDP
growth_rates = np.random.uniform(low=0.01, high=0.05, size=len(years)) 
gdp = []
for rate in growth_rates:
    base_gdp *= (1 + rate)
    gdp.append(base_gdp)

# Battery Costs (Yearly, with some price drops - random)
battery_costs = []
base_cost = 1000  # Base cost
price_drops = np.random.choice([0, -50, -100], size=len(years), replace=True, p=[0.8, 0.15, 0.05])  # Simulate occasional price drops in BESS
for year, drop in zip(years, price_drops):
  battery_costs.append(base_cost + drop)


#################### CODE STARTS BELOW

def adf_test(series, title=''):
    """Perform ADF test and print the results"""
    print(f'ADF Test on "{title}"')
    result = adfuller(series, autolag='AIC')
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result[:4], labels):
        print(f'{label} : {value}')
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis, time series has a unit root, indicating it is non-stationary ")

adf_test(ev_sales, title='Monthly EV Sales')

# Plot ACF and PACF for EV sales data
plt.figure(figsize=(12, 8))
plt.subplot(211)
plot_acf(ev_sales, lags=lag_plot, ax=plt.gca(), title='ACF for EV Sales Data')
plt.subplot(212)
plot_pacf(ev_sales, lags=lag_plot, ax=plt.gca(), title='PACF for EV Sales Dat')
plt.show()

acf_values = acf(ev_sales, nlags=lag_plot, fft=True) 
pacf_values = pacf(ev_sales, nlags=lag_plot)

ev_sales = pd.Series(ev_sales)
lags = 3
lagged_data = pd.DataFrame(index=range(len(ev_sales))) 

for i in range(1, lags + 1):
    lagged_data[f'Data_lag_{i}'] = ev_sales.shift(i)

differenced_data = ev_sales.diff().to_frame('EV_sales_diff')
differenced_data = differenced_data.dropna()

acf_values_diff = acf(differenced_data['EV_sales_diff'], nlags=lag_plot, fft=True)
pacf_values = pacf(differenced_data['EV_sales_diff'], nlags=lag_plot)
plt.figure(figsize=(12, 8))
plt.subplot(211)
plot_acf(differenced_data['EV_sales_diff'], lags=lag_plot, ax=plt.gca(), title='ACF for differenced series')
plt.subplot(212)
plot_pacf(differenced_data['EV_sales_diff'], lags=lag_plot, ax=plt.gca(), title='PACF for differenced series')
plt.show()


########### Chow-Lin Regression Based Data Dis-Aggregation

annual_ev_sales = pd.Series(ev_sales).groupby(pd.Series(ev_sales).index // 12).sum()
df_ols = pd.DataFrame({'gdp': gdp, 'annual_ev_sales': annual_ev_sales})

df_ols['log_gdp'] = np.log(df_ols['gdp'])  # Logarithmic transformation
df_ols['log_annual_ev_sales'] = np.log(df_ols['annual_ev_sales'])

X = df_ols['annual_ev_sales'] 
y = df_ols['gdp']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
residuals = model.resid


plt.figure(figsize=(10, 6))
plt.scatter(annual_ev_sales, y, color='blue')
plt.title('Annual EV Sales vs. Annual GDP')
plt.xlabel('Annual EV Sales')
plt.ylabel('Annual GDP')
plt.show()

### smearing_factor = np.exp(residuals).mean()

intercept, slope = model.params
print(model.summary()) 

# log_predicted_gdp = intercept + slope * np.log(ev_sales)
#predicted_gdp = np.exp(log_predicted_gdp)

predicted_gdp = intercept + slope * (ev_sales)
df_disaggregated = pd.DataFrame({'disaggregated_gdp': predicted_gdp})
yearly_sum_gdp=pd.Series(predicted_gdp).groupby(pd.Series(predicted_gdp).index // 12).sum()


################ Kalman Data Dis-Aggregation Method

n_years = len(df_ols)
n_months = 12 * n_years
normalized_ev_sales = ev_sales / ev_sales.max()

df_monthly = pd.DataFrame(index=range(len(ev_sales)), columns=['Estimated GDP'])

for year in range(n_years):
    # Assuming ev_sales and df_ols are indexed by date or year for easy slicing
    start_index = year * 12
    end_index = (year + 1) * 12
    yearly_ev_sales = ev_sales[start_index:end_index].values
    
    observations = yearly_ev_sales 

    initial_state_mean = np.array([df_ols['gdp'].iloc[year] / 12])  # Starting with the monthly equivalent of the annual GDP of the particular 'year'
    initial_state_covariance = np.array([[1e-1]])  
    transition_matrices = np.array([[1.01]])  # Transition of GDP monthly based on previous GDP
    observation_matrices = np.array([[0.50]])  # Influence of EV sales on GDP
    observation_covariance = np.array([[0.001]])  
    transition_covariance = np.array([[0.001]])  
    
    kf = KalmanFilter(initial_state_mean=initial_state_mean,
                      initial_state_covariance=initial_state_covariance,
                      transition_matrices=transition_matrices,
                      observation_matrices=observation_matrices,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance)
    
    state_means, _ = kf.filter(observations)
    
    df_monthly['Estimated GDP'].iloc[start_index:end_index] = state_means.flatten()

yearly_estimated_gdp = [df_monthly['Estimated GDP'].iloc[i*12:(i+1)*12].sum() for i in range(n_years)]

################### End of Kalman


plt.figure(figsize=(10, 6))  
plt.scatter(df_ols.index, df_ols['gdp'], color='blue', label='Actual GDP')
plt.plot(df_ols.index, yearly_sum_gdp, color='red', label='Regression Fit GDP')
plt.title('Actual GDP vs. Regression Fit GDP')
plt.xlabel('Time')
plt.ylabel('GDP')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))  
plt.plot(yearly_sum_gdp, color='blue', label='Predicted Annual GDP')
plt.plot(gdp, color='red', label='Actual GDP')
plt.xlabel('Time')
plt.ylabel('GDP')
plt.legend()
plt.show()



#### Machine Learning Predictions
if run_arima == 1:
    ARIMA_prediction = forecast_arima(ev_sales, arima_parameters, forecast_steps)
else:
    print("Skipping ARIMA.")
    
if run_lstm == 1:
    LSTM_prediction = forecast_LSTM(ev_sales, forecast_steps, time_step, iter_LSTM)
else:
    print("Skipping LSTM.")

if run_arimax == 1:
    ARIMAX_prediction = forecast_arimax(ev_sales, arima_parameters, forecast_steps)
else:
    print("Skipping ARIMAX.")    