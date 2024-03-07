import matplotlib.pyplot as plt
import pandas as pd
#from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
import statsmodels.tsa.arima.model as sm 
register_matplotlib_converters()


####### ARIMA PREDICTION BELOW ########
def forecast_arima(ev_sales, arima_parameters, forecast_steps):

    model = sm.ARIMA(ev_sales, order=arima_parameters)  # Example order
    model_fit = model.fit()
    
    # Forecast the next days
    forecast = model_fit.forecast(steps=forecast_steps)
    aic_value = model_fit.aic
    bic_value = model_fit.bic
    print("AIC:", aic_value)
    print("BIC:", bic_value)
    
    plt.figure(figsize=(10,6))
    plt.plot(ev_sales, label='Historical Sales')  
    plt.plot(range(len(ev_sales), len(ev_sales)  + len(forecast)), forecast, label='Predicted EV Sales', color='red')     
    plt.xlabel('Time Period')
    plt.ylabel('Sales Number')
    plt.title('ARIMA Forecast')
    plt.legend()
    plt.show()    
    return forecast


####### ARIMAX PREDICTION BELOW ########
def forecast_arimax(ev_sales, arima_parameters, forecast_steps):
    ev_sales = ev_sales.astype('int64')
    model = sm.ARIMA(ev_sales, order=arima_parameters, exog=ev_sales)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_steps)
    
    # forecast now contains the predicted values as a pandas Series
    predicted_sales = forecast
    
    plt.figure(figsize=(10,6))
    plt.plot(ev_sales, label='Historical Sales')
    plt.plot(range(len(ev_sales), len(ev_sales)  + len(forecast)), forecast, label='Predicted EV Sales', color='red')
    plt.xlabel('Date')
    plt.ylabel('Sales Number')
    plt.title('ARIMAX Model Forecast')
    plt.legend()
    plt.show()  
    return forecast

def forecast_LSTM(ev_sales, forecast_steps, time_step, iter_LSTM):
    
    ####### LSTM PREDICTION BELOW ########
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(ev_sales.values.reshape(-1,1))

    # Create a dataset with 'X' number of past days to predict future
    def create_dataset(data, time_step):
        X_data, y_data = [], []
        for i in range(len(data)-time_step-1):
            X_data.append(data[i:(i+time_step), 0])
            y_data.append(data[i + time_step, 0])
        return np.array(X_data), np.array(y_data)

    X, y = create_dataset(scaled_data, time_step)

    # Reshape input to format [samples, time steps, features],required for LSTM
    X = X.reshape(X.shape[0],X.shape[1],1)

    train_size = int(len(X) * 0.80) # 80 20 partition for training and testing
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))  # Dropout for regularization
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    #model.add(LSTM(units=100, return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=100, return_sequences=True))  # Choose these LSTM layers if needed for more complex datasets
    #model.add(Dropout(0.2))
    model.add(LSTM(units=100))  # Last LSTM layer, thus no return_sequences
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=iter_LSTM)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean(predictions - y_test)**2)

    future_days = forecast_steps
    new_input = scaled_data[-time_step:].reshape(1, time_step, 1)
    predicted_lstm = []

    for _ in range(future_days):
        new_output = model.predict(new_input)
    
        if new_output.size == 0:
            raise ValueError("Model did not generate a prediction.")
        predicted_lstm.append(new_output[0, 0])

        new_input = np.append(new_input[:, 1:, :], new_output[0].reshape(1, 1, 1), axis=1)

    if not predicted_lstm:
        raise ValueError("No predictions were made.")
    predicted_lstm = np.array(predicted_lstm).reshape(-1, 1)

    predicted_lstm = scaler.inverse_transform(predicted_lstm)
    predicted_lstm = predicted_lstm.ravel()  # Flatten the array


    plt.figure(figsize=(10,6))
    plt.plot(ev_sales, label='Historical Sales')  
    plt.plot(range(len(ev_sales), len(ev_sales)  + len(predicted_lstm)), predicted_lstm, label='Predicted EV Sales', color='red')     
    plt.xlabel('Time Period')
    plt.ylabel('Sales Number')
    plt.title('LSTM Forecast')
    plt.legend()
    plt.show()
    
    return predicted_lstm