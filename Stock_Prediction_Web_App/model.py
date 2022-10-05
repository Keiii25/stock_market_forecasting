from pickle import TRUE
from unicodedata import name
from black import NewLine
from matplotlib.pyplot import close
import numpy as np
import datetime
import numpy as np
import pandas as pd


import yfinance as yf

from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, date
from pandas.tseries.offsets import DateOffset
from plotly.graph_objs import Scatter, Bar
import plotly.express as px

from dateutil.relativedelta import relativedelta
from darts import TimeSeries
from darts.models import NBEATSModel
from pandas.tseries.offsets import BDay
from pytorch_lightning import Trainer
import os

class Model():
    '''
    A model for predicting the stock price
    '''
    def __init__(self):
        '''
        starting out with our model
        '''
        self.ticket = None
        self.dataset = None
        self.stock_symbol = "AAPL"
        
    def extract_data(self, start, end):           
        '''
        INPUT:
            start - start_date for training period(Reference Period)
            end - end_date for training period(Reference Period)
        OUTPUT:
            training_set - time series dataframe for company stock
        '''    
        #get data from quandl finance api
        self.ticket = yf.Ticker(self.stock_symbol)
        self.dataset= self.ticket.history(start=start,end=end)
        df = self.ticket.history(start=start,end=end)
        training_set = df.iloc[:,3]
        self.training_set = pd.DataFrame(training_set)
        self.training_set.reset_index(inplace = True)

        return self.training_set

    def load_model(self):
        '''
        INPUT: 
               
        OUTPUT:
            trained_model - model trained with the input date
        '''
        #Load trained model 
        logs_path = os.path.join(os.getcwd(), "2022-09-16")
        model_name = "0_NBEATSModel_corr_0.75_icl150_ocl30_gTrue_s30_b1_l4_lw512_bs32_e100_start2019-01-02_end2022-07-01"
        model = NBEATSModel.load_from_checkpoint(model_name = model_name, work_dir = logs_path, best = False)

        return model

    def prediction(self, predict_date):
        '''
        INPUT:
            predict_date - date for prediction
        OUTPUT:
            Prediction - date and log stock returns of the predicted stock   
        '''        
        # Today's Date - YY-mm-dd        
        today = date.today()
        one_year_before = today - relativedelta (years= 1)
                
        # Number of days to be predicted
        pred_date = datetime.strptime(predict_date, '%Y-%m-%d').date()
        n = (pred_date - today).days
        
        # Generate a list of business days
        template = pd.DataFrame(pd.date_range(one_year_before, today, freq=BDay()), columns=['Datetime']).set_index("Datetime")

        # Load dataset
        predict_data_df = self.extract_data(one_year_before, today)
        predict_data_df = predict_data_df.set_index(pd.to_datetime(predict_data_df['Date']))
        
        # Ensure dataframe consists of all business days 
        predict_data_df = pd.merge(template, predict_data_df, left_index=True, right_index=True, how="left")
        predict_data_df.fillna(method = "ffill", inplace = True) # Forward fill missing values
        predict_data_df.fillna(method = "bfill", inplace = True) # Backward fill missing values
        previousdayof_today_closing = predict_data_df['Close'][-1]
        
        # Calculate the log return of stock prices
        predict_data_df["Close"] = (np.log(predict_data_df["Close"]) - np.log(predict_data_df["Close"].shift(1)))
        predict_data_df = predict_data_df[1:]    # To remove the first row of NaN value
        predict_data_series = predict_data_df["Close"]

        # Convert dataframe to timeseries
        data = TimeSeries.from_series(predict_data_series, freq = 'B')
        
        # Perform prediction
        model = self.load_model()
        trainer = Trainer(accelerator="cpu")
        result = model.predict(n, series=data, trainer = trainer)
        
        # Convert timeseries to dataframe        
        result_df = result.pd_dataframe()
        result_df = result_df.rename(columns={'Close':'Log_return'})
        
        # Convert log return to price
        result_df["Log_return"] = result_df["Log_return"] + np.log(previousdayof_today_closing) 
        result_df["Log_return"] = np.exp(result_df["Log_return"])
        result_df = result_df.rename(columns={'Log_return':'Price'})
        
        # validation for prediction beyond the predict date
        validation_index = 0
        for i in range(len(result_df.index)):
            timestamp_date = result_df.index[i]
            required_date = datetime.strftime(timestamp_date, '%Y-%m-%d') 
            if required_date == predict_date:
                validation_index = i
                break
        
        # prediction result 
        output = result_df[:validation_index+1]
            
        return output    
        

    def plot_data(self):


        '''
        INPUT 
            
        OUTPUT
            graph_data - containing data for ploting
        '''


        graph_data = [
        
                Scatter(
                    x=self.training_set['Date'],
                    y=self.training_set['Close'],
                    name='Reference period',
                    marker=dict(color='#5D4E7B')
                ), 
                Scatter(
                     x=self.df['Date'],
                    y=self.df['Forecast'],
                    name='Forecast period',
                    marker=dict(color ='#FD8A75')
                )
            ]
        
        return graph_data

    def plot_earning(self):
        '''
        INPUT

        OUTPUT
            graph_data - containing data for ploting
        '''
        earning = pd.DataFrame(self.ticket.earnings)
        earning = earning.reset_index()

        graph_data = [

            Scatter(
                x= earning['Year'],
                y=earning['Earnings'],
                name='Actual',
                marker=dict(color='#5D4E7B')
            )
        ]

        return graph_data

    def plot_revenue(self):
        '''
        INPUT

        OUTPUT
            graph_data - containing data for ploting
        '''
        earning = pd.DataFrame(self.ticket.earnings)
        earning = earning.reset_index()

        graph_data = [

            Scatter(
                x= earning['Year'],
                y= earning['Revenue'],
                name='Actual',
                marker=dict(color ='#FD8A75')
            )
        ]

        return graph_data

    def plot_income_statement(self):
        income_statement = pd.DataFrame(self.income_statement())
        income_statement = income_statement.transpose()
        income_statement = income_statement.reset_index()
        income_statement.rename(columns={ income_statement.columns[0]: "Year" }, inplace = True)
        income_statement['Year'] = pd.to_datetime(income_statement['Year'], '%Y')
        graph_data = [
            Bar(x= income_statement['Year'].dt.year ,y= income_statement['Total Revenue'],  name='Total Revenue', marker=dict(color='#5D4E7B')),
            Bar(x= income_statement['Year'].dt.year ,y= income_statement['Net Income'], name='Net Income', marker=dict(color ='#FD8A75'))
        ]
        return graph_data

    def plot_balance_sheet(self):
        balance_sheet = pd.DataFrame(self.balance_sheet())
        balance_sheet =  balance_sheet.transpose()
        balance_sheet = balance_sheet.reset_index()
        balance_sheet.rename(columns={ balance_sheet.columns[0]: "Year" }, inplace = True)
        balance_sheet['Year'] = pd.to_datetime(balance_sheet['Year'], '%Y')
        graph_data = [
            Bar(x= balance_sheet['Year'].dt.year,y= balance_sheet['Total Assets'], name='Total Assets', marker=dict(color='#5D4E7B')),
            Bar(x= balance_sheet['Year'].dt.year ,y= balance_sheet['Total Liab'], name='Total Liability', marker=dict(color ='#FD8A75')),
        ]
        return graph_data

    def history_info(self,info):
        return round(self.dataset[info].iloc[-1], 2)

    def business_info(self):
        return self.ticket.info['longBusinessSummary']

    def income_statement(self):
        return self.ticket.financials

    def balance_sheet(self):
        return self.ticket.balancesheet

    def cash_flow(self):
        return self.ticket.cashflow

    def history_change(self,info):
        return round(self.dataset[info].iloc[-1]-self.dataset[info].iloc[-2], 2)

    def history(self):
        return self.dataset

    def ticker(self, obj):
        return self.ticket.info[obj]


prediction_date = "2022-10-20"
m = Model()
#m.extract_data(start_date, end_date)
#m.load_model()
m.prediction(prediction_date)
