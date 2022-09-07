import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import yfinance as yf

from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from plotly.graph_objs import Scatter




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
        
    def extract_data(self, stock_symbol, start, end):
                
        '''
        INPUT:
            stock_symbol - symbol for company stock
            start - start_date for training period(Reference Period)
            end - end_date for training period(Reference Period)
        OUTPUT:
            training_set - time series dataframe for company stock
        '''    
        #get data from quandl finance api


        self.ticket = yf.Ticker(stock_symbol)
        self.dataset= self.ticket.history(start=start,end=end)
        df = self.ticket.history(start=start,end=end)
        training_set = df.iloc[:,3]
        self.training_set = pd.DataFrame(training_set)
        self.training_set.reset_index(inplace = True)

        return self.training_set

    
   

    def model_train(self):
        '''
        INPUT: 
               
        OUTPUT:
            trained_model - model trained with the input date
        '''

        #Prepare the model

       

        model = SARIMAX(self.training_set['Close'],order=(0,0,1),
                        trend='n',
                        seasonal_order=(1,1,1,12))
        self.results = model.fit()

        return self.results



    def predict(self, predict_date):
        '''
        INPUT:
            predict_date - date for prediction
        OUTPUT:
            Prediction - Prediction till date  
        '''
        # data to be predicted - last date in training set
        pred_date = datetime.strptime(predict_date, '%Y-%m-%d')
        diff = pred_date - self.training_set['Date'].iloc[-1]
        span = diff.days +1
        
        #get the dates uptill the predicted date
        future_date = [self.training_set['Date'].iloc[-1] + DateOffset(days = i) for i in range(0, span)]

        #convert to dataframe
        future_date_df1 = pd.DataFrame(future_date, columns = ["Date"])[1:]#.set_index('Date')

        #get the prediction for the future dates
        start_, end_ = len(self.training_set)+1, len(self.training_set)+span
        future_date_df2 = pd.DataFrame(self.results.predict(start = start_, end = end_, dynamic= True).values)

        future_date_df2.columns = ['Forecast']

        self.df = future_date_df1.join(future_date_df2)

        return self.df.iloc[-1]



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
                    name='Reference period'
                ), 
                Scatter(
                     x=self.df['Date'],
                    y=self.df['Forecast'],
                    name='Forecast period'
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
                name='Actual'
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
                name='Actual'
            )
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

