import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from darts import TimeSeries
from darts.models import NBEATSModel
from pandas.tseries.offsets import BDay
from pytorch_lightning import Trainer
from plotly.graph_objs import Scatter, Bar, Heatmap
import warnings

warnings.filterwarnings("ignore")

class Model():
    '''
    A model for predicting the stock price
    '''

    def __init__(self, stock_symbol='AAPL', start_date='2022-09-01'):
        '''
        starting out with our model
        '''
        self.ticket = None
        self.dataset = None
        self.stock_symbol = stock_symbol
        self.start = start_date

    def extract_data(self, start, end):
        '''
        INPUT:
            start - start_date for training period(Reference Period)
            end - end_date for training period(Reference Period)
        OUTPUT:
            training_set - time series dataframe for company stock
        '''
        # get historical data from yfinance
        self.ticket = yf.Ticker(self.stock_symbol)
        self.dataset = self.ticket.history(start=start, end=end)
        training_set = self.dataset.iloc[:, 3]
        self.training_set = pd.DataFrame(training_set)
        self.training_set.reset_index(inplace=True)

        self.training_set['Date'] = pd.to_datetime(self.training_set['Date'])
        mask = self.training_set['Date'] >= datetime.strptime(self.start, '%Y-%m-%d')
        self.reference_data = self.training_set.loc[mask]
        self.reference_data['Close'] = self.reference_data['Close'].apply(lambda x: round(x, 2))

        if len(self.training_set) == 0:
            raise Exception
        return self.training_set

    def load_model(self):
        '''
        INPUT:

        OUTPUT:
            trained_model - model trained with the input date
        '''
        # Load trained model
        model_path = "2022-09-16/0_NBEATSModel_corr_0.75_icl150_ocl30_gTrue_s30_b1_l4_lw512_bs32_e100_start2019-01-02_end2022-07-01/_model.pth.tar"
        model = NBEATSModel.load_model(model_path)

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
        one_year_before = today - relativedelta(years=1)

        # Number of days to be predicted
        pred_date = datetime.strptime(predict_date, '%Y-%m-%d').date()
        n = (pred_date - today).days

        # Generate a list of business days
        template = pd.DataFrame(pd.date_range(one_year_before, today, freq=BDay()), columns=['Datetime']).set_index(
            "Datetime")

        # Load dataset
        predict_data_df = self.extract_data(one_year_before, today)
        predict_data_df = predict_data_df.set_index(pd.to_datetime(predict_data_df['Date']))

        # Ensure dataframe consists of all business days 
        predict_data_df = pd.merge(template, predict_data_df, left_index=True, right_index=True, how="left")
        predict_data_df.fillna(method="ffill", inplace=True)  # Forward fill missing values
        predict_data_df.fillna(method="bfill", inplace=True)  # Backward fill missing values
        previousdayof_today_closing = predict_data_df['Close'][-1]

        # Calculate the log return of stock prices
        predict_data_df["Close"] = (np.log(predict_data_df["Close"]) - np.log(predict_data_df["Close"].shift(1)))
        predict_data_df = predict_data_df[1:]  # To remove the first row of NaN value
        predict_data_series = predict_data_df["Close"]

        # Convert dataframe to timeseries
        data = TimeSeries.from_series(predict_data_series, freq='B')
        data = data.astype(np.float32)

        # Perform prediction
        model = self.load_model()
        trainer = Trainer(accelerator="cpu")
        result = model.predict(n, series=data, trainer=trainer)

        # Convert timeseries to dataframe        
        result_df = result.pd_dataframe()
        result_df = result_df.rename(columns={'Close': 'Log_return'})

        # Convert log return to price
        result_df = self.conversion(result_df, previousdayof_today_closing)

        # validation for prediction beyond the predict date
        validation_index = 0
        for i in range(len(result_df.index)):
            timestamp_date = result_df.index[i]
            required_date = datetime.strftime(timestamp_date, '%Y-%m-%d')
            if required_date == predict_date:
                validation_index = i
                break

        # prediction result 
        self.pred_res = result_df[:validation_index + 1]
        self.pred_res = self.pred_res.reset_index()
        self.pred_res['Price'] = self.pred_res['Price'].apply(lambda x: round(x, 2))

        return self.pred_res

    def conversion(self, result, previousdayof_today_closing):
        """
        INPUT:
            result - prediction table with log return
            previousdayof_today_closing - a close price from yesterday

        OUTPUT: prediction table with close price
        """
        result["Log_return"] = result["Log_return"] + np.log(previousdayof_today_closing)
        result["Log_return"] = np.exp(result["Log_return"])
        result = result.rename(columns={'Log_return': 'Price'})
        return result

    def plot_data(self):
        '''
        INPUT

        OUTPUT
            graph_data - containing data for ploting
        '''

        # merge the last row to the forecast dataframe
        last_row = self.reference_data.tail(1)
        last_row.rename(columns={'Close': 'Price', 'Date': 'Datetime'}, inplace=True)
        self.pred_res = pd.concat([last_row, self.pred_res])

        # plot line graph
        graph_data = [
            Scatter(
                x=self.reference_data['Date'],
                y=self.reference_data['Close'],
                hovertemplate=
                '<i>Date</i>: <b>%{x}</b>' +
                '<br>Period: <b>Reference</b> <br>' +
                '<i>Price</i>: <b>%{y:.2f}</b>',
                name='Reference period',
                marker=dict(color='#5D4E7B')
            ),
            Scatter(
                x=self.pred_res['Datetime'],
                y=self.pred_res['Price'],
                hovertemplate=
                '<i>Date</i>: <b>%{x}</b>' +
                '<br>Period: <b>Forecast</b> <br>' +
                '<i>Price</i>: <b>%{y:.2f}</b>',
                name='Forecast period',
                marker=dict(color='#FD8A75')
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

        # plot line graph
        graph_data = [

            Scatter(
                x=earning['Year'],
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

        # plot line graph
        graph_data = [

            Scatter(
                x=earning['Year'],
                y=earning['Revenue'],
                name='Actual',
                marker=dict(color='#FD8A75')
            )
        ]

        return graph_data

    def plot_income_statement(self):
        income_statement = pd.DataFrame(self.income_statement())
        income_statement = income_statement.transpose()
        income_statement = income_statement.reset_index()
        income_statement.rename(columns={income_statement.columns[0]: "Year"}, inplace=True)
        income_statement['Year'] = pd.to_datetime(income_statement['Year'], '%Y')

        # plot bar graph
        graph_data = [
            Bar(x=income_statement['Year'].dt.year, y=income_statement['Total Revenue'], name='Total Revenue',
                marker=dict(color='#5D4E7B')),
            Bar(x=income_statement['Year'].dt.year, y=income_statement['Net Income'], name='Net Income',
                marker=dict(color='#FD8A75'))
        ]
        return graph_data

    def plot_balance_sheet(self):
        balance_sheet = pd.DataFrame(self.balance_sheet())
        balance_sheet = balance_sheet.transpose()
        balance_sheet = balance_sheet.reset_index()
        balance_sheet.rename(columns={balance_sheet.columns[0]: "Year"}, inplace=True)
        balance_sheet['Year'] = pd.to_datetime(balance_sheet['Year'], '%Y')

        # plot bar graph
        graph_data = [
            Bar(x=balance_sheet['Year'].dt.year, y=balance_sheet['Total Assets'], name='Total Assets',
                marker=dict(color='#5D4E7B')),
            Bar(x=balance_sheet['Year'].dt.year, y=balance_sheet['Total Liab'], name='Total Liability',
                marker=dict(color='#FD8A75')),
        ]
        return graph_data

    def plot_income_corr(self):
        """
        INPUT

        OUTPUT
            graph_data - containing data for ploting
        """
        # Extract yearly close price
        yearly_price = pd.DataFrame(self.ticket.history(period="5y"))
        yearly_price.reset_index(inplace=True)
        yearly_price = yearly_price.groupby(yearly_price["Date"].map(lambda x: x.year)).mean()
        yearly_price.reset_index(inplace=True)
        yearly_price.rename(columns={'Date': 'Year'}, inplace=True)
        yearly_price = yearly_price.filter(items=['Year', 'Close'])

        # Extract yearly income statement
        income_statement = pd.DataFrame(self.income_statement())
        income_statement = income_statement.transpose()
        income_statement = income_statement.reset_index()
        income_statement.rename(columns={income_statement.columns[0]: "Year"}, inplace=True)

        # using dropna() function to remove the null value
        income_statement.dropna()
        income_statement = income_statement.filter(items=["Year", "Net Income",
                                                          "Selling General Administrative",
                                                          "Gross Profit",
                                                          "Operating Income",
                                                          "Total Revenue",
                                                          "Total Operating Expenses",
                                                          "Cost of Revenue",
                                                          "Net Income From Continuing  Ops",
                                                          "Net Income Applicable To Common Shares"])
        income_statement["Year"] = income_statement["Year"].map(lambda x: x.year)

        # make a correlation table
        merge_tbl = yearly_price.set_index('Year').join(income_statement.set_index('Year'), how='right').astype(float)
        corr_tbl = merge_tbl.corr(method='pearson')

        # plot heatmap
        graph_data = [
            Heatmap(
                {'z': corr_tbl.values.tolist(),
                 'x': corr_tbl.columns.tolist(),
                 'y': corr_tbl.index.tolist()},
                colorscale='purples'
            )
        ]

        return graph_data

    def plot_cash_corr(self):
        '''
        INPUT

        OUTPUT
            graph_data - containing data for ploting
        '''
        # Extract yearly close price
        yearly_price = pd.DataFrame(self.ticket.history(period="5y"))
        yearly_price.reset_index(inplace=True)
        yearly_price = yearly_price.groupby(yearly_price["Date"].map(lambda x: x.year)).mean()
        yearly_price.reset_index(inplace=True)
        yearly_price.rename(columns={'Date': 'Year'}, inplace=True)
        yearly_price = yearly_price.filter(items=['Year', 'Close'])

        # Extract yearly icnome statement
        cash_flow = pd.DataFrame(self.cash_flow())
        cash_flow = cash_flow.transpose()
        cash_flow = cash_flow.reset_index()
        cash_flow.rename(columns={cash_flow.columns[0]: "Year"}, inplace=True)

        # using dropna() function to remove the null value
        cash_flow.dropna()
        cash_flow["Year"] = cash_flow["Year"].map(lambda x: x.year)

        # make a correlation table
        merge_tbl = yearly_price.set_index('Year').join(cash_flow.set_index('Year'), how='right').astype(float)
        corr_tbl = merge_tbl.corr(method='pearson')

        # plot heatmap
        graph_data = [
            Heatmap(
                {'z': corr_tbl.values.tolist(),
                 'x': corr_tbl.columns.tolist(),
                 'y': corr_tbl.index.tolist()},
                colorscale='purples'
            )
        ]

        return graph_data

    def history_info(self, info):
        """
        INPUT
            info: column name from the dataset
        OUTPUT
            The last row of the selected column name in 2 decimal point.
        """
        return round(self.dataset[info].iloc[-1], 2)

    def income_statement(self):
        """
        INPUT

        OUTPUT
            all data about Income statement
        """
        return self.ticket.financials

    def balance_sheet(self):
        """
        INPUT

        OUTPUT
            all data about balance sheet
        """
        return self.ticket.balancesheet

    def cash_flow(self):
        """
        INPUT

        OUTPUT
            all data about cash flow
        """
        return self.ticket.cashflow

    def history_change(self, info):
        """
        INPUT
            info - the price name
        OUTPUT
            The change from the previous date
        """
        return round(self.dataset[info].iloc[-1] - self.dataset[info].iloc[-2], 2)

    def history(self):
        """
        return the historical dataset
        """
        return self.dataset

    def ticker(self, obj):
        """
        INPUT
            obj - the stock information name
        OUTPUT
            the value for the stock information
        """
        try:
            return self.ticket.info[obj]
        except:
            return None
