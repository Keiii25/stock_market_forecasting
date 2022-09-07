import json
import plotly
import pandas as pd

from flask import Flask, request,render_template, url_for, jsonify
from flask_bootstrap import Bootstrap
from model import Model

from datetime import datetime
from plotly.graph_objs import Scatter
from numerize import numerize

app = Flask(__name__, static_url_path='/static')
Bootstrap(app)

@app.route('/')

#this links to the index page of the web app
@app.route('/index.html')
def index():
    return render_template('index.html')

#this links to the result page of the web app
@app.route('/dashboard.html')
def predict_plot():

    #get the varaible inputs from the user
    companyname = request.args.get("companyname", "")
    ReferenceStartPeriod = request.args.get("ReferenceStartPeriod", "")
    ReferenceEndPeriod = request.args.get("ReferenceEndPeriod", "")
    PredictionDate = request.args.get("PredictionDate", "")

    
    stock_symbol = companyname.upper() #["WIKI/AMZN"]
    start_date = ReferenceStartPeriod #datetime(2017, 1, 1)
    end_date = ReferenceEndPeriod #datetime(2017, 12, 31)
    prediction_date = PredictionDate

    # stock_symbol = "AAPL"
    # start_date = "2022-08-01"
    # end_date =  "2022-09-03"
    # prediction_date = "2022-09-03"

    #build model
    arima = Model()

    #extract data from api
    arima.extract_data(stock_symbol, start_date, end_date)

    #train the data 
    arima.model_train()

    #Predict the stock price for a given date
    stock_predict = round(arima.predict(prediction_date)[1],2)


    #get the prediction graph
    graph_data = arima.plot_data()
    graphJSON = json.dumps(graph_data, cls = plotly.utils.PlotlyJSONEncoder)

    # get the earning graph
    earning_data = arima.plot_earning()
    earningGraphJSON = json.dumps(earning_data, cls=plotly.utils.PlotlyJSONEncoder)

    # get the revenue graph
    revenue_data = arima.plot_revenue()
    revenueGraphJSON = json.dumps(revenue_data, cls=plotly.utils.PlotlyJSONEncoder)

    income  = pd.DataFrame(arima.income_statement())
    history = pd.DataFrame(arima.history())
    balance = pd.DataFrame(arima.balance_sheet())


    current_close = arima.history_info('Close')
    previous_close_price = history.iloc[-2, history.columns.get_loc("Close")]
    # arima.ticker('previousClose')
    percentage_change = ((current_close - previous_close_price)/previous_close_price)*100

    # Calculate progress bar for DAY'S RANGE
    high_price = arima.history_info('High')
    low_price = arima.history_info('Low')
    day_bar =((high_price - current_close)/(high_price-low_price))*100

    # Calculate progress bar for 52 Week Range
    fiftyTwo_high = arima.ticker('fiftyTwoWeekHigh')
    fiftyTwo_low = arima.ticker('fiftyTwoWeekLow')
    fiftyTwo_bar = ((fiftyTwo_high - current_close)/(fiftyTwo_high-fiftyTwo_low))*100



    return render_template('dashboard.html',
                           stock_predict = stock_predict,
                           graphJSON = graphJSON,
                           earningJSON= earningGraphJSON,
                           revenueJSON=revenueGraphJSON,
                           prediction_date = prediction_date,
                           stock_symbol = stock_symbol,
                           long_name = arima.ticker('longName'),
                           open_price = arima.history_info('Open'),
                           high_price = high_price,
                           low_price = low_price,
                           close_price = current_close,
                           summary = history.to_html(),
                           business_profile = arima.business_info(),
                           income_statement = income.to_html(),
                           balance_sheet = balance.to_html(),
                           cash_flow =   arima.cash_flow(),
                           open_change = arima.history_change('Open'),
                           high_change=  arima.history_change('High'),
                           low_change =  arima.history_change('Low'),
                           close_change= arima.history_change('Close'),
                           previous_close = numerize.numerize(previous_close_price,2),
                           volume= numerize.numerize(arima.ticker('volume'),2),
                           average_vol = numerize.numerize(arima.ticker('averageVolume'),2),
                           market_cap = numerize.numerize(arima.ticker('marketCap'),2),
                           dividend_rate= arima.ticker('dividendRate'),
                           currency = arima.ticker('currency'),
                           pe =  numerize.numerize(arima.ticker('forwardPE'),2),
                           eps =  arima.ticker('forwardEps'),
                           share_float = numerize.numerize(arima.ticker('floatShares'),2),
                           close_percentage_change = numerize.numerize(percentage_change,2),
                           fiftyTwoWeeksHigh = numerize.numerize(fiftyTwo_high,2),
                           fiftyTwoWeeksLow=numerize.numerize(fiftyTwo_low, 2),
                           day_change_progress = day_bar,
                           fiftyTwo_change_progress =  fiftyTwo_bar
                           )



def main():
    app.run(debug =True)

if __name__ == '__main__':
    main()
    
    