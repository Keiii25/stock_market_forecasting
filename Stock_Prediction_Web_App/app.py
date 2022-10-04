import json
import plotly
import pandas as pd

from flask import Flask, request, render_template, url_for, jsonify
from flask_bootstrap import Bootstrap
from model import Model

from datetime import datetime
from plotly.graph_objs import Scatter
from numerize import numerize
from datetime import timedelta

app = Flask(__name__, static_url_path='/static')
Bootstrap(app)

#this links to the result page of the web app
@app.route('/', methods=['GET', 'POST'])
def predict_plot():
    print(request.form)
    if request.method == 'GET':
        companyname = 'AAPL'
        ReferenceStartPeriod = '2022-09-01'
        PredictionDate = datetime.strftime(datetime.today() + timedelta(days=1), '%Y-%m-%d')
        stock_symbol = companyname.upper() #["WIKI/AMZN"]
        start_date = ReferenceStartPeriod #datetime(2017, 1, 1)
        end_date = datetime.strftime(datetime.strptime(PredictionDate, '%Y-%m-%d')-timedelta(days=1), '%Y-%m-%d')
        prediction_date = PredictionDate
    else:

    #get the varaible inputs from the user
        companyname = request.form["companyname"]
        ReferenceStartPeriod = request.form["ReferenceStartPeriod"]
        # ReferenceEndPeriod = request.args.get("ReferenceEndPeriod", "")
        PredictionDate = request.form["PredictionDate"]
        stock_symbol = companyname.upper() #["WIKI/AMZN"]
        start_date = ReferenceStartPeriod #datetime(2017, 1, 1)
        end_date = datetime.strftime(datetime.strptime(PredictionDate, '%Y-%m-%d')-timedelta(days=1), '%Y-%m-%d')
        prediction_date = PredictionDate

    error = False
    #build model
    arima = Model()

    try:
        arima.extract_data(stock_symbol, start_date, end_date)
        arima.model_train()
    except:
        error = True
    #extract data from api
    if error:
        stock_symbol = 'AAPL'
        arima.extract_data(stock_symbol, start_date, end_date)
        arima.model_train()

    #train the data 
    # arima.model_train()

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

    income_statement_graph = arima.plot_income_statement()
    incomeStatementGraphJSON = json.dumps(income_statement_graph,  cls=plotly.utils.PlotlyJSONEncoder)

    balance_sheet_graph = arima.plot_balance_sheet()
    balanceSheetGraphJSON = json.dumps(balance_sheet_graph, cls = plotly.utils.PlotlyJSONEncoder)


    income  = pd.DataFrame(arima.income_statement())
    history = pd.DataFrame(arima.history())
    balance = pd.DataFrame(arima.balance_sheet())
    cash_flow = pd.DataFrame(arima.cash_flow())

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
                            error = error, 
                        stock_predict = stock_predict,
                        graphJSON = graphJSON,
                        earningJSON= earningGraphJSON,
                        revenueJSON=revenueGraphJSON,
                        incomeStatementJSON = incomeStatementGraphJSON,
                        balanceSheetJSON = balanceSheetGraphJSON,
                        prediction_date = prediction_date,
                        stock_symbol = stock_symbol,
                        long_name = arima.ticker('longName'),
                        open_price = arima.history_info('Open'),
                        high_price = high_price,
                        low_price = low_price,
                        close_price = current_close,
                        summary = history.to_html(),
                        income_statement = income.to_html(),
                        balance_sheet = balance.to_html(),
                        cash_flow =   cash_flow.to_html(),
                        open_change = arima.history_change('Open'),
                        high_change=  arima.history_change('High'),
                        low_change =  arima.history_change('Low'),
                        close_change= arima.history_change('Close'),
                        business_profile = arima.business_info(),
                        previous_close = numerize.numerize(previous_close_price,2),
                        volume= numerize.numerize(arima.ticker('volume'),2),
                        average_vol = numerize.numerize(arima.ticker('averageVolume'),2),
                        market_cap = numerize.numerize(arima.ticker('marketCap'),2),
                        dividend_rate= arima.ticker('dividendRate'),
                        currency = arima.ticker('currency'),
                        pe =  None if arima.ticker('forwardPE') is None else numerize.numerize(arima.ticker('forwardPE'),2),
                        eps =  None if arima.ticker('forwardEps') is None else arima.ticker('forwardEps'),
                        share_float = None if arima.ticker('floatShares') is None else numerize.numerize(arima.ticker('floatShares'),2),
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
    
    