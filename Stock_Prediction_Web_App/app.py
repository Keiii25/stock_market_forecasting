import json
from pyexpat import model
import plotly
import pandas as pd

from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
from model import Model

from datetime import datetime
from numerize import numerize
from datetime import timedelta

app = Flask(__name__, static_url_path='/static')
Bootstrap(app)

#this links to the result page of the web app
@app.route('/', methods=['GET', 'POST'])
def predict_plot():
    error = False
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
        PredictionDate = request.form["PredictionDate"]
        stock_symbol = companyname.upper() #["WIKI/AMZN"]
        start_date = ReferenceStartPeriod #datetime(2017, 1, 1)
        end_date = datetime.strftime(datetime.strptime(PredictionDate, '%Y-%m-%d')-timedelta(days=1), '%Y-%m-%d')
        prediction_date = PredictionDate

    error = False
    #build model
    model = Model(stock_symbol, start_date)

    #train the data
    model.load_model()

    try:
        #Predict the stock price for a given date
        stock_predict = model.prediction(prediction_date)
    except:
        error = True

    #extract data from api
    if error:
        stock_symbol = 'AAPL'
        model = Model(stock_symbol, start_date)
        model.load_model()
        model.prediction(prediction_date)


    #get the prediction graph
    graph_data = model.plot_data()
    graphJSON = json.dumps(graph_data, cls = plotly.utils.PlotlyJSONEncoder)

    # get the earning graph
    earning_data = model.plot_earning()
    earningGraphJSON = json.dumps(earning_data, cls=plotly.utils.PlotlyJSONEncoder)

    # get the revenue graph
    revenue_data = model.plot_revenue()
    revenueGraphJSON = json.dumps(revenue_data, cls=plotly.utils.PlotlyJSONEncoder)

    income_statement_graph = model.plot_income_statement()
    incomeStatementGraphJSON = json.dumps(income_statement_graph,  cls=plotly.utils.PlotlyJSONEncoder)

    balance_sheet_graph = model.plot_balance_sheet()
    balanceSheetGraphJSON = json.dumps(balance_sheet_graph, cls = plotly.utils.PlotlyJSONEncoder)

    #get the income statement correlation graph
    income_corr_data = model.plot_income_corr()
    incomeCorrGraphJSON = json.dumps(income_corr_data, cls = plotly.utils.PlotlyJSONEncoder)

    #get the cash flow correlation graph
    cash_corr_data = model.plot_cash_corr()
    cashCorrGraphJSON = json.dumps(cash_corr_data, cls = plotly.utils.PlotlyJSONEncoder)

    # Make the follow var to dataframe
    income  = pd.DataFrame(model.income_statement())
    history = pd.DataFrame(model.history())
    balance = pd.DataFrame(model.balance_sheet())
    cash_flow = pd.DataFrame(model.cash_flow())

    # calculate the percentage change for the close price
    current_close = model.history_info('Close')
    previous_close_price = history.iloc[-2, history.columns.get_loc("Close")]
    percentage_change = ((current_close - previous_close_price)/previous_close_price)*100

    # Calculate progress bar for DAY'S RANGE
    high_price = model.history_info('High')
    low_price = model.history_info('Low')
    day_bar =((high_price - current_close)/(high_price-low_price))*100

    # Calculate progress bar for 52 Week Range
    fiftyTwo_high = model.ticker('fiftyTwoWeekHigh')
    fiftyTwo_low = model.ticker('fiftyTwoWeekLow')
    fiftyTwo_bar = ((fiftyTwo_high - current_close)/(fiftyTwo_high-fiftyTwo_low))*100

    return render_template('dashboard.html',
                            error = error,
                        stock_predict = stock_predict,
                        graphJSON = graphJSON,
                        earningJSON= earningGraphJSON,
                        revenueJSON=revenueGraphJSON,
                        incomeStatementJSON = incomeStatementGraphJSON,
                        balanceSheetJSON = balanceSheetGraphJSON,
                        incomeJSON =incomeCorrGraphJSON,
                        cashJSON=cashCorrGraphJSON,
                        prediction_date = prediction_date,
                        stock_symbol = stock_symbol,
                        long_name = model.ticker('longName'),
                        open_price = model.history_info('Open'),
                        high_price = high_price,
                        low_price = low_price,
                        close_price = current_close,
                        summary = history.to_html(),
                        income_statement = income.to_html(),
                        balance_sheet = balance.to_html(),
                        cash_flow =   cash_flow.to_html(),
                        open_change = model.history_change('Open'),
                        high_change=  model.history_change('High'),
                        low_change =  model.history_change('Low'),
                        close_change= model.history_change('Close'),
                        business_profile = model.ticker('longBusinessSummary'),
                        previous_close = numerize.numerize(previous_close_price,2),
                        volume= numerize.numerize(model.ticker('volume'),2),
                        average_vol = numerize.numerize(model.ticker('averageVolume'),2),
                        market_cap = numerize.numerize(model.ticker('marketCap'),2),
                        dividend_rate= model.ticker('dividendRate'),
                        currency = model.ticker('currency'),
                        pe =  None if model.ticker('forwardPE') is None else numerize.numerize(model.ticker('forwardPE'),2),
                        eps =  None if model.ticker('forwardEps') is None else model.ticker('forwardEps'),
                        share_float = None if model.ticker('floatShares') is None else numerize.numerize(model.ticker('floatShares'),2),
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