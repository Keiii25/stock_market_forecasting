# from argparse import Action
import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd

# set page
st.set_page_config(page_title="N-BEATS", layout='wide')

# Give title and sub-header
st.title(""" Predict Stock with *N-BEATS*""")
st.write("""We are using a meta-learning model called N-BEATS to help traders to forecast the future price of an individual publicly traded stock.""")

# Take stock input
form = st.form(key="annotation")
with form:
    cols = st.columns(3)
    stock_name = cols[0].text_input(label="Ticker",value="aapl", placeholder='Search for ticker')
    calender = cols[1].date_input("Date", date.today())
    day = cols[2].slider("Days", 1, 31)
    submitted = st.form_submit_button(label="Search")


# Return result when click search button
if submitted:
    # Result
    ticker = yf.Ticker(stock_name)


    st.header(ticker.info['longName'])
    st.caption(f"{stock_name.upper()} ")
    st.markdown("""---""")

    # create columns
    g1,g2 = st.columns((1,0.5))
    # line chart for history data
    g1.line_chart(data=ticker.history(period="20y",actions=False))
    with g2:
        st.markdown("""
        some data """)

    # Balance Sheet
    tab1, tab2, tab3, tab4,tab5 = st.tabs(["Profile",
                                     "Balance Sheet",
                                     "Income Statement",
                                     "Recommendation",
                                     "News"])

    tab1.subheader(f"Business Summary")
    tab1.write(ticker.info['longBusinessSummary'])

    tab2.subheader(f"Balance Sheet for {day} days")
    tab2.table(ticker.balance_sheet)

    # Financial Data
    tab3.subheader(f"Income Statement for {day} days")
    tab3.table(ticker.financials)

    recommendation = pd.DataFrame(ticker.recommendations)
    sort_recommed = recommendation.sort_index(ascending=False)
    tab4.table(sort_recommed)

    tab5.write(ticker.info)

