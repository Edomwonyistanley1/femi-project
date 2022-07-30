import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

def predictive_model():
    try:
        # Heading
        st.title('Crypto Price Prediction')

        # Crypto
        currencies = ("BTC-USD","ETH-USD","USDT-USD","BNB-USD","XRP-USD","MATIC-USD", "AVAX-USD", "WBTC-USD", "LEO-USD", "LTC-USD"
                        "HEX-USD", "DAI-USD", "DOT-USD", "TRX-USD", "SHIB-USD","USDC-USD", "ADA-USD", "DOGE-USD", "SOL-USD", "BUSD-USD")

        #Select Box
        selected_crypto = st.selectbox('Select a coin...', currencies)

        # Time and Date
        index_date = '2015-01-01'
        current_date = date.today().strftime('%Y-%m-%d')

        # Load selected coin dataset
        def input_data(tickers):
            df = yf.download(tickers, index_date, current_date)
            df.reset_index(inplace=True)

            return df

        loading_state = st.text('Loading Data...')

        df = input_data(selected_crypto)

        loading_state.text('Loading complete...')


        sub_heading = f'{selected_crypto} Dataset'
        st.subheader(sub_heading)
        st.write(df)

        
        # Visualization of data set
    
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close']))
        fig.update_layout(yaxis_title='Price')

        fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
        )

        st.plotly_chart(fig)


        fig = px.bar(df, x=df.Date, y='High')

        fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
        )

        st.plotly_chart(fig)

        fig = px.line(df, x='Date', y='Volume')
        fig.update_xaxes(ticks= "outside",
                    ticklabelmode= "period", 
                    tickcolor= "black", 
                    ticklen=10, 
                    minor=dict(
                        ticklen=4,  
                        dtick=7*24*60*60*1000,  
                        tick0="2016-07-03", 
                        griddash='dot', 
                        gridcolor='white'
                        )
                    )

        fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
        )

        st.plotly_chart(fig)
        # end

        # Predicting algorithm
        
        df_train_option = df[['Date', 'Close', 'Open', 'High', 'Low']]
        df_train_option = df_train_option.rename(columns=
                                                {"Date": "ds", "Close": "y", 
                                                "Open": "open", "High": "high", 
                                                "Low": "low"})

        algo = Prophet()
        algo.fit(df_train_option)

        # Predicting 1 year into the future
        df_to_predict = algo.make_future_dataframe(periods=365)
        forecast = algo.predict(df_to_predict)

        st.subheader('Forecasted Raw Dataset')
        st.write(forecast.tail())

        fig1 = plot_plotly(algo, forecast)
        st.plotly_chart(fig1)

    except:
        st.text('An error occurred ... Try again in 2 minutes time')


predictive_model()