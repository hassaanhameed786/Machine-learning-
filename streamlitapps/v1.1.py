
# Import necessary libraries
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

# Function to get historical data for a given cryptocurrency
def get_crypto_data(crypto_symbol, start_date, end_date):
    crypto_data = yf.download(crypto_symbol, start=start_date, end=end_date)
    return crypto_data

# Function to create a candlestick chart
def plot_candlestick_chart(data, crypto_symbol):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title=f'Candlestick Chart for {crypto_symbol}',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      template='plotly_dark')
    return fig

# BTC Data Page
def crypto_data_page(selected_crypto):
    st.success("Login Successful!")
    st.subheader(f"Crypto Candlestick Charts for {selected_crypto}")

    # Select date range
    start_date = st.date_input("Select start date", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("Select end date", pd.to_datetime("2022-12-31"))

    # Display selected date
    st.text(f"Selected start date: {start_date.strftime('%Y-%m-%d')}")
    st.text(f"Selected end date: {end_date.strftime('%Y-%m-%d')}")

    # Get crypto data
    crypto_data = get_crypto_data(f'{selected_crypto}-USD', start_date, end_date)

    # Display candlestick chart for selected crypto
    st.plotly_chart(plot_candlestick_chart(crypto_data, selected_crypto))

    # Calculate and display all-time high values
    ath_value = crypto_data['Close'].max()
    ath_date = crypto_data[crypto_data['Close'] == ath_value].index[0].strftime('%Y-%m-%d')

    st.write(f"All-Time High (ATH) Value: {ath_value}")
    st.write(f"ATH Date: {ath_date}")

# Login Page
def login_page():
    st.title("Login Page")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    login_button = st.button("Login")

    if login_button:
        # Check credentials (add your authentication logic here)
        if username == "admin" and password == "admin":
            st.session_state["authenticated"] = True

# Streamlit app
def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        # Sidebar with crypto selection
        selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", ["BTC", "ETH", "ADA"])
        crypto_data_page(selected_crypto)
    else:
        login_page()

if __name__ == "__main__":
    main()
