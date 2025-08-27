import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date

st.title("📈 Felix Abayomi – SPLG DCA Dashboard")

# --- Input today's buy ---
st.subheader("Add today's buy")
today = date.today().strftime("%Y-%m-%d")

amount = st.number_input("Amount invested ($)", min_value=1.0, value=25.0, step=1.0)
price = st.number_input("Execution price ($/share)", min_value=1.0, value=75.0, step=0.01)

if "log" not in st.session_state:
    st.session_state.log = pd.DataFrame(columns=["Date", "Amount", "Price", "Shares"])

if st.button("➕ Add entry"):
    shares = amount / price
    new_entry = pd.DataFrame([[today, amount, price, shares]], 
                             columns=["Date", "Amount", "Price", "Shares"])
    st.session_state.log = pd.concat([st.session_state.log, new_entry], ignore_index=True)

# --- Display log ---
if not st.session_state.log.empty:
    df = st.session_state.log.copy()

    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Cumulative calculations
    df["Cum_Shares"] = df["Shares"].cumsum()
    df["Cum_Invested"] = df["Amount"].cumsum()

    # Get latest SPLG price from Yahoo Finance
    try:
        latest_price = yf.Ticker("SPLG").history(period="1d")["Close"].iloc[-1]
    except:
        latest_price = price  # fallback if API fails

    df["Value"] = df["Cum_Shares"] * latest_price
    df["Daily Gain/Loss"] = df["Value"].diff().fillna(0)
    df["Total Gain/Loss"] = df["Value"] - df["Cum_Invested"]

    st.subheader("📊 Portfolio Log")
    st.dataframe(df)

    st.subheader("📈 Performance Over Time")
    st.line_chart(df.set_index("Date")[["Value", "Cum_Invested"]])

    st.metric("Portfolio Value", f"${df['Value'].iloc[-1]:,.2f}")
    st.metric("Total Gain/Loss", f"${df['Total Gain/Loss'].iloc[-1]:,.2f}")
else:
    st.info("No entries yet. Add your first $25 investment above.")
