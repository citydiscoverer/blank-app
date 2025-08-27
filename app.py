import streamlit as st
import pandas as pd

st.title("📊 Investment Tracker Dashboard")

# Example portfolio data
data = {
    "Date": ["2025-08-25", "2025-08-26"],
    "Portfolio Value ($)": [24.90, 50.11],
    "Daily Gain/Loss ($)": [-0.10, +0.20],
    "Total Gain/Loss ($)": [-0.10, +0.11]
}

df = pd.DataFrame(data)

st.subheader("Portfolio Performance")
st.line_chart(df.set_index("Date")["Portfolio Value ($)"])

st.subheader("Daily Summary")
st.dataframe(df)
