import streamlit as st
import pandas as pd
from datetime import date

st.set_page_config(page_title="Felix DCA Dashboard", layout="centered")
st.title("📈 Felix Abayomi – SPLG DCA Dashboard")

st.write(
    "Log your daily $25 buys (or any amount), track shares, cost basis, value, and download your CSV."
)

# --- Load data (user can upload a CSV) ---
st.subheader("1) Load or start a log")
uploaded = st.file_uploader("Upload your CSV (columns: date,amount,price)", type="csv")

if "df" not in st.session_state:
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=["date"])
    else:
        # Start empty
        df = pd.DataFrame(columns=["date","amount","price"])
else:
    df = st.session_state.df

# --- Add a new buy ---
st.subheader("2) Add today's buy")
col1, col2, col3 = st.columns(3)
with col1:
    d = st.date_input("Date", value=date.today())
with col2:
    amt = st.number_input("Amount ($)", min_value=0.0, value=25.0, step=1.0)
with col3:
    px = st.number_input("Execution price ($/share)", min_value=0.0, value=75.0, step=0.01)

if st.button("➕ Add entry"):
    new_row = {"date": pd.to_datetime(d), "amount": float(amt), "price": float(px)}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).sort_values("date")
    st.success("Added.")
    st.session_state.df = df

# --- Calculations ---
if not df.empty:
    calc = df.copy()
    calc["shares"] = calc["amount"] / calc["price"]
    calc["cum_shares"] = calc["shares"].cumsum()
    cost_basis = calc["amount"].sum()

    # Let user set current price to see live value
    st.subheader("3) Portfolio metrics")
    current_price = st.number_input("Current SPLG price ($)", value=75.0, step=0.01)
    portfolio_value = calc["cum_shares"].iloc[-1] * current_price
    gain = portfolio_value - cost_basis
    ret = (gain / cost_basis) * 100 if cost_basis else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cost basis", f"${cost_basis:,.2f}")
    c2.metric("Shares", f"{calc['cum_shares'].iloc[-1]:.4f}")
    c3.metric("Portfolio value", f"${portfolio_value:,.2f}")
    c4.metric("Gain / Return", f"${gain:,.2f}", f"{ret:,.2f}%")

    st.subheader("4) Log & charts")
    st.dataframe(calc[["date","amount","price","shares","cum_shares"]], use_container_width=True)

    # Simple charts
    v = pd.DataFrame({
        "date": calc["date"],
        "Portfolio Value": calc["cum_shares"] * current_price,
        "Cumulative Shares": calc["cum_shares"]
    }).set_index("date")
    st.line_chart(v[["Portfolio Value"]])
    st.line_chart(v[["Cumulative Shares"]])

    # Download CSV
    st.subheader("5) Save your data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name="felix_dca_log.csv", mime="text/csv")
else:
    st.info("No entries yet. Upload a CSV or add your first $25 buy above.")
