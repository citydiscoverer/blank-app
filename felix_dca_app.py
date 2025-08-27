# felix_dca_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

st.set_page_config(page_title="Felix – SPLG DCA Dashboard", layout="wide")

# ---------- helpers
def _to_date(x):
    if isinstance(x, (datetime, pd.Timestamp)):
        return pd.to_datetime(x).date()
    if isinstance(x, date):
        return x
    return pd.to_datetime(str(x)).date()

def init_state():
    if "buys" not in st.session_state:
        st.session_state.buys = pd.DataFrame(columns=["date","amount","price","shares"])
    if "eod" not in st.session_state:
        # broker end-of-day total account values you type in
        st.session_state.eod = pd.DataFrame(columns=["date","eod_value"])

def add_buy_row(d, amount, price):
    d = _to_date(d)
    shares = 0 if price == 0 else round(amount/price, 6)
    row = pd.DataFrame([{"date": d, "amount": float(amount), "price": float(price), "shares": shares}])
    st.session_state.buys = pd.concat([st.session_state.buys, row], ignore_index=True)
    st.success("Buy added.")

def add_eod_row(d, eod_value):
    d = _to_date(d)
    row = pd.DataFrame([{"date": d, "eod_value": float(eod_value)}])
    # keep latest if duplicate date
    df = pd.concat([st.session_state.eod, row], ignore_index=True)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    st.session_state.eod = df.reset_index(drop=True)
    st.success("EOD value saved.")

def compute_daily(buys: pd.DataFrame, eod: pd.DataFrame):
    if buys.empty:
        return pd.DataFrame(columns=[
            "date","amount","price","shares","cum_shares","cum_invested",
            "portfolio_value","daily_pnl","cum_pnl","ret%"
        ])

    df = buys.copy()
    df["date"] = df["date"].apply(_to_date)
    df = df.sort_values("date")

    # cumulative contributions & shares
    df["cum_invested"] = df["amount"].cumsum()
    df["cum_shares"]   = df["shares"].cumsum()

    # merge broker EOD values (your app screenshots numbers)
    e = eod.copy()
    if not e.empty:
        e["date"] = e["date"].apply(_to_date)
    df = df.merge(e, on="date", how="left")

    # portfolio_value: prefer broker EOD if provided, else estimate (price * cum_shares)
    df["est_value"] = (df["price"] * df["cum_shares"]).round(2)
    df["portfolio_value"] = df["eod_value"].fillna(df["est_value"])

    # cumulative P&L (value minus total money in)
    df["cum_pnl"] = (df["portfolio_value"] - df["cum_invested"]).round(2)

    # daily P&L: true market move = ΔValue − new contribution that day
    df["value_change"] = df["portfolio_value"].diff()
    df["contrib"] = df["amount"]
    df.loc[df.index[0], "value_change"] = np.nan  # no prior day
    df["daily_pnl"] = (df["value_change"] - df["contrib"].fillna(0)).round(2)
    df.loc[df.index[0], "daily_pnl"] = 0.00

    # return %
    df["ret%"] = np.where(df["cum_invested"]>0,
                          (df["cum_pnl"]/df["cum_invested"]*100).round(2),
                          0.0)

    keep_cols = ["date","amount","price","shares","cum_shares","cum_invested",
                 "portfolio_value","daily_pnl","cum_pnl","ret%"]
    return df[keep_cols]

def fmt_money(x):
    try:
        return f"${x:,.2f}"
    except:
        return x

# ---------- UI
init_state()
st.title("Felix Abayomi – SPLG DCA Dashboard")

with st.expander("Upload/Download data", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Upload **buys** CSV with columns: date,amount,price")
        up = st.file_uploader("Upload buys CSV", type=["csv"], key="up_buys")
        if up:
            df = pd.read_csv(up)
            df["date"] = df["date"].apply(_to_date)
            # compute shares from amount/price
            df["shares"] = (df["amount"]/df["price"]).round(6)
            st.session_state.buys = df[["date","amount","price","shares"]]
            st.success("Buys loaded.")
    with c2:
        st.caption("Upload **broker EOD** CSV with columns: date,eod_value")
        up2 = st.file_uploader("Upload broker EOD CSV", type=["csv"], key="up_eod")
        if up2:
            df = pd.read_csv(up2)
            df["date"] = df["date"].apply(_to_date)
            st.session_state.eod = df[["date","eod_value"]]
            st.success("Broker EOD loaded.")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button(
            "Download buys CSV",
            st.session_state.buys.to_csv(index=False).encode("utf-8"),
            "buys.csv",
            "text/csv")
    with dc2:
        st.download_button(
            "Download broker EOD CSV",
            st.session_state.eod.to_csv(index=False).encode("utf-8"),
            "broker_eod.csv",
            "text/csv")

st.markdown("### 1) Add a buy")
dcol, acol, pcol, bcol = st.columns([1.2,1,1,1])
with dcol:
    d_in = st.date_input("Date", value=date.today(), format="YYYY/MM/DD")
with acol:
    amt_in = st.number_input("Amount ($)", value=25.00, min_value=0.0, step=1.0)
with pcol:
    price_in = st.number_input("Execution price ($/share)", value=75.00, min_value=0.0, step=0.01)
with bcol:
    st.write("")
    if st.button("➕ Add entry", use_container_width=True):
        add_buy_row(d_in, amt_in, price_in)

st.markdown("### 2) Manage buys")
if st.session_state.buys.empty:
    st.info("No buys yet. Add your first $25 buy above.")
else:
    df_b = st.session_state.buys.sort_values("date").reset_index(drop=True)
    st.dataframe(df_b, use_container_width=True, hide_index=True)
    # delete UI
    options = [f"{i}. {r.date} – ${r.amount} @ ${r.price}" for i, r in df_b.iterrows()]
    sel = st.multiselect("Select rows to delete", options, default=[])
    if st.button("🗑️ Delete selected"):
        idx_to_drop = [int(s.split(".")[0]) for s in sel]
        st.session_state.buys = df_b.drop(index=idx_to_drop).reset_index(drop=True)
        st.rerun()

st.markdown("### 3) Enter broker end-of-day (EOD) account value")
ec1, ec2 = st.columns([1,1])
with ec1:
    eod_date = st.date_input("EOD date", value=date.today(), format="YYYY/MM/DD", key="eod_date")
with ec2:
    eod_val = st.number_input("EOD account value ($)", value=0.0, min_value=0.0, step=0.01, key="eod_val")
if st.button("💾 Save EOD value"):
    if eod_val <= 0:
        st.warning("Please enter the actual account value shown in JP Morgan (e.g., 24.90, 50.11).")
    else:
        add_eod_row(eod_date, eod_val)

# ---------- metrics
daily = compute_daily(st.session_state.buys, st.session_state.eod)

st.markdown("### 4) Portfolio metrics")
if daily.empty:
    st.info("Add at least one buy to see metrics.")
else:
    latest = daily.iloc[-1]
    cost_basis = latest["cum_invested"]
    shares = latest["cum_shares"]
    value = latest["portfolio_value"]
    cum_pnl = latest["cum_pnl"]
    ret_pct = latest["ret%"]
    # daily P&L (last row)
    day_pnl = latest["daily_pnl"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cost basis", fmt_money(cost_basis))
    m2.metric("Shares", f"{shares:.6f}")
    m3.metric("Portfolio value", fmt_money(value),
              delta=fmt_money(day_pnl) if not pd.isna(day_pnl) else None)
    m4.metric("Cumulative P&L", fmt_money(cum_pnl))
    m5.metric("Return", f"{ret_pct:.2f}%")

    st.caption("Daily P&L is the market move only (ΔValue minus new contributions).")

    # charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Portfolio value over time")
        st.line_chart(daily.set_index("date")[["portfolio_value"]])
    with c2:
        st.subheader("Daily P&L ($)")
        st.bar_chart(daily.set_index("date")[["daily_pnl"]])

    st.subheader("Last 30 days (clean)")
    last30 = daily.tail(30).copy()
    last30["date"] = last30["date"].astype(str)
    show_cols = ["date","amount","price","shares","cum_shares","cum_invested",
                 "portfolio_value","daily_pnl","cum_pnl","ret%"]
    st.dataframe(last30[show_cols], use_container_width=True, hide_index=True)
