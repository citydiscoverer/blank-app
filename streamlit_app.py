# Felix Abayomi – SPLG DCA (Supabase-persisted)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from supabase import create_client, Client

st.set_page_config(page_title="Felix DCA Dashboard", layout="wide")
st.title("📈 Felix Abayomi – SPLG DCA Dashboard (Cloud-saved)")

# --------- Supabase client ----------
@st.cache_resource
def get_sb() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["service_role_key"]  # server side only
    return create_client(url, key)

sb = get_sb()

# --------- DB helpers ----------
def load_buys_df() -> pd.DataFrame:
    res = sb.table("buys").select("*").order("d").execute()
    rows = res.data or []
    if not rows: 
        return pd.DataFrame(columns=["id","d","amount","price","created_at"])
    df = pd.DataFrame(rows)
    df["d"] = pd.to_datetime(df["d"]).dt.date
    return df[["id","d","amount","price","created_at"]]

def load_eod_df() -> pd.DataFrame:
    res = sb.table("eod_values").select("*").order("d").execute()
    rows = res.data or []
    if not rows:
        return pd.DataFrame(columns=["d","eod_value","updated_at"])
    df = pd.DataFrame(rows)
    df["d"] = pd.to_datetime(df["d"]).dt.date
    return df[["d","eod_value","updated_at"]]

def add_buy(d: date, amount: float, price: float):
    sb.table("buys").insert({"d": str(d), "amount": amount, "price": price}).execute()

def delete_buys(ids: list[int]):
    if ids:
        sb.table("buys").delete().in_("id", ids).execute()

def upsert_eod(d: date, eod_value: float):
    # upsert by primary key (d)
    sb.table("eod_values").upsert({"d": str(d), "eod_value": eod_value}).execute()

def delete_eod(dates: list[date]):
    if dates:
        dates_str = [str(d) for d in dates]
        sb.table("eod_values").delete().in_("d", dates_str).execute()

# --------- Calculations ----------
def compute_daily(buys: pd.DataFrame, eod: pd.DataFrame):
    if buys.empty and eod.empty:
        return pd.DataFrame(columns=[
            "date","amount","price","shares","cum_shares","cum_invested",
            "portfolio_value","daily_pnl","cum_pnl","ret%"
        ])

    # Aggregate buys by day
    b = buys.copy()
    if not b.empty:
        b = (b.groupby("d", as_index=False)
               .agg(amount=("amount","sum"), price=("price","mean")))
        b["shares"] = b["amount"] / b["price"]
    else:
        b = pd.DataFrame(columns=["d","amount","price","shares"])

    # Join EOD values
    e = eod.copy().rename(columns={"d":"d", "eod_value":"eod_value"})

    # Build continuous timeline from min date to today
    dates = []
    if not b.empty: dates.append(min(b["d"]))
    if not e.empty: dates.append(min(e["d"]))
    if not dates:
        start = date.today()
    else:
        start = min(dates)
    end = date.today()
    timeline = pd.DataFrame({"d": pd.date_range(start, end, freq="D").date})

    # Merge buys and eod to timeline
    t = timeline.merge(b, how="left", left_on="d", right_on="d")
    t = t.merge(e, how="left", left_on="d", right_on="d")
    t[["amount","price","shares","eod_value"]] = t[["amount","price","shares","eod_value"]].fillna(0.0)

    # Cumulated contributions and shares
    t["cum_invested"] = t["amount"].cumsum()
    t["cum_shares"] = t["shares"].cumsum()

    # Portfolio value: prefer broker EOD if present (>0), else estimate using that day's average price if any,
    # or last nonzero price (carry forward estimate)
    # For simplicity, use latest known average buy price as proxy if no EOD:
    # We'll create a rolling proxy price = last non-zero 'price' seen.
    t["proxy_price"] = t["price"].replace(0, np.nan).ffill().fillna(method="bfill").fillna(1.0)
    t["est_value"] = (t["cum_shares"] * t["proxy_price"]).round(2)
    t["portfolio_value"] = np.where(t["eod_value"] > 0, t["eod_value"], t["est_value"])

    # Daily P&L = value change – new contributions
    t["value_change"] = t["portfolio_value"].diff()
    t.loc[t.index[0], "value_change"] = 0.0
    t["daily_pnl"] = (t["value_change"] - t["amount"]).round(2)

    # Cumulative P&L
    t["cum_pnl"] = (t["portfolio_value"] - t["cum_invested"]).round(2)
    t["ret%"] = np.where(t["cum_invested"]>0, (t["cum_pnl"]/t["cum_invested"]*100).round(2), 0.0)

    # Pretty output
    out = t.rename(columns={"d":"date"})
    out["shares"] = out["shares"].round(6)
    out["cum_shares"] = out["cum_shares"].round(6)
    keep = ["date","amount","price","shares","cum_shares","cum_invested",
            "portfolio_value","daily_pnl","cum_pnl","ret%"]
    return out[keep]

# --------- UI: Data entry ----------
with st.sidebar:
    st.header("☁️ Cloud Save: Supabase")
    st.caption("Your data auto-saves to Supabase. Nothing is lost on refresh.")

# Load current data
buys_df = load_buys_df()
eod_df = load_eod_df()

st.subheader("1) Add a buy")
c1,c2,c3,c4 = st.columns([1.2,1,1,0.8])
with c1:
    d_in = st.date_input("Date", value=date.today())
with c2:
    amt = st.number_input("Amount ($)", min_value=0.0, value=25.0, step=1.0)
with c3:
    px = st.number_input("Execution price ($/share)", min_value=0.0, value=75.00, step=0.01)
with c4:
    if st.button("➕ Add buy", use_container_width=True, type="primary"):
        add_buy(d_in, float(amt), float(px))
        st.success("Buy saved to cloud.")
        st.rerun()

st.subheader("2) Add broker end-of-day (EOD) value")
e1,e2 = st.columns([1.2,1])
with e1:
    d_eod = st.date_input("EOD date", value=date.today(), key="eod_date")
with e2:
    v_eod = st.number_input("EOD account value ($)", min_value=0.0, value=0.0, step=0.01)
if st.button("💾 Save EOD value"):
    if v_eod <= 0:
        st.warning("Enter the exact value from J.P. Morgan (e.g., 24.90, 50.11).")
    else:
        upsert_eod(d_eod, float(v_eod))
        st.success("EOD value saved to cloud.")
        st.rerun()

# --------- Manage entries ----------
st.subheader("3) Manage entries")
colL, colR = st.columns(2)
with colL:
    st.markdown("**Buys**")
    if buys_df.empty:
        st.info("No buys yet.")
    else:
        st.dataframe(buys_df, use_container_width=True)
        ids = st.multiselect("Delete buys by ID", buys_df["id"].tolist())
        if st.button("🗑 Delete selected buys"):
            delete_buys(ids)
            st.success("Deleted.")
            st.rerun()

with colR:
    st.markdown("**EOD values**")
    if eod_df.empty:
        st.info("No EOD values yet.")
    else:
        st.dataframe(eod_df, use_container_width=True)
        sel_dates = st.multiselect("Delete EOD rows (by date)", [r for r in eod_df["d"]])
        if st.button("🗑 Delete selected EOD"):
            delete_eod(sel_dates)
            st.success("Deleted.")
            st.rerun()

# --------- Metrics & Charts ----------
st.subheader("4) Portfolio metrics & charts")
daily = compute_daily(buys_df, eod_df)

if daily.empty:
    st.info("Add a buy to see analytics.")
else:
    latest = daily.iloc[-1]
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Cost basis", f"${latest['cum_invested']:,.2f}")
    m2.metric("Shares", f"{latest['cum_shares']:.6f}")
    m3.metric("Portfolio value", f"${latest['portfolio_value']:,.2f}")
    m4.metric("Cumulative P&L", f"${latest['cum_pnl']:,.2f}")
    m5.metric("Return", f"{latest['ret%']:.2f}%")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Portfolio value over time**")
        st.line_chart(daily.set_index("date")[["portfolio_value"]])
    with c2:
        st.markdown("**Daily P&L ($)**")
        st.bar_chart(daily.set_index("date")[["daily_pnl"]])

    st.markdown("**Last 30 days**")
    view = daily.tail(30).copy()
    view["date"] = view["date"].astype(str)
    st.dataframe(view, use_container_width=True)
