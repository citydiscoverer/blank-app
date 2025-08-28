# Felix Abayomi – DCA Dashboard (Supabase + Multi-Ticker, Stable + Correct P&L)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from supabase import create_client, Client

st.set_page_config(page_title="Felix DCA (Cloud-saved)", layout="wide")
st.title("📈 Felix Abayomi – DCA Dashboard (Cloud-saved)")

# --------------------- CONFIG ---------------------
TICKERS = ["SPLG", "SCHD", "QQQM", "VOO", "SPY"]

# ------------------ SUPABASE CLIENT ----------------
@st.cache_resource
def get_sb() -> Client | None:
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["service_role_key"]
        return create_client(url, key)
    except Exception:
        st.error("❌ Supabase secrets not found. Add them in Settings → Secrets.")
        return None

sb = get_sb()

# ------------------ DB HELPERS ---------------------
def load_buys(ticker: str) -> pd.DataFrame:
    if sb is None:
        return pd.DataFrame(columns=["id","d","amount","price","created_at","ticker"])
    res = sb.table("buys").select("*").eq("ticker", ticker).order("d").execute()
    rows = res.data or []
    if not rows:
        return pd.DataFrame(columns=["id","d","amount","price","created_at","ticker"])
    df = pd.DataFrame(rows)
    df["d"] = pd.to_datetime(df["d"]).dt.date
    return df[["id","d","amount","price","created_at","ticker"]]

def load_eod(ticker: str) -> pd.DataFrame:
    if sb is None:
        return pd.DataFrame(columns=["ticker","d","eod_value","updated_at"])
    res = sb.table("eod_values").select("*").eq("ticker", ticker).order("d").execute()
    rows = res.data or []
    if not rows:
        return pd.DataFrame(columns=["ticker","d","eod_value","updated_at"])
    df = pd.DataFrame(rows)
    df["d"] = pd.to_datetime(df["d"]).dt.date
    return df[["ticker","d","eod_value","updated_at"]]

def add_buy(ticker: str, d: date, amount: float, price: float):
    if sb is None: return
    sb.table("buys").insert({"ticker": ticker, "d": str(d), "amount": amount, "price": price}).execute()

def delete_buys(ids: list[int]):
    if sb is None or not ids: return
    sb.table("buys").delete().in_("id", ids).execute()

def upsert_eod(ticker: str, d: date, eod_value: float):
    if sb is None: return
    sb.table("eod_values").upsert({"ticker": ticker, "d": str(d), "eod_value": eod_value}).execute()

def delete_eod(ticker: str, dates: list[date]):
    if sb is None or not dates: return
    sb.table("eod_values").delete().eq("ticker", ticker).in_("d", [str(d) for d in dates]).execute()

def clear_ticker(ticker: str):
    if sb is None: return
    sb.table("buys").delete().eq("ticker", ticker).execute()
    sb.table("eod_values").delete().eq("ticker", ticker).execute()

# --------------- CALCULATIONS (PATCHED) -------------
def compute_daily(buys: pd.DataFrame, eod: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a daily table with:
      date, amount, price, shares, cum_shares, cum_invested,
      portfolio_value, daily_pnl, cum_pnl, ret%
    Rules:
      - shares = amount / price (0 if price is 0)
      - value uses broker EOD if provided for that date; else estimate = cum_shares * proxy_price
      - daily P&L = Δ(value) − amount_today (market move only)
      - first day daily P&L = 0
    """
    # Handle empty case
    if buys.empty and eod.empty:
        return pd.DataFrame(columns=[
            "date","amount","price","shares","cum_shares","cum_invested",
            "portfolio_value","daily_pnl","cum_pnl","ret%"
        ])

    # Aggregate buys per day
    if not buys.empty:
        b = (buys.groupby("d", as_index=False)
                  .agg(amount=("amount","sum"), price=("price","mean")))
        b["shares"] = np.where(b["price"] > 0, b["amount"]/b["price"], 0.0)
    else:
        b = pd.DataFrame(columns=["d","amount","price","shares"])

    # EOD values
    e = eod.copy()[["d","eod_value"]] if not eod.empty else pd.DataFrame(columns=["d","eod_value"])

    # Timeline from first relevant date to today
    starts = []
    if not b.empty: starts.append(min(b["d"]))
    if not e.empty: starts.append(min(e["d"]))
    start = min(starts) if starts else date.today()
    end   = date.today()
    t = pd.DataFrame({"d": pd.date_range(start, end, freq="D").date})

    # Merge
    t = t.merge(b, how="left", on="d")
    t = t.merge(e, how="left", on="d")
    for col in ["amount","price","shares","eod_value"]:
        if col in t.columns:
            t[col] = t[col].fillna(0.0)
        else:
            t[col] = 0.0

    # Cum sums
    t["cum_invested"] = t["amount"].cumsum().round(2)
    t["cum_shares"]   = t["shares"].cumsum().round(6)

    # Value: prefer EOD; else estimate using latest non-zero price proxy
    t["proxy_price"]     = t["price"].replace(0, np.nan).ffill().bfill().fillna(1.0)
    t["est_value"]       = (t["cum_shares"] * t["proxy_price"]).round(2)
    t["portfolio_value"] = np.where(t["eod_value"] > 0, t["eod_value"], t["est_value"])

    # Daily P&L = Δvalue − today's contribution
    t["value_change"] = t["portfolio_value"].diff()
    t.loc[t.index[0], "value_change"] = 0.0  # no prior day
    t["daily_pnl"] = (t["value_change"] - t["amount"]).round(2)

    # Cumulative P&L & Return
    t["cum_pnl"] = (t["portfolio_value"] - t["cum_invested"]).round(2)
    t["ret%"]    = np.where(t["cum_invested"] > 0,
                            (t["cum_pnl"]/t["cum_invested"]*100).round(2),
                            0.0)

    # Tidy rounding
    t["amount"]          = t["amount"].round(2)
    t["price"]           = t["price"].round(4)
    t["shares"]          = t["shares"].round(6)
    t["portfolio_value"] = t["portfolio_value"].round(2)

    out = t.rename(columns={"d":"date"})
    return out[["date","amount","price","shares","cum_shares","cum_invested",
                "portfolio_value","daily_pnl","cum_pnl","ret%"]]

# ------------------- SIDEBAR ------------------------
st.sidebar.header("☁️ Cloud Save: Supabase")
ticker = st.sidebar.selectbox("Ticker", TICKERS, index=0)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("🔄 Refresh data"):
        st.experimental_rerun()
with col_sb2:
    if st.button("🧹 Clear THIS ticker"):
        clear_ticker(ticker)
        st.success(f"Cleared all data for {ticker}.")
        st.experimental_rerun()

# ---------------- LOAD DATA ONCE --------------------
buys_df = load_buys(ticker)
eod_df  = load_eod(ticker)

# ------------------ ADD BUY (FORM) -----------------
st.subheader(f"1) Add a buy for {ticker}")
with st.form("add_buy_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        d_in = st.date_input("Date", value=date.today())
    with c2:
        amt  = st.number_input("Amount ($)", min_value=0.0, value=25.0, step=1.0)
    with c3:
        px   = st.number_input("Execution price ($/share)", min_value=0.0, value=75.00, step=0.01)
    submitted = st.form_submit_button("➕ Save buy", use_container_width=True)
    if submitted:
        add_buy(ticker, d_in, float(amt), float(px))
        st.success("Buy saved to cloud.")
        st.experimental_rerun()

# ------------------ ADD EOD (FORM) -----------------
st.subheader(f"2) Add broker end-of-day (EOD) value for {ticker}")
with st.form("add_eod_form", clear_on_submit=True):
    e1, e2 = st.columns([1.2, 1])
    with e1:
        d_eod = st.date_input("EOD date", value=date.today())
    with e2:
        v_eod = st.number_input("EOD account value ($)", min_value=0.0, value=0.0, step=0.01)
    submitted_eod = st.form_submit_button("💾 Save EOD value", use_container_width=True)
    if submitted_eod:
        if v_eod <= 0:
            st.warning("Enter the exact EOD value from your brokerage (e.g., 24.90, 50.11).")
        else:
            upsert_eod(ticker, d_eod, float(v_eod))
            st.success("EOD saved to cloud.")
            st.experimental_rerun()

# ------------------ MANAGE ENTRIES -----------------
st.subheader("3) Manage entries")
colL, colR = st.columns(2)
with colL:
    st.markdown("**Buys (cloud)**")
    if buys_df.empty:
        st.info("No buys yet.")
    else:
        st.dataframe(buys_df, use_container_width=True)
        ids = st.multiselect("Delete buys by ID", buys_df["id"].tolist())
        if st.button("🗑 Delete selected buys"):
            delete_buys(ids)
            st.success("Deleted.")
            st.experimental_rerun()

with colR:
    st.markdown("**EOD values (cloud)**")
    if eod_df.empty:
        st.info("No EOD values yet.")
    else:
        st.dataframe(eod_df, use_container_width=True)
        sels = st.multiselect("Delete EOD rows (by date)", eod_df["d"].tolist())
        if st.button("🗑 Delete selected EOD"):
            delete_eod(ticker, sels)
            st.success("Deleted.")
            st.experimental_rerun()

# --------------- METRICS & CHARTS ------------------
st.subheader(f"4) Metrics & charts — {ticker}")
daily = compute_daily(buys_df, eod_df)

if daily.empty:
    st.info("Add a buy to see analytics.")
else:
    latest = daily.iloc[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cost basis", f"${latest['cum_invested']:,.2f}")
    m2.metric("Shares", f"{latest['cum_shares']:.6f}")
    m3.metric("Portfolio value", f"${latest['portfolio_value']:,.2f}")
    m4.metric("Cumulative P&L", f"${latest['cum_pnl']:,.2f}")
    m5.metric("Return", f"{latest['ret%']:.2f}%")

    c1, c2 = st.columns(2)
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
