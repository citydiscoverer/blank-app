# Felix Abayomi – SPLG DCA Dashboard (single-file version)
import streamlit as st
import pandas as pd
from datetime import date

st.set_page_config(page_title="Felix DCA Dashboard", layout="wide")
st.title("📈 Felix Abayomi – SPLG DCA Dashboard")

st.caption(
    "Log your daily buys, optionally add end-of-day **account value** from J.P. Morgan, "
    "and track true P&L. Download/upload CSV anytime."
)

# ---------- helpers ----------
CSV_COLS = ["date", "amount", "price", "eod_value"]

def init_state():
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=CSV_COLS)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.floor("D")
    num_cols = ["amount", "price", "eod_value"]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    return out

def compute_metrics(df: pd.DataFrame, current_price: float) -> dict:
    """Returns dict with calc tables and headline metrics.
       If eod_value present for latest date, use it for value and P&L;
       otherwise fall back to mark-to-market using current_price.
    """
    if df.empty:
        return dict(df=df, calc=None, value=0.0, cost=0.0, pnl=0.0, pnl_pct=0.0)

    calc = df.copy()
    calc["shares"] = calc["amount"] / calc["price"]
    calc["cum_shares"] = calc["shares"].cumsum()
    calc["cum_invested"] = calc["amount"].cumsum()

    # Daily P&L using provided end-of-day account value (if present)
    calc["daily_pnl"] = pd.NA
    calc["value"] = pd.NA

    prev_eod = 0.0
    for i, row in calc.iterrows():
        # account value: if user provided eod_value use it; else mark-to-market for that row
        if pd.notna(row.get("eod_value")):
            value_i = float(row["eod_value"])
        else:
            value_i = float(calc.loc[i, "cum_shares"] * current_price)

        calc.loc[i, "value"] = value_i

        # daily pnl = today's end value - (yesterday end value + today's new cash)
        amt_today = float(row["amount"]) if pd.notna(row["amount"]) else 0.0
        daily = value_i - (prev_eod + amt_today)
        calc.loc[i, "daily_pnl"] = daily
        prev_eod = value_i

    cost_basis = float(calc["cum_invested"].iloc[-1])
    portfolio_value = float(calc["value"].iloc[-1])
    pnl_abs = portfolio_value - cost_basis
    pnl_pct = (pnl_abs / cost_basis * 100.0) if cost_basis else 0.0

    return dict(
        df=df, calc=calc, value=portfolio_value, cost=cost_basis, pnl=pnl_abs, pnl_pct=pnl_pct
    )

# ---------- UI ----------
init_state()

with st.sidebar:
    st.header("📦 Data & Settings")
    up = st.file_uploader("Upload CSV (columns: date,amount,price,eod_value)", type="csv")
    if up is not None:
        tmp = pd.read_csv(up)
        st.session_state.df = clean_df(tmp)
        st.success("Loaded CSV.")

    if not st.session_state.df.empty:
        dl = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", dl, file_name="felix_dca_log.csv", mime="text/csv")

    if st.button("🧹 Clear ALL data", type="secondary"):
        st.session_state.df = pd.DataFrame(columns=CSV_COLS)
        st.experimental_rerun()

# add a buy
st.subheader("1) Add a buy")
c1, c2, c3 = st.columns([1,1,1])
with c1:
    d_in = st.date_input("Date", value=date.today())
with c2:
    amt = st.number_input("Amount ($)", min_value=0.0, value=25.0, step=1.0)
with c3:
    px = st.number_input("Execution price ($/share)", min_value=0.0, value=75.00, step=0.01)

# optional end-of-day account value for **that same date**
eod = st.number_input(
    "Optional: End-of-day account value (from J.P. Morgan) for this date", min_value=0.0, value=0.0, step=0.01
)

if st.button("➕ Add entry"):
    new = pd.DataFrame([{
        "date": pd.to_datetime(d_in),
        "amount": float(amt),
        "price": float(px),
        "eod_value": float(eod) if eod > 0 else pd.NA
    }])
    st.session_state.df = clean_df(pd.concat([st.session_state.df, new], ignore_index=True))
    st.success("Added entry.")

# manage entries (delete)
st.subheader("2) Manage entries")
df_show = st.session_state.df.copy()
st.dataframe(df_show, use_container_width=True, hide_index=True)
if not df_show.empty:
    idxs = st.multiselect("Select rows to delete", options=df_show.index.tolist(), format_func=lambda i: str(df_show.loc[i,"date"].date()))
    if st.button("🗑️ Delete selected") and idxs:
        st.session_state.df = df_show.drop(index=idxs).reset_index(drop=True)
        st.success("Deleted.")
        st.experimental_rerun()

# portfolio metrics
st.subheader("3) Portfolio metrics & charts")
cur_price = st.number_input("If no EOD value provided, use this current SPLG price ($):", value=75.00, step=0.01)
res = compute_metrics(st.session_state.df, current_price=cur_price)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cost basis", f"${res['cost']:,.2f}")
c2.metric("Shares", f"{(res['calc']['cum_shares'].iloc[-1] if res['calc'] is not None else 0):.4f}")
c3.metric("Portfolio value", f"${res['value']:,.2f}")
trend = "↑" if res["pnl"] >= 0 else "↓"
c4.metric("Cumulative P&L", f"${res['pnl']:,.2f}", f"{res['pnl_pct']:.2f}%")

# charts
if res["calc"] is not None and not res["calc"].empty:
    left, right = st.columns(2)
    with left:
        st.write("**Portfolio value over time**")
        st.line_chart(res["calc"][["date","value"]].set_index("date"))
    with right:
        st.write("**Daily gain/loss ($)**")
        st.bar_chart(res["calc"][["date","daily_pnl"]].set_index("date"))

    st.write("**Detail (last 30 days)**")
    tail_cols = ["date","amount","price","eod_value","shares","cum_shares","cum_invested","value","daily_pnl"]
    st.dataframe(res["calc"][tail_cols].tail(30).reset_index(drop=True), use_container_width=True)
else:
    st.info("No entries yet.")
