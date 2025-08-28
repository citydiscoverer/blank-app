# Felix Abayomi – DCA Dashboard (Supabase + Multi-Ticker, Stable + Correct P&L)
# streamlit_app.py
import os, math
import streamlit as st
import pandas as pd
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, getcontext

# ---------- Precision ----------
getcontext().prec = 28
SHARE_Q = Decimal("0.00001")   # 5dp shares
CENT    = Decimal("0.01")
def D(x): return Decimal(str(x)) if x is not None and str(x) != "" else Decimal("0")

# ---------- Page ----------
st.set_page_config(page_title="Felix Legacy – Multi-Ticker DCA", layout="wide")
st.title("📈 Felix Abayomi – Multi-Ticker DCA (Buys + EOD)")

# ======================================================
#            STORAGE: CSV first, Supabase optional
# ======================================================
BUYS_CSV = "buys.csv"   # id, ticker, trade_date, amount, price
EOD_CSV  = "eod.csv"    # id, ticker, eod_date, eod_value

def _ensure_csv(path, cols):
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path, index=False)

_ensure_csv(BUYS_CSV, ["id","ticker","trade_date","amount","price"])
_ensure_csv(EOD_CSV,  ["id","ticker","eod_date","eod_value"])

def load_buys_csv(ticker=None):
    df = pd.read_csv(BUYS_CSV)
    if df.empty: return df
    if ticker: df = df[df["ticker"]==ticker]
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df

def save_buys_csv(df): df.to_csv(BUYS_CSV, index=False)

def insert_buy_csv(ticker, d:date, amount:Decimal, price:Decimal):
    df = load_buys_csv()
    new_id = (df["id"].max()+1) if not df.empty else 1
    row = {"id":new_id,"ticker":ticker,"trade_date":d.isoformat(),
           "amount":float(amount),"price":float(price)}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_buys_csv(df)

def delete_buys_ids_csv(ids):
    if not ids: return
    df = load_buys_csv()
    df = df[~df["id"].isin(ids)]
    save_buys_csv(df)

def clear_ticker_csv(ticker):
    b = load_buys_csv();  b = b[b["ticker"]!=ticker];  save_buys_csv(b)
    e = load_eod_csv();   e = e[e["ticker"]!=ticker];  save_eod_csv(e)

def load_eod_csv(ticker=None):
    df = pd.read_csv(EOD_CSV)
    if df.empty: return df
    if ticker: df = df[df["ticker"]==ticker]
    df["eod_date"] = pd.to_datetime(df["eod_date"]).dt.date
    return df

def save_eod_csv(df): df.to_csv(EOD_CSV, index=False)

def upsert_eod_csv(ticker, d:date, value:Decimal):
    df = load_eod_csv()
    mask = (df["ticker"]==ticker) & (pd.to_datetime(df["eod_date"]).dt.date==d)
    if mask.any():
        df.loc[mask,"eod_value"] = float(value)
    else:
        new_id = (df["id"].max()+1) if not df.empty else 1
        row = {"id":new_id,"ticker":ticker,"eod_date":d.isoformat(),"eod_value":float(value)}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_eod_csv(df)

def delete_eod_rows_csv(ticker, dates:list[date]):
    if not dates: return
    df = load_eod_csv()
    keep = ~((df["ticker"]==ticker) & (df["eod_date"].isin([d.isoformat() for d in dates])))
    save_eod_csv(df[keep])

# ---------- Optional Supabase (won’t crash if missing) ----------
def get_supabase_client():
    try:
        from supabase import create_client
    except Exception:
        return None
    s = st.secrets.get("supabase", {}) if hasattr(st,"secrets") else {}
    url = s.get("url") or os.environ.get("SUPABASE_URL")
    key = s.get("service_role_key") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key: return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None

sb = get_supabase_client()
CONNECTED = sb is not None

# If you want cloud persistence today, set both secrets:
# [supabase]
# url="https://YOUR-REF.supabase.co"
# service_role_key="YOUR_SERVICE_ROLE_KEY"
# And create tables with SQL at the bottom of this message.

# ======================================================
#                   CALCULATIONS
# ======================================================
def compute_daily(buys_df: pd.DataFrame, eod_df: pd.DataFrame) -> pd.DataFrame:
    """
    date, amount, price, shares, cum_shares, cum_invested, portfolio_value, daily_pnl, cum_pnl, ret%
    Uses EOD when present; otherwise estimates via last non-zero price × cum_shares.
    Daily P&L isolates market move: Δ(value) − contribution_today. First day P&L = 0.
    """
    if (buys_df is None or buys_df.empty) and (eod_df is None or eod_df.empty):
        return pd.DataFrame(columns=["date","amount","price","shares","cum_shares","cum_invested",
                                     "portfolio_value","daily_pnl","cum_pnl","ret%"])

    b = buys_df.copy()
    b = b.rename(columns={"trade_date":"date"}) if "trade_date" in b.columns else b
    b["date"] = pd.to_datetime(b["date"]).dt.date if not b.empty else b
    e = eod_df.copy()
    if not e.empty:
        if "date" not in e.columns:
            e = e.rename(columns={"eod_date":"date"})
        e["date"] = pd.to_datetime(e["date"]).dt.date

    # Aggregate buys per day
    if not b.empty:
        b_agg = b.groupby("date", as_index=False).agg(amount=("amount","sum"), price=("price","mean"))
        b_agg["shares"] = (b_agg["amount"].fillna(0.0) / b_agg["price"].replace(0, pd.NA)).fillna(0.0).round(5)
    else:
        b_agg = pd.DataFrame(columns=["date","amount","price","shares"])

    # Build timeline
    starts = []
    if not b_agg.empty: starts.append(b_agg["date"].min())
    if not e.empty:     starts.append(e["date"].min())
    start = min(starts) if starts else date.today()
    end   = max([date.today()] + ([b_agg["date"].max()] if not b_agg.empty else []) + ([e["date"].max()] if not e.empty else []))
    t = pd.DataFrame({"date": pd.date_range(start, end, freq="D").date})

    # Merge
    t = t.merge(b_agg, how="left", on="date")
    t = t.merge(e[["date","eod_value"]] if not e.empty else pd.DataFrame(columns=["date","eod_value"]),
                how="left", on="date")
    for col in ["amount","price","shares","eod_value"]:
        if col not in t.columns: t[col] = 0.0
        t[col] = t[col].fillna(0.0)

    # Running totals
    t["cum_invested"] = t["amount"].cumsum().round(2)
    t["cum_shares"]   = t["shares"].cumsum().round(5)

    # Estimate when no EOD
    proxy = t["price"].replace(0, pd.NA).ffill().bfill().fillna(1.0)
    est_value = (t["cum_shares"] * proxy).round(2)
    t["portfolio_value"] = t["eod_value"].where(t["eod_value"] > 0, est_value)

    # P&L
    change = t["portfolio_value"].diff().fillna(0.0)
    t["daily_pnl"] = (change - t["amount"]).round(2)
    if len(t) > 0: t.loc[t.index[0], "daily_pnl"] = 0.00

    t["cum_pnl"] = (t["portfolio_value"] - t["cum_invested"]).round(2)
    t["ret%"] = (t["cum_pnl"]/t["cum_invested"]*100).replace([pd.NA, float("inf"), -float("inf")], 0).fillna(0).round(2)

    return t[["date","amount","price","shares","cum_shares","cum_invested","portfolio_value","daily_pnl","cum_pnl","ret%"]]

def metrics_from_daily(daily: pd.DataFrame):
    if daily.empty:
        return {"cost":0.0,"sh":0.0,"val":0.0,"pnl":0.0,"ret":0.0}
    last = daily.iloc[-1]
    return {"cost":float(last["cum_invested"]),
            "sh":float(last["cum_shares"]),
            "val":float(last["portfolio_value"]),
            "pnl":float(last["cum_pnl"]),
            "ret":float(last["ret%"])}

# ======================================================
#                          UI
# ======================================================
with st.sidebar:
    st.subheader("💾 Storage")
    if CONNECTED:
        st.success("Supabase configured (cloud save coming next).")
    else:
        st.info("Using local CSV files (buys.csv, eod.csv).")
    st.download_button("⬇️ Export buys.csv", data=open(BUYS_CSV,"rb"), file_name="buys.csv")
    st.download_button("⬇️ Export eod.csv",  data=open(EOD_CSV,"rb"),  file_name="eod.csv")

# Ticker choices
all_b = load_buys_csv()
all_e = load_eod_csv()
tickers = sorted(set(all_b["ticker"].tolist() + all_e["ticker"].tolist()) | {"SPLG","SCHD","VOO","SPY"})
ticker = st.selectbox("Ticker", options=tickers, index=tickers.index("SPLG") if "SPLG" in tickers else 0)

# 1) Add buy
st.markdown("### 1) Add a buy")
c1,c2,c3,c4 = st.columns([1.2,1,1,1.2])
with c1: d = st.date_input("Date", value=date.today())
with c2: amt = D(st.number_input("Amount ($)", value=25.00, min_value=0.0, step=1.0))
with c3: px  = D(st.number_input("Execution price ($/share)", value=75.00, min_value=0.0, step=0.01))
with c4:
    st.write(" ")
    if st.button("➕ Add entry", use_container_width=True):
        insert_buy_csv(ticker, d, amt, px); st.success("Added."); st.rerun()

# 2) Add EOD
st.markdown("### 2) Add broker end-of-day (EOD) value")
e1,e2,e3 = st.columns([1.2,1,1])
with e1: d_eod = st.date_input("EOD date", value=date.today(), key="eod_date")
with e2: v_eod = D(st.number_input("EOD account value ($)", min_value=0.0, value=0.0, step=0.01))
with e3:
    st.write(" ")
    if st.button("💾 Save EOD value", use_container_width=True):
        if v_eod <= 0:
            st.warning("Enter your broker's exact end-of-day value (e.g., 50.11).")
        else:
            upsert_eod_csv(ticker, d_eod, v_eod); st.success("EOD saved."); st.rerun()

# 3) Manage entries
buys_df = load_buys_csv(ticker)
eod_df  = load_eod_csv(ticker)

st.divider()
st.markdown(f"### 3) Manage entries — {ticker}")
L,R = st.columns(2)

with L:
    st.markdown("**Buys**")
    if buys_df.empty:
        st.info("No buys yet.")
    else:
        show = buys_df.copy()
        show["shares (lot)"] = (show["amount"]/show["price"]).round(5)
        show = show[["id","trade_date","amount","price","shares (lot)"]].rename(columns={"trade_date":"date","id":"_id"})
        st.dataframe(show.drop(columns=["_id"]), use_container_width=True, hide_index=True)
        ids = st.multiselect("Delete buys by ID", options=show["_id"].tolist(),
                             format_func=lambda x: f"Row #{show.index[show['_id']==x][0]+1}")
        if st.button("🗑️ Delete selected buys"):
            delete_buys_ids_csv(ids); st.success("Deleted."); st.rerun()

with R:
    st.markdown("**EOD values**")
    if eod_df.empty:
        st.info("No EOD values yet.")
    else:
        showe = eod_df.rename(columns={"eod_date":"date"})
        st.dataframe(showe, use_container_width=True, hide_index=True)
        dsel = st.multiselect("Delete EOD rows (by date)", options=eod_df["eod_date"].tolist())
        if st.button("🗑️ Delete selected EOD"):
            delete_eod_rows_csv(ticker, dsel); st.success("Deleted."); st.rerun()

# 4) Metrics & charts
st.divider()
st.markdown(f"### 4) Metrics & charts — {ticker}")
daily = compute_daily(buys_df.rename(columns={"trade_date":"date"}) if not buys_df.empty else pd.DataFrame(),
                      eod_df.copy() if not eod_df.empty else pd.DataFrame())
if daily.empty:
    st.info("Add a buy to see analytics.")
else:
    met = metrics_from_daily(daily)
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Cost basis", f"${met['cost']:,.2f}")
    m2.metric("Shares", f"{met['sh']:.5f}")
    m3.metric("Portfolio value", f"${met['val']:,.2f}")
    m4.metric("Cumulative P&L", f"${met['pnl']:,.2f}")
    m5.metric("Return", f"{met['ret']:.2f}%")

    C1,C2 = st.columns(2)
    with C1:
        st.markdown("**Portfolio value over time (EOD-aware)**")
        st.line_chart(daily.set_index("date")[["portfolio_value"]], height=260, use_container_width=True)
    with C2:
        st.markdown("**Daily P&L ($) – market move**")
        st.bar_chart(daily.set_index("date")[["daily_pnl"]], height=260, use_container_width=True)

    st.markdown("**Last 30 days**")
    st.dataframe(daily.tail(30), use_container_width=True, hide_index=True)

# 5) Positions — all tickers
st.divider()
st.markdown("### 5) Positions — all tickers")
all_buys = load_buys_csv()
all_eod  = load_eod_csv()
if all_buys.empty:
    st.info("No buys yet.")
else:
    rows = []
    for tkr in sorted(all_buys["ticker"].unique()):
        b = all_buys[all_buys["ticker"]==tkr].rename(columns={"trade_date":"date"})
        e = all_eod[all_eod["ticker"]==tkr].rename(columns={"eod_date":"date"})
        daily_t = compute_daily(b, e)
        met_t   = metrics_from_daily(daily_t)
        rows.append({
            "ticker": tkr,
            "cost_basis": round(met_t["cost"],2),
            "shares": round(met_t["sh"],5),
            "value": round(met_t["val"],2),
            "pnl": round(met_t["pnl"],2),
            "ret_pct": round(met_t["ret"],2),
        })
    pos = pd.DataFrame(rows)
    if not pos.empty:
        tot_cost = pos["cost_basis"].sum()
        tot_val  = pos["value"].sum()
        tot_pnl  = pos["pnl"].sum()
        tot_ret  = round((tot_pnl/tot_cost*100),2) if tot_cost>0 else 0.00
        total_row = pd.DataFrame([{
            "ticker":"TOTAL","cost_basis":round(tot_cost,2),
            "shares":float(Decimal(str(pos["shares"].sum())).quantize(SHARE_Q, ROUND_HALF_UP)),
            "value":round(tot_val,2),"pnl":round(tot_pnl,2),"ret_pct":tot_ret
        }])
        st.dataframe(pd.concat([pos.sort_values("ticker"), total_row], ignore_index=True),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No positions yet.")
