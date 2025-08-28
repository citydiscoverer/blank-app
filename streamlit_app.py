# Felix Abayomi – DCA Dashboard (Supabase + Multi-Ticker, Stable + Correct P&L)
# streamlit_app.py
import os
import streamlit as st
import pandas as pd
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Optional, List, Dict, Any

# ---------- Precision / helpers ----------
getcontext().prec = 28
SHARE_Q = Decimal("0.00001")   # shares rounded to 5 dp
CENT    = Decimal("0.01")

def D(x):  # safe Decimal
    return Decimal(str(x)) if x is not None and str(x) != "" else Decimal("0")

# ---------- Page config ----------
st.set_page_config(page_title="Felix Legacy – Multi-Ticker DCA", layout="wide")
st.title("📈 Felix Abayomi – Multi-Ticker DCA (Cloud-saved)")

# ======================================================
#                 DATA BACKENDS (DB / CSV)
# ======================================================
BUYS_TABLE = "buys"  # expected DB schema: id, ticker, trade_date, amount, price

class StoreBase:
    def list_tickers(self) -> List[str]: ...
    def load_buys(self, ticker: Optional[str]) -> pd.DataFrame: ...
    def insert_buy(self, ticker: str, d: date, amount: Decimal, price: Decimal) -> None: ...
    def delete_ids(self, ids: List[Any]) -> None: ...
    def clear_ticker(self, ticker: str) -> None: ...
    def load_all_raw(self) -> pd.DataFrame: ...

# -------- Supabase backend --------
class StoreSupabase(StoreBase):
    def __init__(self, client):
        self.sb = client

    def list_tickers(self) -> List[str]:
        res = self.sb.table(BUYS_TABLE).select("ticker").execute()
        return sorted({row["ticker"] for row in (res.data or [])})

    def load_buys(self, ticker: Optional[str]) -> pd.DataFrame:
        q = self.sb.table(BUYS_TABLE).select("*")
        if ticker:
            q = q.eq("ticker", ticker)
        res = q.order("trade_date", desc=False).execute()
        df = pd.DataFrame(res.data or [])
        if df.empty: return df
        df = df.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["amount"] = df["amount"].astype(float)
        df["price"]  = df["price"].astype(float)
        return df

    def insert_buy(self, ticker: str, d: date, amount: Decimal, price: Decimal) -> None:
        self.sb.table(BUYS_TABLE).insert({
            "ticker": ticker.upper().strip(),
            "trade_date": d.isoformat(),
            "amount": float(amount),
            "price": float(price),
        }).execute()

    def delete_ids(self, ids: List[Any]) -> None:
        if not ids: return
        self.sb.table(BUYS_TABLE).delete().in_("id", ids).execute()

    def clear_ticker(self, ticker: str) -> None:
        self.sb.table(BUYS_TABLE).delete().eq("ticker", ticker).execute()

    def load_all_raw(self) -> pd.DataFrame:
        res = self.sb.table(BUYS_TABLE).select("*").order("ticker").order("trade_date").execute()
        return pd.DataFrame(res.data or [])

# -------- Local CSV fallback backend --------
class StoreCSV(StoreBase):
    def __init__(self, path: str = "local_buys.csv"):
        self.path = path
        if not os.path.exists(self.path):
            pd.DataFrame(columns=["id","ticker","trade_date","amount","price"]).to_csv(self.path, index=False)

    def _read(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        return df

    def _write(self, df: pd.DataFrame) -> None:
        df.to_csv(self.path, index=False)

    def list_tickers(self) -> List[str]:
        df = self._read()
        return sorted(df["ticker"].dropna().unique().tolist()) if not df.empty else []

    def load_buys(self, ticker: Optional[str]) -> pd.DataFrame:
        df = self._read()
        if df.empty: return df
        if ticker:
            df = df[df["ticker"] == ticker]
        df = df.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.astype({"amount":float,"price":float})

    def insert_buy(self, ticker: str, d: date, amount: Decimal, price: Decimal) -> None:
        df = self._read()
        new_id = (df["id"].max() + 1) if not df.empty else 1
        row = {"id": new_id, "ticker": ticker.upper().strip(),
               "trade_date": d.isoformat(), "amount": float(amount), "price": float(price)}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._write(df)

    def delete_ids(self, ids: List[Any]) -> None:
        if not ids: return
        df = self._read()
        df = df[~df["id"].isin(ids)]
        self._write(df)

    def clear_ticker(self, ticker: str) -> None:
        df = self._read()
        df = df[df["ticker"] != ticker]
        self._write(df)

    def load_all_raw(self) -> pd.DataFrame:
        return self._read()

# ======================================================
#               CONNECT TO SUPABASE (SAFE)
# ======================================================
@st.cache_resource(show_spinner=False)
def get_supabase_client():
    try:
        from supabase import create_client
    except Exception:
        return None  # package not installed yet
    # Prefer Streamlit secrets; fallback to env vars
    s = st.secrets.get("supabase", {}) if hasattr(st, "secrets") else {}
    url = s.get("url") or os.environ.get("SUPABASE_URL")
    key = (s.get("service_role_key") or s.get("service_key") or s.get("key")
           or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY"))
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None

sb_client = get_supabase_client()
store: StoreBase = StoreSupabase(sb_client) if sb_client else StoreCSV()
connected = isinstance(store, StoreSupabase)

# ======================================================
#                     CALCULATIONS
# ======================================================
def compute_lots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(shares_lot=[], cum_shares=[], cum_invested=[])
    tmp = df.copy().sort_values("date")
    tmp["amountD"] = tmp["amount"].apply(D)
    tmp["priceD"]  = tmp["price"].apply(D)
    tmp["shares_lotD"]   = (tmp["amountD"] / tmp["priceD"]).apply(lambda s: s.quantize(SHARE_Q, ROUND_HALF_UP))
    tmp["cum_sharesD"]   = tmp["shares_lotD"].cumsum()
    tmp["cum_investedD"] = tmp["amountD"].cumsum()
    out = tmp.copy()
    out["shares_lot"]   = out["shares_lotD"].astype(float)
    out["cum_shares"]   = out["cum_sharesD"].astype(float)
    out["cum_invested"] = out["cum_investedD"].astype(float)
    return out

def calc_metrics(df_lots: pd.DataFrame, current_price: Decimal):
    if df_lots.empty:
        return {"cost_basis": D(0), "shares": D(0), "value": D(0), "cum_pnl": D(0),
                "ret_pct": D(0), "series": pd.DataFrame()}
    shares_total = df_lots["cum_sharesD"].iloc[-1]
    cost_basis   = df_lots["cum_investedD"].iloc[-1]
    pv_today = (shares_total * current_price).quantize(CENT, ROUND_HALF_UP)
    cum_pnl  = (pv_today - cost_basis).quantize(CENT, ROUND_HALF_UP)
    ret_pct  = (cum_pnl / cost_basis * D(100)).quantize(Decimal("0.01"), ROUND_HALF_UP) if cost_basis > 0 else D(0)

    ser = df_lots[["date", "cum_sharesD", "amountD"]].copy()
    ser["portfolio_valueD"] = ser["cum_sharesD"].apply(lambda s: (s * current_price).quantize(CENT, ROUND_HALF_UP))
    ser["prev_valueD"] = ser["portfolio_valueD"].shift(1).fillna(D(0))
    ser["daily_market_moveD"] = (ser["portfolio_valueD"] - (ser["prev_valueD"] + ser["amountD"])).apply(
        lambda x: x.quantize(CENT, ROUND_HALF_UP)
    )
    ser_out = ser.copy()
    ser_out["portfolio_value"]   = ser_out["portfolio_valueD"].astype(float)
    ser_out["daily_market_move"] = ser_out["daily_market_moveD"].astype(float)

    return {
        "cost_basis": cost_basis, "shares": shares_total, "value": pv_today,
        "cum_pnl": cum_pnl, "ret_pct": ret_pct,
        "series": ser_out[["date", "portfolio_value", "daily_market_move"]]
    }

def summarize_positions(all_buys: pd.DataFrame, price_map: Dict[str, Decimal]) -> pd.DataFrame:
    if all_buys.empty: return pd.DataFrame()
    rows = []
    for tkr, grp in all_buys.groupby("ticker"):
        grp2 = grp.rename(columns={"trade_date":"date"}) if "trade_date" in grp.columns else grp
        grp2["date"] = pd.to_datetime(grp2["date"]).dt.date
        lots = compute_lots(grp2.astype({"amount":float,"price":float}))
        price = price_map.get(tkr, D(grp2["price"].iloc[-1]))
        met = calc_metrics(lots, price)
        rows.append({
            "ticker": tkr,
            "cost_basis": float(met["cost_basis"]),
            "shares": float(met["shares"]),
            "price": float(price),
            "value": float(met["value"]),
            "pnl": float(met["cum_pnl"]),
            "ret_pct": float(met["ret_pct"]),
        })
    dfp = pd.DataFrame(rows)
    if dfp.empty: return dfp
    dfp["ret_pct"] = dfp["ret_pct"].round(2)
    tot_cost = dfp["cost_basis"].sum()
    tot_val  = dfp["value"].sum()
    tot_pnl  = dfp["pnl"].sum()
    tot_ret  = round((tot_pnl / tot_cost * 100), 2) if tot_cost > 0 else 0.00
    total_row = pd.DataFrame([{
        "ticker": "TOTAL",
        "cost_basis": round(tot_cost, 2),
        "shares": float(Decimal(str(dfp["shares"].sum())).quantize(SHARE_Q, ROUND_HALF_UP)),
        "price": float("nan"),
        "value": round(tot_val, 2),
        "pnl": round(tot_pnl, 2),
        "ret_pct": tot_ret,
    }])
    return pd.concat([dfp.sort_values("ticker"), total_row], ignore_index=True)

# ======================================================
#                          UI
# ======================================================
with st.sidebar:
    st.subheader("☁️ Data source")
    if connected:
        st.success("Supabase: Connected")
    else:
        st.warning("Supabase NOT set → using local CSV")
        st.caption("Add secrets or env vars to enable cloud saving.")

existing = store.list_tickers()
defaults = ["SPLG", "SCHD", "VOO", "SPY"]
picker_opts = sorted(set(existing + defaults)) if existing else defaults
default_tkr = "SPLG" if "SPLG" in picker_opts else picker_opts[0]
ticker = st.sidebar.selectbox("Ticker", options=picker_opts, index=picker_opts.index(default_tkr))

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("🔄 Refresh", use_container_width=True): st.rerun()
with c2:
    if st.button("🧹 Clear ticker", type="secondary", use_container_width=True):
        store.clear_ticker(ticker); st.success(f"Cleared {ticker}"); st.rerun()

# ---- Add buy ----
st.markdown("### 1) Add a buy")
ac1, ac2, ac3, ac4 = st.columns([1.2,1,1,1.2])
with ac1:
    d = st.date_input("Date", value=date.today())
with ac2:
    amt = D(st.number_input("Amount ($)", value=25.00, min_value=0.0, step=1.0))
with ac3:
    px  = D(st.number_input("Execution price ($/share)", value=75.00, min_value=0.0, step=0.01))
with ac4:
    st.write(" ")
    if st.button("➕ Add entry", use_container_width=True):
        store.insert_buy(ticker, d, amt, px); st.success("Added."); st.rerun()

# ---- Manage entries ----
df = store.load_buys(ticker)

st.divider()
st.markdown(f"### 2) Manage entries — {ticker}")
if df.empty:
    st.info("No entries for this ticker yet.")
else:
    df_disp = df.copy()
    df_disp["shares (lot)"] = (df_disp["amount"] / df_disp["price"]).round(5)
    df_disp = df_disp[["id","date","amount","price","shares (lot)"]]
    df_disp = df_disp.rename(columns={"id":"_id"})
    st.dataframe(df_disp.drop(columns=["_id"]), use_container_width=True, hide_index=True)
    ids = st.multiselect("Select rows to delete", options=df_disp["_id"].tolist(),
                         format_func=lambda x: f"Row #{df_disp.index[df_disp['_id']==x][0]+1}")
    dc = st.columns([1,3,1,1])
    with dc[3]:
        if st.button("🗑️ Delete selected"):
            store.delete_ids(ids); st.success("Deleted."); st.rerun()

# ---- Current price ----
st.divider()
st.markdown(f"### 3) Current price — {ticker}")
default_px = df["price"].iloc[-1] if not df.empty else 75.0
cur_price = D(st.number_input(f"{ticker} price ($)", value=float(default_px), step=0.01))

# ---- Metrics & charts ----
st.divider()
st.markdown(f"### 4) Metrics & charts — {ticker}")
lots = compute_lots(df) if not df.empty else pd.DataFrame()
metrics = calc_metrics(lots, cur_price) if not lots.empty else {
    "cost_basis": D(0), "shares": D(0), "value": D(0), "cum_pnl": D(0), "ret_pct": D(0), "series": pd.DataFrame()
}

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Cost basis", f"${metrics['cost_basis']:,.2f}")
m2.metric("Shares", f"{metrics['shares']:.5f}")
m3.metric("Portfolio value", f"${metrics['value']:,.2f}")
m4.metric("Cumulative P&L", f"${metrics['cum_pnl']:,.2f}")
m5.metric("Return", f"{metrics['ret_pct']}%")

if not metrics["series"].empty:
    lc, rc = st.columns(2)
    with lc:
        st.markdown("**Portfolio value over time**")
        st.line_chart(metrics["series"].set_index("date")[["portfolio_value"]], height=260, use_container_width=True)
    with rc:
        st.markdown("**Daily P&L (market move)**")
        st.bar_chart(metrics["series"].set_index("date")[["daily_market_move"]], height=260, use_container_width=True)

st.markdown("#### Last 30 days")
if not lots.empty and not metrics["series"].empty:
    recent = metrics["series"].merge(
        lots[["date","amount","price","shares_lot","cum_shares","cum_invested"]],
        on="date", how="left"
    ).tail(30).rename(columns={
        "shares_lot":"shares",
        "daily_market_move":"day P&L (market)",
        "portfolio_value":"value"
    })
    st.dataframe(recent, use_container_width=True, hide_index=True)

# ---- Positions ticket (ALL tickers) ----
st.divider()
st.markdown("### 5) Positions — all tickers")
all_raw = store.load_all_raw()
if all_raw is None or all_raw.empty:
    st.info("No buys yet.")
else:
    tickers_all = sorted(all_raw["ticker"].unique())
    st.caption("Set current price for each ticker (defaults to last buy price).")
    cols = st.columns(min(4, len(tickers_all))) + st.columns(max(0, len(tickers_all) - 4))
    price_map = {}
    for i, tkr in enumerate(tickers_all):
        last_px = float(all_raw.loc[all_raw["ticker"] == tkr, "price"].iloc[-1])
        with cols[i]:
            price_map[tkr] = D(st.number_input(f"{tkr} price", value=last_px, step=0.01, key=f"px_{tkr}"))
    all_for_summ = all_raw.rename(columns={"trade_date":"date"})
    all_for_summ["date"] = pd.to_datetime(all_for_summ["date"]).dt.date
    positions = summarize_positions(all_for_summ.astype({"amount":float,"price":float}), price_map)
    st.dataframe(positions, use_container_width=True, hide_index=True)

st.caption("Tip: on Cloud, add Supabase secrets. Locally, a CSV fallback keeps everything usable.")
