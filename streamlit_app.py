# Felix Abayomi – DCA Dashboard (Supabase + Multi-Ticker, Stable + Correct P&L)
# streamlit_app.py
import os
import math
import streamlit as st
import pandas as pd
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Optional, List, Dict, Any

# ---------------- Precision / helpers ----------------
getcontext().prec = 28
SHARE_Q = Decimal("0.00001")   # shares rounded to 5 dp
CENT    = Decimal("0.01")
def D(x):  # safe Decimal
    return Decimal(str(x)) if x is not None and str(x) != "" else Decimal("0")

st.set_page_config(page_title="Felix Legacy – Multi-Ticker DCA", layout="wide")
st.title("📈 Felix Abayomi – Multi-Ticker DCA (Cloud-saved + EOD)")

# ======================================================
#                 DATA BACKENDS (DB / CSV)
# ======================================================
BUYS_TABLE = "buys"   # id, ticker, trade_date, amount, price
EOD_TABLE  = "eod"    # id, ticker, eod_date, eod_value

class StoreBase:
    # buys
    def list_tickers(self) -> List[str]: ...
    def load_buys(self, ticker: Optional[str]) -> pd.DataFrame: ...
    def insert_buy(self, ticker: str, d: date, amount: Decimal, price: Decimal) -> None: ...
    def delete_ids(self, ids: List[Any]) -> None: ...
    def clear_ticker(self, ticker: str) -> None: ...
    def load_all_raw(self) -> pd.DataFrame: ...
    # eod
    def load_eod(self, ticker: str) -> pd.DataFrame: ...
    def upsert_eod(self, ticker: str, d: date, value: Decimal) -> None: ...
    def delete_eod(self, ticker: str, dates: List[date]) -> None: ...

# -------- Supabase backend --------
class StoreSupabase(StoreBase):
    def __init__(self, client):
        self.sb = client

    # buys
    def list_tickers(self) -> List[str]:
        res = self.sb.table(BUYS_TABLE).select("ticker").execute()
        return sorted({row["ticker"] for row in (res.data or [])})

    def load_buys(self, ticker: Optional[str]) -> pd.DataFrame:
        q = self.sb.table(BUYS_TABLE).select("*")
        if ticker: q = q.eq("ticker", ticker)
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
        self.sb.table(EOD_TABLE).delete().eq("ticker", ticker).execute()

    def load_all_raw(self) -> pd.DataFrame:
        res = self.sb.table(BUYS_TABLE).select("*").order("ticker").order("trade_date").execute()
        return pd.DataFrame(res.data or [])

    # eod
    def load_eod(self, ticker: str) -> pd.DataFrame:
        res = self.sb.table(EOD_TABLE).select("*").eq("ticker", ticker).order("eod_date").execute()
        df = pd.DataFrame(res.data or [])
        if df.empty: return df
        df["eod_date"] = pd.to_datetime(df["eod_date"]).dt.date
        df["eod_value"] = df["eod_value"].astype(float)
        return df

    def upsert_eod(self, ticker: str, d: date, value: Decimal) -> None:
        self.sb.table(EOD_TABLE).upsert({
            "ticker": ticker.upper().strip(),
            "eod_date": d.isoformat(),
            "eod_value": float(value),
        }, on_conflict="ticker,eod_date").execute()

    def delete_eod(self, ticker: str, dates: List[date]) -> None:
        if not dates: return
        self.sb.table(EOD_TABLE).delete().eq("ticker", ticker).in_("eod_date", [d.isoformat() for d in dates]).execute()

# -------- Local CSV fallback backend --------
class StoreCSV(StoreBase):
    def __init__(self, buys_path="local_buys.csv", eod_path="local_eod.csv"):
        self.buys_path = buys_path
        self.eod_path  = eod_path
        if not os.path.exists(self.buys_path):
            pd.DataFrame(columns=["id","ticker","trade_date","amount","price"]).to_csv(self.buys_path, index=False)
        if not os.path.exists(self.eod_path):
            pd.DataFrame(columns=["id","ticker","eod_date","eod_value"]).to_csv(self.eod_path, index=False)

    # internal helpers
    def _read_buys(self): return pd.read_csv(self.buys_path)
    def _write_buys(self, df): df.to_csv(self.buys_path, index=False)
    def _read_eod(self):  return pd.read_csv(self.eod_path)
    def _write_eod(self, df): df.to_csv(self.eod_path, index=False)

    # buys
    def list_tickers(self) -> List[str]:
        df = self._read_buys()
        return sorted(df["ticker"].dropna().unique().tolist()) if not df.empty else []

    def load_buys(self, ticker: Optional[str]) -> pd.DataFrame:
        df = self._read_buys()
        if df.empty: return df
        if ticker: df = df[df["ticker"] == ticker]
        df = df.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.astype({"amount":float,"price":float})

    def insert_buy(self, ticker: str, d: date, amount: Decimal, price: Decimal) -> None:
        df = self._read_buys()
        new_id = (df["id"].max() + 1) if not df.empty else 1
        row = {"id": new_id, "ticker": ticker.upper().strip(),
               "trade_date": d.isoformat(), "amount": float(amount), "price": float(price)}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._write_buys(df)

    def delete_ids(self, ids: List[Any]) -> None:
        if not ids: return
        df = self._read_buys()
        df = df[~df["id"].isin(ids)]
        self._write_buys(df)

    def clear_ticker(self, ticker: str) -> None:
        dfb = self._read_buys()
        dfe = self._read_eod()
        dfb = dfb[dfb["ticker"] != ticker]
        dfe = dfe[dfe["ticker"] != ticker]
        self._write_buys(dfb)
        self._write_eod(dfe)

    def load_all_raw(self) -> pd.DataFrame:
        return self._read_buys()

    # eod
    def load_eod(self, ticker: str) -> pd.DataFrame:
        df = self._read_eod()
        if df.empty: return df
        df = df[df["ticker"] == ticker]
        if df.empty: return df
        df["eod_date"] = pd.to_datetime(df["eod_date"]).dt.date
        df["eod_value"] = df["eod_value"].astype(float)
        return df

    def upsert_eod(self, ticker: str, d: date, value: Decimal) -> None:
        df = self._read_eod()
        mask = (df["ticker"] == ticker) & (pd.to_datetime(df["eod_date"]).dt.date == d)
        if mask.any():
            df.loc[mask, "eod_value"] = float(value)
        else:
            new_id = (df["id"].max() + 1) if not df.empty else 1
            row = {"id": new_id, "ticker": ticker, "eod_date": d.isoformat(), "eod_value": float(value)}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._write_eod(df)

    def delete_eod(self, ticker: str, dates: List[date]) -> None:
        if not dates: return
        df = self._read_eod()
        keep = ~((df["ticker"] == ticker) & (df["eod_date"].isin([d.isoformat() for d in dates])))
        self._write_eod(df[keep])

# ======================================================
#               CONNECT TO SUPABASE (SAFE)
# ======================================================
@st.cache_resource(show_spinner=False)
def get_supabase_client():
    try:
        from supabase import create_client
    except Exception:
        return None
    s = st.secrets.get("supabase", {}) if hasattr(st, "secrets") else {}
    url = s.get("url") or os.environ.get("SUPABASE_URL")
    key = (s.get("service_role_key") or s.get("service_key") or s.get("key") or s.get("anon_key")
           or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY"))
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
def compute_daily(buys: pd.DataFrame, eod: pd.DataFrame) -> pd.DataFrame:
    """
    Returns daily table with:
      date, amount, price, shares, cum_shares, cum_invested, portfolio_value, daily_pnl, cum_pnl, ret%
    Uses EOD value when available; otherwise estimates using last known non-zero price * cum_shares.
    Daily P&L isolates market move: Δ(value) − today_contribution.
    """
    if buys.empty and eod.empty:
        return pd.DataFrame(columns=["date","amount","price","shares","cum_shares","cum_invested",
                                     "portfolio_value","daily_pnl","cum_pnl","ret%"])
    b = buys.copy()
    if not b.empty:
        b["shares"] = (b["amount"]/b["price"]).round(5)  # display
        b_agg = b.groupby("date", as_index=False).agg(amount=("amount","sum"), price=("price","mean"))
        b_agg["shares"] = (b_agg["amount"]/b_agg["price"]).round(5)
    else:
        b_agg = pd.DataFrame(columns=["date","amount","price","shares"])

    e = eod.copy().rename(columns={"eod_date":"date"}) if not eod.empty else pd.DataFrame(columns=["date","eod_value"])

    # Build timeline
    dates = []
    if not b_agg.empty: dates.append(min(b_agg["date"]))
    if not e.empty:     dates.append(min(e["date"]))
    start = min(dates) if dates else date.today()
    end   = date.today()
    t = pd.DataFrame({"date": pd.date_range(start, end, freq="D").date})

    # Merge
    t = t.merge(b_agg, how="left", on="date")
    t = t.merge(e,     how="left", on="date")
    for col in ["amount","price","shares","eod_value"]:
        if col not in t.columns: t[col] = 0.0
        t[col] = t[col].fillna(0.0)

    # Cum totals (shares as Decimal with 5dp)
    shares_lotD = t["shares"].apply(D)
    t["cum_shares"]   = shares_lotD.cumsum().astype(float)
    t["cum_invested"] = t["amount"].cumsum().round(2)

    # Proxy price for estimate (last non-zero)
    proxy = t["price"].replace(0, pd.NA).ffill().bfill().fillna(1.0)
    est_value = (t["cum_shares"] * proxy).round(2)
    t["portfolio_value"] = t["eod_value"].where(t["eod_value"] > 0, est_value)

    # Daily P&L (market move only)
    change = t["portfolio_value"].diff().fillna(0.0)
    t["daily_pnl"] = (change - t["amount"]).round(2)

    t["cum_pnl"] = (t["portfolio_value"] - t["cum_invested"]).round(2)
    t["ret%"] = (t["cum_pnl"]/t["cum_invested"]*100).replace([pd.NA, float("inf"), -float("inf")], 0).fillna(0).round(2)

    # Display shares (lot) for the day, and cum_shares
    t["shares"] = t["shares"].round(5)
    t["cum_shares"] = t["cum_shares"].round(5)
    return t[["date","amount","price","shares","cum_shares","cum_invested","portfolio_value","daily_pnl","cum_pnl","ret%"]]

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

def calc_metrics_from_daily(daily: pd.DataFrame):
    if daily.empty:
        return {"cost_basis": 0.0, "shares": 0.0, "value": 0.0, "cum_pnl": 0.0, "ret_pct": 0.0}
    last = daily.iloc[-1]
    return {
        "cost_basis": float(last["cum_invested"]),
        "shares": float(last["cum_shares"]),
        "value": float(last["portfolio_value"]),
        "cum_pnl": float(last["cum_pnl"]),
        "ret_pct": float(last["ret%"]),
    }

def summarize_positions(all_buys: pd.DataFrame, all_eod: pd.DataFrame, price_map: Dict[str, Decimal]) -> pd.DataFrame:
    if all_buys.empty: return pd.DataFrame()
    rows = []
    for tkr, grp in all_buys.groupby("ticker"):
        grp2 = grp.rename(columns={"trade_date":"date"}) if "trade_date" in grp.columns else grp
        grp2["date"] = pd.to_datetime(grp2["date"]).dt.date
        eod_t = all_eod[all_eod["ticker"] == tkr].copy() if not all_eod.empty else pd.DataFrame(columns=["ticker","eod_date","eod_value"])
        daily = compute_daily(grp2.astype({"amount":float,"price":float}), eod_t.rename(columns={"eod_date":"date"}))
        met = calc_metrics_from_daily(daily)
        rows.append({
            "ticker": tkr,
            "cost_basis": round(met["cost_basis"], 2),
            "shares": round(met["shares"], 5),
            "value": round(met["value"], 2),
            "pnl": round(met["cum_pnl"], 2),
            "ret_pct": round(met["ret_pct"], 2),
        })
    dfp = pd.DataFrame(rows)
    if dfp.empty: return dfp
    tot_cost = dfp["cost_basis"].sum()
    tot_val  = dfp["value"].sum()
    tot_pnl  = dfp["pnl"].sum()
    tot_ret  = round((tot_pnl / tot_cost * 100), 2) if tot_cost > 0 else 0.00
    total_row = pd.DataFrame([{
        "ticker": "TOTAL",
        "cost_basis": round(tot_cost, 2),
        "shares": float(Decimal(str(dfp["shares"].sum())).quantize(SHARE_Q, ROUND_HALF_UP)),
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
        st.caption("Add secrets/env vars to enable cloud saving.")

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

# ---- Add EOD value ----
st.markdown("### 2) Add broker end-of-day (EOD) value")
e1, e2, e3 = st.columns([1.2,1,1])
with e1:
    d_eod = st.date_input("EOD date", value=date.today(), key="eod_date")
with e2:
    v_eod = D(st.number_input("EOD account value ($)", min_value=0.0, value=0.0, step=0.01))
with e3:
    st.write(" ")
    if st.button("💾 Save EOD value", use_container_width=True):
        if v_eod <= 0:
            st.warning("Enter the exact EOD value from your brokerage (e.g., 24.90, 50.11).")
        else:
            store.upsert_eod(ticker, d_eod, v_eod); st.success("EOD saved."); st.rerun()

# ---- Manage entries ----
buys_df = store.load_buys(ticker)
eod_df  = store.load_eod(ticker)

st.divider()
st.markdown(f"### 3) Manage entries — {ticker}")
colL, colR = st.columns(2)

with colL:
    st.markdown("**Buys**")
    if buys_df.empty:
        st.info("No buys yet.")
    else:
        df_disp = buys_df.copy()
        df_disp["shares (lot)"] = (df_disp["amount"] / df_disp["price"]).round(5)
        df_disp = df_disp[["id","date","amount","price","shares (lot)"]].rename(columns={"id":"_id"})
        st.dataframe(df_disp.drop(columns=["_id"]), use_container_width=True, hide_index=True)
        ids = st.multiselect("Delete buys by ID", options=df_disp["_id"].tolist(),
                             format_func=lambda x: f"Row #{df_disp.index[df_disp['_id']==x][0]+1}")
        if st.button("🗑️ Delete selected buys"):
            store.delete_ids(ids); st.success("Deleted."); st.rerun()

with colR:
    st.markdown("**EOD values**")
    if eod_df.empty:
        st.info("No EOD values yet.")
    else:
        st.dataframe(eod_df.rename(columns={"eod_date":"date"}), use_container_width=True, hide_index=True)
        sel_dates = st.multiselect("Delete EOD rows (by date)", options=eod_df["eod_date"].tolist())
        if st.button("🗑️ Delete selected EOD"):
            store.delete_eod(ticker, sel_dates); st.success("Deleted."); st.rerun()

# ---- Metrics & charts (EOD-aware) ----
st.divider()
st.markdown(f"### 4) Metrics & charts — {ticker}")
daily = compute_daily(buys_df.astype({"amount":float,"price":float}), eod_df.copy())
if daily.empty:
    st.info("Add a buy to see analytics.")
else:
    met = calc_metrics_from_daily(daily)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cost basis", f"${met['cost_basis']:,.2f}")
    m2.metric("Shares", f"{met['shares']:.5f}")
    m3.metric("Portfolio value", f"${met['value']:,.2f}")
    m4.metric("Cumulative P&L", f"${met['cum_pnl']:,.2f}")
    m5.metric("Return", f"{met['ret_pct']:.2f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Portfolio value over time (EOD-aware)**")
        st.line_chart(daily.set_index("date")[["portfolio_value"]], height=260, use_container_width=True)
    with c2:
        st.markdown("**Daily P&L ($) – market move**")
        st.bar_chart(daily.set_index("date")[["daily_pnl"]], height=260, use_container_width=True)

    st.markdown("**Last 30 days**")
    st.dataframe(daily.tail(30), use_container_width=True, hide_index=True)

# ---- Positions ticket (ALL tickers) ----
st.divider()
st.markdown("### 5) Positions — all tickers")
all_buys_raw = store.load_all_raw()
all_eod_raw  = pd.DataFrame()
if connected:
    # load raw eod for all tickers from Supabase
    try:
        res_all_eod = sb_client.table(EOD_TABLE).select("*").order("ticker").order("eod_date").execute()
        all_eod_raw = pd.DataFrame(res_all_eod.data or [])
    except Exception:
        all_eod_raw = pd.DataFrame()
else:
    # from CSV fallback
    if os.path.exists("local_eod.csv"):
        all_eod_raw = pd.read_csv("local_eod.csv")

if all_buys_raw is None or all_buys_raw.empty:
    st.info("No buys yet.")
else:
    tickers_all = sorted(all_buys_raw["ticker"].unique())
    st.caption("Set current price for each ticker only if you want to estimate days without EOD (optional).")
    # Build a responsive grid (avoid columns(0) crash)
    price_map: Dict[str, Decimal] = {}
    if tickers_all:
        per_row = 4
        rows = math.ceil(len(tickers_all)/per_row)
        idx = 0
        for _ in range(rows):
            cols = st.columns(min(per_row, len(tickers_all)-idx))
            for c in cols:
                if idx >= len(tickers_all): break
                tkr = tickers_all[idx]
                last_px = float(all_buys_raw.loc[all_buys_raw["ticker"] == tkr, "price"].iloc[-1])
                price_map[tkr] = D(c.number_input(f"{tkr} est. price", value=last_px, step=0.01, key=f"px_{tkr}"))
                idx += 1
    # Summarize using EOD when present
    positions = summarize_positions(all_buys_raw, all_eod_raw, price_map)
    st.dataframe(positions, use_container_width=True, hide_index=True)

st.caption("Tip: enter EOD values daily to match your broker exactly. The app estimates only when EOD is missing.")
