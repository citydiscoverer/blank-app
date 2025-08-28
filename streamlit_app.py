# Felix Abayomi – DCA Dashboard (Supabase + Multi-Ticker, Stable + Correct P&L)
# streamlit_app.py
import os, math
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
BUYS_TABLE = "buys"           # id, ticker, trade_date, amount, price
EOD_FALLBACK_CSV = "local_eod.csv"

class StoreBase:
    # buys
    def list_tickers(self) -> List[str]: ...
    def load_buys(self, ticker: Optional[str]) -> pd.DataFrame: ...
    def insert_buy(self, ticker: str, d: date, amount: Decimal, price: Decimal) -> None: ...
    def delete_ids(self, ids: List[Any]) -> None: ...
    def clear_ticker(self, ticker: str) -> None: ...
    def load_all_buys_raw(self) -> pd.DataFrame: ...
    # eod
    def load_eod(self, ticker: str) -> pd.DataFrame: ...
    def upsert_eod(self, ticker: str, d: date, value: Decimal) -> None: ...
    def delete_eod(self, ticker: str, dates: List[date]) -> None: ...
    def load_all_eod_raw(self) -> pd.DataFrame: ...

# -------- Local CSV EOD helper --------
def ensure_eod_csv_exists(path=EOD_FALLBACK_CSV):
    if not os.path.exists(path):
        pd.DataFrame(columns=["id","ticker","eod_date","eod_value"]).to_csv(path, index=False)

def eod_csv_read(path=EOD_FALLBACK_CSV) -> pd.DataFrame:
    ensure_eod_csv_exists(path)
    df = pd.read_csv(path)
    return df

def eod_csv_write(df: pd.DataFrame, path=EOD_FALLBACK_CSV):
    df.to_csv(path, index=False)

# -------- Supabase backend (buys) + hybrid EOD --------
class StoreHybrid(StoreBase):
    """
    - Buys: Supabase
    - EOD: Supabase if we can detect table/columns; otherwise CSV fallback
    """
    def __init__(self, sb_client):
        self.sb = sb_client
        # Try to detect an EOD table/columns in Supabase
        self.eod_ready = False
        self.eod_table = None
        self.eod_date_col = None
        self.eod_value_col = None
        if self.sb:
            self._detect_eod_table()

    # ---------- EOD schema auto-detect ----------
    def _detect_eod_table(self):
        candidates = ["eod", "eod_values"]
        date_candidates = ["eod_date", "trade_date", "d", "date"]
        value_candidates = ["eod_value", "value", "val", "close", "portfolio_value"]

        for t in candidates:
            try:
                # limit 1 works even if table is empty; PostgREST returns [] with 200
                res = self.sb.table(t).select("*").limit(1).execute()
                keys = set(res.data[0].keys()) if (res.data and len(res.data) > 0) else set()
                # If empty, we still guess defaults and try an order to confirm; if it fails, we continue
                dcol = next((k for k in date_candidates if k in keys), None) or "eod_date"
                vcol = next((k for k in value_candidates if k in keys), None) or "eod_value"
                # Try a harmless ordered select to validate column name exists
                try:
                    _ = self.sb.table(t).select("*").order(dcol).limit(1).execute()
                    self.eod_table, self.eod_date_col, self.eod_value_col = t, dcol, vcol
                    self.eod_ready = True
                    return
                except Exception:
                    # try without order to allow non-existent dcol
                    _ = self.sb.table(t).select("*").limit(1).execute()
                    # still accept, but keep guessed cols; upsert will work if columns exist
                    self.eod_table, self.eod_date_col, self.eod_value_col = t, dcol, vcol
                    self.eod_ready = True
                    return
            except Exception:
                continue
        # nothing detected
        self.eod_ready = False

    # ---------- buys ----------
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
        if self.eod_ready:
            self.sb.table(self.eod_table).delete().eq("ticker", ticker).execute()
        else:
            # CSV fallback cleanup
            df = eod_csv_read()
            df = df[df["ticker"] != ticker]
            eod_csv_write(df)

    def load_all_buys_raw(self) -> pd.DataFrame:
        res = self.sb.table(BUYS_TABLE).select("*").order("ticker").order("trade_date").execute()
        return pd.DataFrame(res.data or [])

    # ---------- EOD (Supabase or CSV fallback) ----------
    def load_eod(self, ticker: str) -> pd.DataFrame:
        if self.eod_ready:
            try:
                res = self.sb.table(self.eod_table).select("*").eq("ticker", ticker).execute()
                df = pd.DataFrame(res.data or [])
                if df.empty: return df
                # normalize to eod_date/eod_value
                if self.eod_date_col != "eod_date":
                    df = df.rename(columns={self.eod_date_col: "eod_date"})
                if self.eod_value_col != "eod_value":
                    df = df.rename(columns={self.eod_value_col: "eod_value"})
                df["eod_date"] = pd.to_datetime(df["eod_date"]).dt.date
                df["eod_value"] = df["eod_value"].astype(float)
                return df[["ticker","eod_date","eod_value"]]
            except Exception:
                pass  # fall through to CSV
        # CSV fallback
        df = eod_csv_read()
        df = df[df["ticker"] == ticker].copy()
        if df.empty: return df
        df["eod_date"] = pd.to_datetime(df["eod_date"]).dt.date
        df["eod_value"] = df["eod_value"].astype(float)
        return df

    def upsert_eod(self, ticker: str, d: date, value: Decimal) -> None:
        if self.eod_ready:
            body = {"ticker": ticker.upper().strip(),
                    self.eod_date_col: d.isoformat(),
                    self.eod_value_col: float(value)}
            try:
                self.sb.table(self.eod_table).upsert(body, on_conflict=f"ticker,{self.eod_date_col}").execute()
                return
            except Exception:
                pass  # fall through to CSV
        # CSV fallback
        df = eod_csv_read()
        mask = (df["ticker"] == ticker) & (pd.to_datetime(df["eod_date"]).dt.date == d)
        if mask.any():
            df.loc[mask, "eod_value"] = float(value)
        else:
            new_id = (df["id"].max() + 1) if not df.empty else 1
            row = {"id": new_id, "ticker": ticker, "eod_date": d.isoformat(), "eod_value": float(value)}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        eod_csv_write(df)

    def delete_eod(self, ticker: str, dates: List[date]) -> None:
        if not dates: return
        if self.eod_ready:
            try:
                self.sb.table(self.eod_table).delete().eq("ticker", ticker).in_(self.eod_date_col, [d.isoformat() for d in dates]).execute()
                return
            except Exception:
                pass
        # CSV fallback
        df = eod_csv_read()
        keep = ~((df["ticker"] == ticker) & (df["eod_date"].isin([d.isoformat() for d in dates])))
        eod_csv_write(df[keep])

    def load_all_eod_raw(self) -> pd.DataFrame:
        if self.eod_ready:
            try:
                res = self.sb.table(self.eod_table).select("*").order("ticker").execute()
                df = pd.DataFrame(res.data or [])
                if df.empty: return df
                if self.eod_date_col != "eod_date":
                    df = df.rename(columns={self.eod_date_col: "eod_date"})
                if self.eod_value_col != "eod_value":
                    df = df.rename(columns={self.eod_value_col: "eod_value"})
                return df[["ticker","eod_date","eod_value"]]
            except Exception:
                pass
        # CSV fallback
        return eod_csv_read()

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
store: StoreBase = StoreHybrid(sb_client) if sb_client else StoreHybrid(None)
connected = sb_client is not None

# ======================================================
#                     CALCULATIONS
# ======================================================
def compute_daily(buys: pd.DataFrame, eod: pd.DataFrame) -> pd.DataFrame:
    """
    Returns daily table with:
      date, amount, price, shares, cum_shares, cum_invested, portfolio_value, daily_pnl, cum_pnl, ret%
    Uses EOD value when available; otherwise estimates using last known non-zero price * cum_shares.
    Daily P&L isolates market move: Δ(value) − today_contribution. First day P&L = 0.
    """
    if buys.empty and (eod is None or eod.empty):
        return pd.DataFrame(columns=["date","amount","price","shares","cum_shares","cum_invested",
                                     "portfolio_value","daily_pnl","cum_pnl","ret%"])

    b_agg = (buys.groupby("date", as_index=False)
                  .agg(amount=("amount","sum"), price=("price","mean"))) if not buys.empty else pd.DataFrame(columns=["date","amount","price"])
    b_agg["shares"] = (b_agg["amount"].fillna(0.0) / b_agg["price"].replace(0, pd.NA)).fillna(0.0).round(5)

    e = eod.copy() if (eod is not None and not eod.empty) else pd.DataFrame(columns=["eod_date","eod_value"])
    if not e.empty and "date" not in e.columns:
        e = e.rename(columns={"eod_date":"date"})
    if not e.empty:
        e["date"] = pd.to_datetime(e["date"]).dt.date

    # Build timeline
    candidates = []
    if not b_agg.empty: candidates.append(b_agg["date"].min())
    if not e.empty:     candidates.append(e["date"].min())
    start = min(candidates) if candidates else date.today()
    end   = date.today()
    t = pd.DataFrame({"date": pd.date_range(start, end, freq="D").date})

    # Merge
    t = t.merge(b_agg, how="left", on="date")
    t = t.merge(e[["date","eod_value"]] if not e.empty else pd.DataFrame(columns=["date","eod_value"]), how="left", on="date")
    for col in ["amount","price","shares","eod_value"]:
        if col not in t.columns: t[col] = 0.0
        t[col] = t[col].fillna(0.0)

    # Cum totals
    t["cum_invested"] = t["amount"].cumsum().round(2)
    t["cum_shares"]   = t["shares"].cumsum().round(5)

    # Estimate when no EOD
    proxy = t["price"].replace(0, pd.NA).ffill().bfill().fillna(1.0)
    est_value = (t["cum_shares"] * proxy).round(2)
    t["portfolio_value"] = t["eod_value"].where(t["eod_value"] > 0, est_value)

    # Daily P&L
    change = t["portfolio_value"].diff().fillna(0.0)
    t.loc[t.index[0], "amount"] = t.loc[t.index[0], "amount"]  # keep as is
    t["daily_pnl"] = (change - t["amount"]).round(2)
    if len(t) > 0: t.loc[t.index[0], "daily_pnl"] = 0.00

    t["cum_pnl"] = (t["portfolio_value"] - t["cum_invested"]).round(2)
    t["ret%"] = (t["cum_pnl"]/t["cum_invested"]*100).replace([pd.NA, float("inf"), -float("inf")], 0).fillna(0).round(2)

    # Clean display
    t["amount"] = t["amount"].round(2)
    t["price"]  = t["price"].round(4)
    return t[["date","amount","price","shares","cum_shares","cum_invested","portfolio_value","daily_pnl","cum_pnl","ret%"]]

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

# ======================================================
#                          UI
# ======================================================
with st.sidebar:
    st.subheader("☁️ Data source")
    st.success("Supabase: Connected") if connected else st.warning("Supabase NOT set → using local CSV for EOD")
    st.caption("Tip: Add correct Supabase secrets to persist EOD in cloud.")

# Ticker picker
existing = store.list_tickers() if connected else []
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
        if connected:
            store.insert_buy(ticker, d, amt, px); st.success("Added."); st.rerun()
        else:
            # If not connected, you likely created buys in Supabase earlier—warn:
            st.error("Supabase not connected. Configure secrets to save buys to cloud.")

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
        store.upsert_eod(ticker, d_eod, v_eod); st.success("EOD saved."); st.rerun()

# ---- Manage entries ----
buys_df = store.load_buys(ticker) if connected else pd.DataFrame()
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
    if eod_df is None or eod_df.empty:
        st.info("No EOD values yet.")
    else:
        st.dataframe(eod_df.rename(columns={"eod_date":"date"}), use_container_width=True, hide_index=True)
        sel_dates = st.multiselect("Delete EOD rows (by date)", options=eod_df["eod_date"].tolist())
        if st.button("🗑️ Delete selected EOD"):
            store.delete_eod(ticker, sel_dates); st.success("Deleted."); st.rerun()

# ---- Metrics & charts (EOD-aware) ----
st.divider()
st.markdown(f"### 4) Metrics & charts — {ticker}")
daily = compute_daily(buys_df.astype({"amount":float,"price":float}) if not buys_df.empty else pd.DataFrame(),
                      eod_df.copy() if (eod_df is not None and not eod_df.empty) else pd.DataFrame())
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
all_buys_raw = store.load_all_buys_raw() if connected else pd.DataFrame()
all_eod_raw  = store.load_all_eod_raw()

if all_buys_raw is None or all_buys_raw.empty:
    st.info("No buys yet.")
else:
    tickers_all = sorted(all_buys_raw["ticker"].unique())
    st.caption("Enter an estimated price only if you want to estimate days without EOD (optional).")

    # Responsive grid for price inputs (avoid columns(0) crash)
    price_map: Dict[str, Decimal] = {}
    if tickers_all:
        per_row = 4
        rows = math.ceil(len(tickers_all)/per_row)
        idx = 0
        for _ in range(rows):
            ncols = min(per_row, len(tickers_all)-idx)
            cols = st.columns(ncols) if ncols > 0 else []
            for c in cols:
                if idx >= len(tickers_all): break
                tkr = tickers_all[idx]
                last_px = float(all_buys_raw.loc[all_buys_raw["ticker"] == tkr, "price"].iloc[-1])
                price_map[tkr] = D(c.number_input(f"{tkr} est. price", value=last_px, step=0.01, key=f"px_{tkr}"))
                idx += 1

    # Summarize positions using EOD when present
    def summarize_positions(all_buys: pd.DataFrame, all_eod: pd.DataFrame, price_map: Dict[str, Decimal]) -> pd.DataFrame:
        if all_buys.empty: return pd.DataFrame()
        rows = []
        for tkr, grp in all_buys.groupby("ticker"):
            grp2 = grp.rename(columns={"trade_date":"date"}) if "trade_date" in grp.columns else grp
            grp2["date"] = pd.to_datetime(grp2["date"]).dt.date
            eod_t = all_eod[all_eod["ticker"] == tkr].copy() if not all_eod.empty else pd.DataFrame(columns=["ticker","eod_date","eod_value"])
            eod_t = eod_t.rename(columns={"eod_date":"date"}) if "eod_date" in eod_t.columns else eod_t
            daily_t = compute_daily(grp2.astype({"amount":float,"price":float}), eod_t)
            met = calc_metrics_from_daily(daily_t)
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

    positions = summarize_positions(all_buys_raw, all_eod_raw, price_map)
    st.dataframe(positions, use_container_width=True, hide_index=True)

st.caption("Tip: enter EOD values daily to match your broker exactly. App estimates only when EOD is missing.")
