# Felix Abayomi – DCA Dashboard (Supabase + Multi-Ticker, Stable + Correct P&L)
# streamlit_app.py

import streamlit as st
import pandas as pd
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, getcontext
import json

# ───────────────── Streamlit config (must be first st.* call) ─────────────────
st.set_page_config(page_title="Felix Legacy – Multi-Ticker DCA", layout="wide")
st.title("📈 Felix Abayomi – Multi-Ticker DCA Dashboard (Cloud-saved)")

# ───────────────── Precision & helpers ─────────────────
getcontext().prec = 28
SHARE_Q = Decimal("0.00001")   # shares rounded to 5 dp (broker-like)
CENT    = Decimal("0.01")

def D(x):
    """Safe Decimal from any numeric/string/None."""
    return Decimal(str(x)) if x is not None and str(x) != "" else Decimal("0")

# ───────────────── Supabase client (from Streamlit Secrets only) ─────────────────
@st.cache_resource(show_spinner=False)
def get_supabase():
    # library import
    try:
        from supabase import create_client
    except Exception as e:
        st.error("Supabase client not installed. Add `supabase>=2.5` to requirements.txt.")
        st.caption(f"Import error: {e}")
        return None

    # secrets (no fallbacks by design)
    try:
        url = st.secrets["supabase"]["url"].strip().rstrip("/")
        key = st.secrets["supabase"]["service_role_key"].strip()
    except KeyError:
        st.error('Missing secrets. In Settings → Secrets add:\n[supabase]\nurl="https://<ref>.supabase.co"\nservice_role_key="<SERVICE_ROLE_KEY>"')
        return None

    try:
        return create_client(url, key)
    except Exception as e:
        st.error(f"Supabase init error: {e}")
        return None

sb = get_supabase()

# ───────────────── DB helpers ─────────────────
BUYS_TABLE = "buys"  # expected columns: id (uuid/bigint), ticker text, date date, amount numeric, price numeric

def _show_postgrest_error(e, context_msg: str):
    """Pretty-print PostgREST API error payload if available."""
    try:
        from postgrest.exceptions import APIError
        if isinstance(e, APIError):
            st.error(context_msg)
            st.code(json.dumps(e.args[0], indent=2), language="json")
            return
    except Exception:
        pass
    st.error(f"{context_msg}: {e}")

def list_tickers() -> list[str]:
    if not sb: return []
    try:
        res = sb.table(BUYS_TABLE).select("ticker").execute()
        return sorted({row["ticker"] for row in (res.data or []) if row.get("ticker")})
    except Exception as e:
        _show_postgrest_error(e, "Could not list tickers")
        return []

def load_buys(ticker: str | None) -> pd.DataFrame:
    if not sb: return pd.DataFrame()
    try:
        q = sb.table(BUYS_TABLE).select("*")
        if ticker:
            q = q.eq("ticker", ticker)
        # Avoid server-side ORDER BY on 'date'; sort locally to dodge edge cases.
        res = q.execute()
        df = pd.DataFrame(res.data or [])
        if df.empty:
            return df
        if "date" not in df.columns:
            st.error("Expected column 'date' not found in table 'buys'.")
            return pd.DataFrame()
        df["date"]   = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df["price"]  = pd.to_numeric(df["price"],  errors="coerce").fillna(0.0)
        return df.sort_values("date")
    except Exception as e:
        _show_postgrest_error(e, "Load error")
        return pd.DataFrame()

def insert_buy(ticker: str, d: date, amount: Decimal, price: Decimal):
    if not sb: return
    try:
        sb.table(BUYS_TABLE).insert({
            "ticker": ticker.upper().strip(),
            "date":   d.isoformat(),
            "amount": float(amount),
            "price":  float(price),
        }).execute()
    except Exception as e:
        _show_postgrest_error(e, "Insert failed")

def delete_ids(ids: list[str | int]):
    if not sb or not ids: return
    try:
        sb.table(BUYS_TABLE).delete().in_("id", ids).execute()
    except Exception as e:
        _show_postgrest_error(e, "Delete failed")

def clear_ticker(ticker: str):
    if not sb: return
    try:
        sb.table(BUYS_TABLE).delete().eq("ticker", ticker).execute()
    except Exception as e:
        _show_postgrest_error(e, "Clear failed")

# ───────────────── Math (broker-style) ─────────────────
def compute_lots(df: pd.DataFrame) -> pd.DataFrame:
    """Per-lot shares rounded to 5 dp; running totals with Decimal; outputs both Decimal & float columns."""
    if df.empty:
        # keep same columns pattern the rest of the app expects
        return df.assign(shares_lot=[], cum_shares=[], cum_invested=[])

    tmp = df.copy().sort_values("date")
    tmp["amountD"] = tmp["amount"].apply(D)
    tmp["priceD"]  = tmp["price"].apply(D)

    def _lot(a: Decimal, p: Decimal) -> Decimal:
        return (a / p).quantize(SHARE_Q, ROUND_HALF_UP) if p > 0 else Decimal("0")

    tmp["shares_lotD"]   = [_lot(a, p) for a, p in zip(tmp["amountD"], tmp["priceD"])]
    tmp["cum_sharesD"]   = tmp["shares_lotD"].cumsum()
    tmp["cum_investedD"] = tmp["amountD"].cumsum()

    out = tmp.copy()
    out["shares_lot"]   = out["shares_lotD"].astype(float)
    out["cum_shares"]   = out["cum_sharesD"].astype(float)
    out["cum_invested"] = out["cum_investedD"].astype(float)
    return out

def calc_metrics(df_lots: pd.DataFrame, current_price: Decimal):
    """Returns headline metrics + series (value and day market move), using Decimal cents rounding."""
    if df_lots.empty:
        return {
            "cost_basis": Decimal("0"),
            "shares": Decimal("0"),
            "value": Decimal("0"),
            "cum_pnl": Decimal("0"),
            "ret_pct": Decimal("0"),
            "series": pd.DataFrame(),
        }

    shares_total = df_lots["cum_sharesD"].iloc[-1]
    cost_basis   = df_lots["cum_investedD"].iloc[-1]

    pv_today = (shares_total * current_price).quantize(CENT, ROUND_HALF_UP)
    cum_pnl  = (pv_today - cost_basis).quantize(CENT, ROUND_HALF_UP)
    ret_pct  = (cum_pnl / cost_basis * Decimal("100")).quantize(Decimal("0.01"), ROUND_HALF_UP) if cost_basis > 0 else Decimal("0.00")

    ser = df_lots[["date", "cum_sharesD", "amountD"]].copy()
    ser["portfolio_valueD"] = ser["cum_sharesD"].apply(lambda s: (s * current_price).quantize(CENT, ROUND_HALF_UP))
    ser["prev_valueD"] = ser["portfolio_valueD"].shift(1).fillna(Decimal("0"))
    # day market move (exclude new cash): V_t − (V_{t-1} + contribution_t)
    ser["daily_market_moveD"] = (ser["portfolio_valueD"] - (ser["prev_valueD"] + ser["amountD"])).apply(
        lambda x: x.quantize(CENT, ROUND_HALF_UP)
    )

    ser_out = ser.copy()
    ser_out["portfolio_value"]   = ser_out["portfolio_valueD"].astype(float)
    ser_out["daily_market_move"] = ser_out["daily_market_moveD"].astype(float)

    return {
        "cost_basis": cost_basis,
        "shares": shares_total,
        "value": pv_today,
        "cum_pnl": cum_pnl,
        "ret_pct": ret_pct,
        "series": ser_out[["date", "portfolio_value", "daily_market_move"]],
    }

def summarize_positions(all_buys: pd.DataFrame, price_map: dict[str, Decimal]) -> pd.DataFrame:
    """One row per ticker: cost basis, shares, price, value, P&L; plus TOTAL row."""
    if all_buys.empty:
        return pd.DataFrame()
    rows = []
    for tkr, grp in all_buys.groupby("ticker"):
        lots = compute_lots(grp)
        price = price_map.get(tkr, D(grp["price"].iloc[-1]))
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
    if dfp.empty:
        return dfp
    dfp["ret_pct"] = dfp["ret_pct"].round(2)
    # TOTAL row
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

# ───────────────── Sidebar ─────────────────
st.sidebar.subheader("☁️ Supabase")
if sb:
    st.sidebar.success("Connected")
    # quick reachability probe (non-fatal if table empty)
    try:
        probe = sb.table(BUYS_TABLE).select("id").limit(1).execute()
        st.sidebar.caption(f"'{BUYS_TABLE}' reachable ✓ (visible rows: {len(probe.data)})")
    except Exception as e:
        _show_postgrest_error(e, "Table reachability failed")
else:
    st.sidebar.error("Not connected")

existing = list_tickers()
defaults = ["SPLG", "SCHD", "VOO", "SPY"]
picker_opts = sorted(set(existing + defaults)) if existing else defaults
default_ticker = "SPLG" if "SPLG" in picker_opts else picker_opts[0]
ticker = st.sidebar.selectbox("Ticker", options=picker_opts, index=picker_opts.index(default_ticker))

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("🔄 Refresh data", use_container_width=True):
        st.rerun()
with col_sb2:
    if st.button("🧹 Clear THIS ticker", type="secondary", use_container_width=True):
        if sb:
            clear_ticker(ticker)
            st.success(f"Cleared {ticker}")
            st.rerun()

# ───────────────── Add buy ─────────────────
st.markdown("### 1) Add a buy")
c1, c2, c3, c4 = st.columns([1.2,1,1,1.2])
with c1:
    d = st.date_input("Date", value=date.today())
with c2:
    amt = D(st.number_input("Amount ($)", value=25.00, min_value=0.0, step=1.0))
with c3:
    px  = D(st.number_input("Execution price ($/share)", value=75.00, min_value=0.0, step=0.01))
with c4:
    st.write(" ")
    if st.button("➕ Add entry", use_container_width=True):
        if sb:
            insert_buy(ticker, d, amt, px)
            st.success("Added.")
            st.rerun()
        else:
            st.error("Supabase not connected.")

# ───────────────── Manage entries ─────────────────
df = load_buys(ticker)

st.divider()
st.markdown(f"### 2) Manage entries — {ticker}")
if df.empty:
    st.info("No entries for this ticker yet.")
else:
    df_disp = df.copy()
    # avoid div/0 issues
    df_disp["shares (lot)"] = [(a/p if p else 0.0) for a, p in zip(df_disp["amount"], df_disp["price"])]
    df_disp["shares (lot)"] = df_disp["shares (lot)"].round(5)
    has_id = "id" in df_disp.columns
    if has_id:
        df_disp = df_disp.rename(columns={"id": "_id"})
    st.dataframe(df_disp.drop(columns=["_id"], errors="ignore"), use_container_width=True, hide_index=True)

    ids = []
    if has_id:
        ids = st.multiselect(
            "Select rows to delete",
            options=df_disp["_id"].tolist(),
            format_func=lambda x: f"Row #{df_disp.index[df_disp['_id']==x][0]+1}"
        )
    del_cols = st.columns([1,3,1,1])
    with del_cols[3]:
        if st.button("🗑️ Delete selected"):
            if has_id and ids:
                delete_ids(ids)
                st.success("Deleted.")
                st.rerun()
            else:
                st.warning("No selectable IDs in this dataset.")

# ───────────────── Current price ─────────────────
st.divider()
st.markdown(f"### 3) Current price — {ticker}")
default_px = float(df["price"].iloc[-1]) if not df.empty else 75.0
cur_price = D(st.number_input(f"{ticker} price ($)", value=float(default_px), step=0.01))

# ───────────────── Metrics & charts ─────────────────
st.divider()
st.markdown(f"### 4) Metrics & charts — {ticker}")
lots = compute_lots(df)
metrics = calc_metrics(lots, cur_price)

cA, cB, cC, cD, cE = st.columns(5)
cA.metric("Cost basis", f"${metrics['cost_basis']:,.2f}")
cB.metric("Shares", f"{metrics['shares']:.5f}")
cC.metric("Portfolio value", f"${metrics['value']:,.2f}")
cD.metric("Cumulative P&L", f"${metrics['cum_pnl']:,.2f}")
cE.metric("Return", f"{metrics['ret_pct']}%")

if not metrics["series"].empty:
    L, R = st.columns(2)
    with L:
        st.markdown("**Portfolio value over time**")
        st.line_chart(metrics["series"].set_index("date")[["portfolio_value"]], height=260, use_container_width=True)
    with R:
        st.markdown("**Daily P&L (market move)**")
        st.bar_chart(metrics["series"].set_index("date")[["daily_market_move"]], height=260, use_container_width=True)

st.markdown("#### Last 30 days")
if not lots.empty and not metrics["series"].empty:
    recent = metrics["series"].merge(
        lots[["date","amount","price","shares_lot","cum_shares","cum_invested"]],
        on="date", how="left"
    ).tail(30)
    recent = recent.rename(columns={
        "shares_lot":"shares",
        "daily_market_move":"day P&L (market)",
        "portfolio_value":"value"
    })
    st.dataframe(recent, use_container_width=True, hide_index=True)

# ───────────────── Positions — all tickers ─────────────────
st.divider()
st.markdown("### 5) Positions — all tickers")

all_buys = load_buys(None)
if all_buys.empty:
    st.info("No buys yet.")
else:
    tickers_all = sorted(all_buys["ticker"].unique())
    st.caption("Set current price for each ticker (defaults to last buy price).")
    cols = st.columns(min(4, len(tickers_all))) + st.columns(max(0, len(tickers_all)-4))
    price_map: dict[str, Decimal] = {}
    for i, tkr in enumerate(tickers_all):
        last_px = float(all_buys.loc[all_buys["ticker"] == tkr, "price"].iloc[-1])
        with cols[i]:
            price_map[tkr] = D(st.number_input(f"{tkr} price", value=last_px, step=0.01, key=f"px_{tkr}"))
    positions = summarize_positions(all_buys, price_map)
    if positions.empty:
        st.info("No positions yet.")
    else:
        st.dataframe(positions, use_container_width=True, hide_index=True)

st.caption("Tip: To match your broker exactly, type the broker's live price in the price boxes.")
