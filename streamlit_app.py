# Felix Abayomi – DCA Dashboard (Supabase + Multi-Ticker, Stable + Correct P&L)

# ──────────────────────────────────────────────────────────────────────────────
# Imports (no Streamlit calls above this line)
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, getcontext

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit config — MUST be the first st.* call and only once
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Felix Legacy – Multi-Ticker DCA", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Precision / rounding helpers
# ──────────────────────────────────────────────────────────────────────────────
getcontext().prec = 28
SHARE_Q = Decimal("0.00001")  # broker-style 5-decimal shares
CENT    = Decimal("0.01")

def D(x) -> Decimal:
    return Decimal(str(x)) if x is not None and str(x) != "" else Decimal("0")

# ──────────────────────────────────────────────────────────────────────────────
# Supabase client (v2). Surfaces import/init errors clearly in the UI.
# ──────────────────────────────────────────────────────────────────────────────
def _import_supabase():
    try:
        from supabase import create_client
        return create_client
    except Exception as e:
        st.error(f"Supabase client not installed or failed to import: {e}")
        return None

_CREATE_CLIENT = _import_supabase()

@st.cache_resource(show_spinner=False)
def get_supabase():
    if _CREATE_CLIENT is None:
        return None
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["service_role_key"]  # server-side only
        return _CREATE_CLIENT(url, key)
    except KeyError:
        st.error('Supabase secrets missing. Add to Secrets:\n[supabase]\nurl="..."\nservice_role_key="..."')
        return None
    except Exception as e:
        st.error(f"Supabase init error: {e}")
        return None

sb = get_supabase()

# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────
BUYS_TABLE = "buys"  # expected columns: id, ticker, date, amount, price

def list_tickers() -> list[str]:
    if not sb:
        return []
    try:
        res = sb.table(BUYS_TABLE).select("ticker").execute()
        rows = res.data or []
        return sorted({r["ticker"] for r in rows if r.get("ticker")})
    except Exception as e:
        st.error(f"Could not list tickers: {e}")
        return []

def load_buys(ticker: str | None) -> pd.DataFrame:
    if not sb:
        return pd.DataFrame()
    try:
        q = sb.table(BUYS_TABLE).select("*")
        if ticker:
            q = q.eq("ticker", ticker)
        # Avoid server-side ORDER BY on a column named 'date' (quirky in some stacks)
        res = q.execute()
        df = pd.DataFrame(res.data or [])
        if df.empty:
            return df

        # Column normalization/parsing
        if "date" not in df.columns:
            st.error("Expected column 'date' not found in table 'buys'.")
            return pd.DataFrame()
        df["date"]   = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["amount"] = pd.to_numeric(df.get("amount", 0.0), errors="coerce").fillna(0.0)
        df["price"]  = pd.to_numeric(df.get("price", 0.0),  errors="coerce").fillna(0.0)
        return df.sort_values("date")
    except Exception as e:
        # Show full PostgREST payload when available
        try:
            from postgrest.exceptions import APIError
            if isinstance(e, APIError):
                import json
                st.error("Supabase API error while reading 'buys':")
                st.code(json.dumps(e.args[0], indent=2), language="json")
                return pd.DataFrame()
        except Exception:
            pass
        st.error(f"Unexpected error loading buys: {e}")
        return pd.DataFrame()

def insert_buy(ticker: str, d: date, amount: Decimal, price: Decimal):
    if not sb:
        return
    try:
        row = {
            "ticker": ticker.upper().strip(),
            "date":   d.isoformat(),
            "amount": float(amount),
            "price":  float(price),
        }
        sb.table(BUYS_TABLE).insert(row).execute()
    except Exception as e:
        st.error(f"Insert failed: {e}")

def delete_ids(ids: list[str | int]):
    if not sb or not ids:
        return
    try:
        sb.table(BUYS_TABLE).delete().in_("id", ids).execute()
    except Exception as e:
        st.error(f"Delete failed: {e}")

def clear_ticker(ticker: str):
    if not sb:
        return
    try:
        sb.table(BUYS_TABLE).delete().eq("ticker", ticker).execute()
    except Exception as e:
        st.error(f"Clear failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Math helpers (Decimal-based)
# ──────────────────────────────────────────────────────────────────────────────
def _empty_lots_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "id","ticker","date","amount","price",
        "amountD","priceD","shares_lotD","cum_sharesD","cum_investedD",
        "shares_lot","cum_shares","cum_invested"
    ])

def compute_lots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_lots_df()

    tmp = df.copy().sort_values("date")
    tmp["amountD"] = tmp["amount"].apply(D)
    tmp["priceD"]  = tmp["price"].apply(D)

    def shares_for_lot(a: Decimal, p: Decimal) -> Decimal:
        if p <= 0:
            return Decimal("0")
        return (a / p).quantize(SHARE_Q, ROUND_HALF_UP)

    tmp["shares_lotD"]   = [shares_for_lot(a, p) for a, p in zip(tmp["amountD"], tmp["priceD"])]
    tmp["cum_sharesD"]   = tmp["shares_lotD"].cumsum()
    tmp["cum_investedD"] = tmp["amountD"].cumsum()

    out = tmp.copy()
    out["shares_lot"]   = out["shares_lotD"].astype(float)
    out["cum_shares"]   = out["cum_sharesD"].astype(float)
    out["cum_invested"] = out["cum_investedD"].astype(float)
    return out

def calc_metrics(df_lots: pd.DataFrame, current_price: Decimal):
    if df_lots.empty:
        return {
            "cost_basis": Decimal("0"),
            "shares": Decimal("0"),
            "value": Decimal("0"),
            "cum_pnl": Decimal("0"),
            "ret_pct": Decimal("0.00"),
            "series": pd.DataFrame(columns=["date","portfolio_value","daily_market_move"]),
        }

    shares_total = df_lots["cum_sharesD"].iloc[-1]
    cost_basis   = df_lots["cum_investedD"].iloc[-1]

    pv_today = (shares_total * current_price).quantize(CENT, ROUND_HALF_UP)
    cum_pnl  = (pv_today - cost_basis).quantize(CENT, ROUND_HALF_UP)
    ret_pct  = (cum_pnl / cost_basis * Decimal("100")).quantize(Decimal("0.01"), ROUND_HALF_UP) if cost_basis > 0 else Decimal("0.00")

    ser = df_lots[["date","cum_sharesD","amountD"]].copy()
    ser["portfolio_valueD"] = ser["cum_sharesD"].apply(lambda s: (s * current_price).quantize(CENT, ROUND_HALF_UP))

    ser["prev_valueD"] = ser["portfolio_valueD"].shift(1).fillna(Decimal("0"))
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
        "series": ser_out[["date","portfolio_value","daily_market_move"]],
    }

def summarize_positions(all_buys: pd.DataFrame, price_map: dict[str, Decimal]) -> pd.DataFrame:
    if all_buys.empty:
        return pd.DataFrame(columns=["ticker","cost_basis","shares","price","value","pnl","ret_pct"])

    rows = []
    for tkr, grp in all_buys.groupby("ticker"):
        lots = compute_lots(grp)
        last_px = D(grp["price"].iloc[-1]) if not grp.empty else D("0")
        price = price_map.get(tkr, last_px)
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

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ret_pct"] = df["ret_pct"].round(2)
    return df.sort_values("ticker")

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("📈 Felix Abayomi – Multi-Ticker DCA Dashboard")

with st.sidebar:
    st.subheader("☁️ Cloud Save: Supabase")
    st.write(" ")
    # Diagnostics block
    st.markdown("**🔎 Diagnostics**")
    try:
        if sb:
            st.success("Supabase client: Connected")
            # quick reachability probe
            try:
                test = sb.table(BUYS_TABLE).select("id").limit(1).execute()
                st.caption(f"Table '{BUYS_TABLE}' reachable ✓  (visible rows: {len(test.data)})")
            except Exception as e:
                st.error("Table reachability failed:")
                st.code(str(e))
        else:
            st.error("Supabase client not available (check requirements & secrets).")
    except Exception as e:
        st.error(f"Diagnostics error: {e}")

    # Ticker chooser
    existing = list_tickers()
    default_tkr = "SPLG" if ("SPLG" in existing or not existing) else existing[0]
    tkr_options = sorted(set(existing + ["SPLG","SCHD","VOO","SPY"]))
    ticker = st.selectbox("Ticker", options=tkr_options, index=tkr_options.index(default_tkr))

    colR = st.columns(2)
    with colR[0]:
        if st.button("🔄 Refresh data", use_container_width=True):
            st.experimental_rerun()
    with colR[1]:
        if st.button("🧹 Clear THIS ticker", type="secondary", use_container_width=True):
            if sb:
                clear_ticker(ticker)
                st.success(f"Cleared {ticker}")
                st.experimental_rerun()

# ── Section 1: Add a buy
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
            st.experimental_rerun()
        else:
            st.error("Supabase not connected.")

# ── Load data for selected ticker
df = load_buys(ticker)

# ── Section 2: Manage entries
st.divider()
st.markdown(f"### 2) Manage entries — {ticker}")
if df.empty:
    st.info("No entries for this ticker yet.")
else:
    df_disp = df.copy()
    df_disp["shares (lot)"] = (df_disp["amount"] / df_disp["price"]).replace([float("inf")], 0.0).fillna(0.0).round(5)
    # show table (hide id in view but keep for delete selection if present)
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
                st.experimental_rerun()
            else:
                st.warning("No selectable IDs in this dataset.")

# ── Section 3: Current price input
st.divider()
st.markdown(f"### 3) Current price — {ticker}")
default_px = float(df["price"].iloc[-1]) if not df.empty else 75.0
cur_price = D(st.number_input(f"Current price for {ticker} ($)", value=float(default_px), step=0.01))

# ── Section 4: Metrics & charts
st.divider()
st.markdown(f"### 4) Metrics & charts — {ticker}")

lots = compute_lots(df)
metrics = calc_metrics(lots, cur_price)

colA, colB, colC, colD, colE = st.columns(5)
colA.metric("Cost basis",     f"${metrics['cost_basis']:,.2f}")
colB.metric("Shares",         f"{metrics['shares']:.5f}")
colC.metric("Portfolio value",f"${metrics['value']:,.2f}")
colD.metric("Cumulative P&L", f"${metrics['cum_pnl']:,.2f}")
colE.metric("Return",         f"{metrics['ret_pct']}%")

if not metrics["series"].empty:
    cL, cR = st.columns(2)
    with cL:
        st.line_chart(
            metrics["series"].set_index("date")[["portfolio_value"]],
            height=260,
            use_container_width=True
        )
    with cR:
        st.bar_chart(
            metrics["series"].set_index("date")[["daily_market_move"]],
            height=260,
            use_container_width=True
        )

st.markdown("#### Last 30 days")
if not lots.empty and not metrics["series"].empty:
    recent = metrics["series"].merge(
        lots[["date","amount","price","shares_lot","cum_shares","cum_invested"]],
        on="date",
        how="left"
    ).tail(30)
    recent = recent.rename(columns={
        "shares_lot":"shares",
        "daily_market_move":"day P&L (market)",
        "portfolio_value":"value"
    })
    st.dataframe(recent, use_container_width=True, hide_index=True)

# ── Section 5: Positions (All tickers)
st.divider()
st.markdown("### 5) Positions — all tickers")

all_buys = load_buys(None)
if all_buys.empty:
    st.info("No buys yet.")
else:
    tickers_all = sorted(all_buys["ticker"].unique().tolist())
    st.caption("Set the current price for each ticker (defaults to last buy price).")

    # up to 4 columns for nice layout
    cols = st.columns(max(1, min(4, len(tickers_all))))
    price_map: dict[str, Decimal] = {}
    for i, tkr in enumerate(tickers_all):
        col = cols[i % len(cols)]
        with col:
            last_px = float(all_buys.loc[all_buys["ticker"] == tkr, "price"].iloc[-1])
            price_map[tkr] = D(st.number_input(f"{tkr} price", value=last_px, step=0.01, key=f"px_{tkr}"))

    pos = summarize_positions(all_buys, price_map)
    if not pos.empty:
        pos["cost_basis"] = pos["cost_basis"].round(2)
        pos["value"]      = pos["value"].round(2)
        pos["pnl"]        = pos["pnl"].round(2)

        total_row = pd.DataFrame([{
            "ticker":"TOTAL",
            "cost_basis": pos["cost_basis"].sum().round(2),
            "shares": float(Decimal(str(pos["shares"].sum())).quantize(SHARE_Q, ROUND_HALF_UP)),
            "price": float("nan"),
            "value": pos["value"].sum().round(2),
            "pnl": pos["pnl"].sum().round(2),
            "ret_pct": (pos["pnl"].sum() / pos["cost_basis"].sum() * 100).round(2) if pos["cost_basis"].sum() > 0 else 0.00
        }])

        pos_show = pd.concat([pos, total_row], ignore_index=True)
        st.dataframe(pos_show, use_container_width=True, hide_index=True)
    else:
        st.info("No positions yet.")

st.caption("Tip: To match your broker to the cent, type the broker's live price into the Current price boxes.")
