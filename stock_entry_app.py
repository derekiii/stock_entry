import streamlit as st
import yfinance as yf
import pandas as pd
import re
import statistics
from datetime import datetime, timezone, timedelta, date
import time
import cloudscraper
from bs4 import BeautifulSoup
import plotly.graph_objects as go

# Global system trading parameters
DEFAULT_GLOBAL_CAPITAL = 6000.00  
RISK_PERCENT = 0.01       # 1% max risk per trade
OFFSET_PCT = 0.005        # 0.5% offset for Entry Price

# --- STREAMLIT CACHING ENGINE TO BYPASS CLOUD RATE LIMITS ---
@st.cache_data(ttl=300)
def fetch_stock_data_cached(ticker_symbol):
    """
    Multi-layer market data engine.
    Individual Yahoo endpoints can fail independently.
    """
    ticker_symbol = ticker_symbol.strip().upper()
    ticker = yf.Ticker(ticker_symbol)

    # Yahoo history can intermittently fail on cloud/shared IPs, so retry.
    full_df = pd.DataFrame()
    history_error = None
    for attempt in range(3):
        try:
            full_df = ticker.history(
                period="2y",
                interval="1d",
                auto_adjust=False,
                actions=False
            )
            if not full_df.empty:
                break
        except Exception as e:
            history_error = str(e)
        if attempt < 2:
            time.sleep(1.0 * (attempt + 1))

    if full_df.empty:
        return None, f"Unable to retrieve Yahoo historical prices after 3 attempts. {history_error or ''}".strip()

    full_df.columns = [str(col).strip() for col in full_df.columns]
    required_cols = {"Open", "High", "Low", "Close"}
    if not required_cols.issubset(full_df.columns):
        return None, "Yahoo returned incomplete OHLC history."

    full_df["EMA20"] = full_df["Close"].ewm(span=20, adjust=False).mean()
    full_df["EMA50"] = full_df["Close"].ewm(span=50, adjust=False).mean()
    full_df["EMA200"] = full_df["Close"].ewm(span=200, adjust=False).mean()

    # Each endpoint fails independently.
    try:
        info_dict = ticker.info or {}
    except Exception:
        info_dict = {}

    try:
        calendar_dict = ticker.calendar or {}
    except Exception:
        calendar_dict = {}

    # Historical earnings: use the actual earnings-history endpoint first.
    # Yahoo's earnings_dates endpoint can prioritize upcoming estimates, so
    # also request offset=1 specifically for the most recent reported results.
    earnings_history = None
    try:
        earnings_history = ticker.get_earnings_history()
    except Exception:
        try:
            earnings_history = ticker.earnings_history
        except Exception:
            earnings_history = None

    earnings_dates = None
    try:
        earnings_dates = ticker.get_earnings_dates(limit=50, offset=0)
    except Exception:
        try:
            earnings_dates = ticker.earnings_dates
        except Exception:
            earnings_dates = None

    try:
        quarterly_income = ticker.quarterly_income_stmt
    except Exception:
        quarterly_income = None

    return {
        "df": full_df,
        "info": info_dict,
        "calendar": calendar_dict,
        "earnings_dates": earnings_dates,
        "earnings_history": earnings_history,
        "quarterly_income": quarterly_income
    }, None

# --- SCRAPER ENGINE: FINVIZ COMBINED METRICS SNAPSHOT ---
@st.cache_data(ttl=1800)
def scrape_finviz_fallback_data(ticker):
    fallback = {
        "trailing_pe": "N/A",
        "forward_pe": "N/A",
        "peg_ratio": "N/A",
        "last_earnings_date": "N/A"
    }
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    scraper = cloudscraper.create_scraper()
    try:
        response = scraper.get(url, timeout=10)
        if response.status_code != 200: return fallback
        soup = BeautifulSoup(response.text, 'html.parser')
        snapshot_table = soup.find("table", class_="snapshot-table2")
        if snapshot_table:
            cells = snapshot_table.find_all("td")
            for idx, cell in enumerate(cells):
                cell_text = cell.text.strip()
                if cell_text == "P/E" and idx + 1 < len(cells):
                    val = cells[idx + 1].text.strip()
                    fallback["trailing_pe"] = float(val) if (val != "-" and val != "") else "N/A"
                elif cell_text == "Forward P/E" and idx + 1 < len(cells):
                    val = cells[idx + 1].text.strip()
                    fallback["forward_pe"] = float(val) if (val != "-" and val != "") else "N/A"
                elif cell_text == "PEG" and idx + 1 < len(cells):
                    val = cells[idx + 1].text.strip()
                    fallback["peg_ratio"] = float(val) if (val != "-" and val != "") else "N/A"
                elif cell_text == "Earnings" and idx + 1 < len(cells):
                    val = cells[idx + 1].text.strip()
                    if val != "-" and val != "":
                        try:
                            parts = val.split()
                            if len(parts) >= 2:
                                clean_date_str = f"{parts[0]} {parts[1]}"
                                current_year = datetime.now().year
                                parsed = None
                                for year in (current_year, current_year - 1):
                                    formatted_str = f"{clean_date_str} {year}"
                                    for fmt in ("%b %d %Y", "%B %d %Y"):
                                        try:
                                            candidate = datetime.strptime(formatted_str, fmt).date()
                                            # A displayed earnings date should not be
                                            # materially in the future when used as
                                            # "last earnings".
                                            if candidate <= datetime.now().date():
                                                parsed = candidate
                                                break
                                        except ValueError:
                                            continue
                                    if parsed:
                                        break
                                if parsed:
                                    fallback["last_earnings_date"] = parsed
                        except Exception:
                            fallback["last_earnings_date"] = "N/A"
    except Exception: pass
    return fallback


# --- SCRAPER FALLBACK ENGINE: STOCKANALYSIS EARNINGS ---
@st.cache_data(ttl=1800)
def scrape_stockanalysis_earnings(ticker):
    """
    Extract the reported earnings announcement date from StockAnalysis.

    StockAnalysis currently exposes an "Earnings Date" field on the stock
    overview page. This is used as a fallback when Yahoo does not return a
    usable historical earnings-calendar date.
    """
    fallback = {"last_earnings_date": None, "next_earnings_date": None}

    url = f"https://stockanalysis.com/stocks/{ticker.lower()}/"
    scraper = cloudscraper.create_scraper()

    try:
        response = scraper.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        if response.status_code != 200:
            return fallback

        soup = BeautifulSoup(response.text, "html.parser")
        page_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

        # Current StockAnalysis page format:
        # "... Earnings Date | Jul 30, 2026 ..."
        match = re.search(
            r"Earnings Date\s*\|\s*([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})",
            page_text
        )

        if match:
            try:
                fallback["last_earnings_date"] = datetime.strptime(
                    match.group(1), "%b %d, %Y"
                ).date()
            except ValueError:
                pass

        # Some pages may use "Earnings Date Jul 30, 2026" without a pipe.
        if fallback["last_earnings_date"] is None:
            match = re.search(
                r"Earnings Date\s+([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})",
                page_text
            )
            if match:
                try:
                    fallback["last_earnings_date"] = datetime.strptime(
                        match.group(1), "%b %d, %Y"
                    ).date()
                except ValueError:
                    pass

    except Exception:
        pass

    return fallback


# --- SCRAPER FALLBACK ENGINE 2: MARKETBEAT TARGETS ---
@st.cache_data(ttl=1800)
def scrape_marketbeat_fallback_data(ticker):
    fallback = {
        "trailing_pe": "N/A", 
        "next_earnings_date": None,
        "post_earnings_median_matp": None
    }
    url = f"https://www.marketbeat.com/stocks/NYSE/{ticker}/forecast/"
    scraper = cloudscraper.create_scraper()
    try:
        response = scraper.get(url, timeout=10)
        if response.status_code != 200:
            alt_url = f"https://www.marketbeat.com/stocks/NASDAQ/{ticker}/forecast/"
            response = scraper.get(alt_url, timeout=10)
        if response.status_code != 200: return fallback
            
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text()
        pe_match = re.search(r'P/E\s+ratio\s+of\s+(\d+(?:\.\d+)?)', text_content, re.IGNORECASE)
        if pe_match: fallback["trailing_pe"] = float(pe_match.group(1))

        history_table = None
        for table in soup.find_all("table"):
            first_row = table.find("tr")
            if first_row:
                header_cells = [cell.text.lower().strip() for cell in first_row.find_all(["th", "td"])]
                if any("date" in h for h in header_cells) and any("brokerage" in h for h in header_cells):
                    history_table = table
                    break
                    
        if history_table:
            header_cells = [cell.text.lower().strip() for cell in history_table.find("tr").find_all(["th", "td"])]
            date_idx = next((i for i, h in enumerate(header_cells) if "date" in h), 0)
            target_idx = next((i for i, h in enumerate(header_cells) if "target" in h), 3)
            scraped_dates = []
            post_earnings_targets = []
            
            for row in history_table.find_all("tr"):
                cols = row.find_all(["td", "th"])
                if len(cols) <= max(date_idx, target_idx): continue
                raw_date_str = cols[date_idx].text.strip()
                raw_target_str = cols[target_idx].text.strip()
                if "date" in raw_date_str.lower() or "brokerage" in raw_date_str.lower(): continue
                cleaned_date_str = re.sub(r'^[A-Za-z]+,\s+', '', raw_date_str)
                cleaned_date_str = re.sub(r'\s+', ' ', cleaned_date_str).replace(",", "").replace(".", "").strip()
                
                row_date = None
                for fmt in ("%m/%d/%Y", "%b %d %Y", "%B %d %Y", "%m/%d/%y"):
                    try:
                        row_date = datetime.strptime(cleaned_date_str, fmt).date()
                        break
                    except ValueError: continue
                
                if row_date:
                    scraped_dates.append(row_date)
                    final_target_segment = raw_target_str.split("➝")[-1].strip() if "➝" in raw_target_str else raw_target_str
                    numeric_match = re.search(r'\d+(?:\.\d+)?', final_target_segment.replace(",", ""))
                    if numeric_match: post_earnings_targets.append(float(numeric_match.group(0)))
            
            if scraped_dates:
                today = datetime.now(timezone.utc).date()
                futures = [d for d in scraped_dates if d > today]
                if futures: fallback["next_earnings_date"] = min(futures)
            if post_earnings_targets: fallback["post_earnings_median_matp"] = statistics.median(post_earnings_targets)
    except Exception: pass
    return fallback

# --- PROFILE MATRICES GENERATORS ---
def _to_date(value):
    """Convert common Yahoo/HTML date values into a date."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def get_yahoo_earnings_dates(earnings_dates, earnings_history=None):
    """
    Return the latest actual earnings-calendar announcement date and the
    nearest future earnings-calendar date.

    IMPORTANT:
    earnings_dates is used for announcement dates. earnings_history is NOT
    used for this field because its dates can represent reporting-period data.
    """
    today = datetime.now(timezone.utc).date()
    past_dates = []
    future_dates = []

    if earnings_dates is None:
        return None, None

    try:
        if isinstance(earnings_dates, pd.DataFrame):
            raw_dates = list(earnings_dates.index)
        elif isinstance(earnings_dates, pd.Series):
            raw_dates = list(earnings_dates.index)
        else:
            raw_dates = []
    except Exception:
        raw_dates = []

    for raw in raw_dates:
        d = _to_date(raw)
        if not d:
            continue
        if d <= today:
            past_dates.append(d)
        else:
            future_dates.append(d)

    return (
        max(past_dates) if past_dates else None,
        min(future_dates) if future_dates else None
    )


def get_earnings_diagnostics(cached_earnings_dates, finviz_data, stockanalysis_data=None):
    yahoo_past, yahoo_next = get_yahoo_earnings_dates(cached_earnings_dates)
    return {
        "yahoo_last": yahoo_past,
        "yahoo_next": yahoo_next,
        "stockanalysis_last": (
            stockanalysis_data.get("last_earnings_date")
            if stockanalysis_data else None
        ),
        "finviz_last": (
            finviz_data.get("last_earnings_date")
            if finviz_data else "N/A"
        )
    }


def get_earnings_profile(ticker_symbol, cached_calendar, cached_earnings_dates,
                         cached_earnings_history, cached_financials, finviz_data,
                         stockanalysis_data, mb_fallback):
    today_date = datetime.now(timezone.utc).date()

    profile = {
        "past_date": "N/A", "past_elapsed": "N/A", "past_days_val": None,
        "next_date": "N/A", "next_days": "N/A", "next_days_val": None,
        "past_source": "N/A", "next_source": "N/A",
        "trend_str": "", "is_3q_uptrend": False
    }

    pst_dt = None
    nxt_dt = None

    # Last earnings = actual earnings announcement date.
    #
    # Priority:
    # 1. Yahoo earnings calendar
    # 2. StockAnalysis earnings-date field
    # 3. Finviz snapshot
    #
    # We intentionally do NOT use earnings_history here because its date
    # can represent a reporting/fiscal period rather than the announcement.
    yahoo_past, yahoo_next = get_yahoo_earnings_dates(cached_earnings_dates)

    if yahoo_past:
        pst_dt = yahoo_past
        profile["past_source"] = "Yahoo"
    elif stockanalysis_data.get("last_earnings_date"):
        pst_dt = _to_date(stockanalysis_data.get("last_earnings_date"))
        if pst_dt:
            profile["past_source"] = "StockAnalysis"
    elif finviz_data.get("last_earnings_date") != "N/A":
        pst_dt = _to_date(finviz_data.get("last_earnings_date"))
        if pst_dt:
            profile["past_source"] = "Finviz"

    # Next earnings: Yahoo calendar first.
    try:
        if cached_calendar and "Earnings Date" in cached_calendar:
            dates = cached_calendar["Earnings Date"]
            if isinstance(dates, (list, tuple, pd.Series)):
                parsed = sorted(
                    d for d in (_to_date(x) for x in dates) if d
                )
                futures = [d for d in parsed if d >= today_date]
                if futures:
                    nxt_dt = futures[0]
                    profile["next_source"] = "Yahoo"
    except Exception:
        pass

    if not nxt_dt and yahoo_next:
        nxt_dt = yahoo_next
        profile["next_source"] = "Yahoo"

    if not nxt_dt and mb_fallback.get("next_earnings_date"):
        nxt_dt = _to_date(mb_fallback.get("next_earnings_date"))
        if nxt_dt:
            profile["next_source"] = "MarketBeat"

    if pst_dt:
        profile["past_date"] = pst_dt.strftime("%b %d, %Y")
        profile["past_days_val"] = (today_date - pst_dt).days
        profile["past_elapsed"] = f"{profile['past_days_val']}d ago"

    if nxt_dt:
        profile["next_date"] = nxt_dt.strftime("%b %d, %Y")
        profile["next_days_val"] = (nxt_dt - today_date).days
        profile["next_days"] = (
            f"{profile['next_days_val']}d away"
            if profile["next_days_val"] > 0 else "Today"
        )

    try:
        q_income = cached_financials
        if q_income is not None and not q_income.empty and "Net Income" in q_income.index:
            net_incomes = q_income.loc["Net Income"].tolist()
            net_incomes = [float(x) for x in net_incomes if pd.notna(x)]
            pct_values = []
            for i in range(min(3, len(net_incomes) - 1)):
                prev = net_incomes[i + 1]
                pct_values.append(
                    ((net_incomes[i] - prev) / abs(prev)) * 100 if prev else 0.0
                )
            pct_values.reverse()
            trend_formatted = [
                f"{'▲' if p > 0 else '▼' if p < 0 else '►'} {int(p)}%"
                for p in pct_values
            ]
            profile["trend_str"] = " | Trends: " + " -> ".join(trend_formatted)
            profile["is_3q_uptrend"] = len(pct_values) >= 3 and all(p > 0 for p in pct_values)
    except Exception:
        pass

    return profile


def find_strong_support_levels(df, current_price, lookback=180,
                               pivot_window=3, tolerance_pct=0.012):
    """
    Detect recent swing-low support zones below current price.

    A support zone becomes stronger when it has:
    - multiple swing-low touches
    - recent tests
    - bullish rejection from the low
    - above-normal volume on a test

    The returned candidates are ranked by significance first and proximity
    second, so the app doesn't blindly choose the nearest weak low.
    """
    if df is None or df.empty:
        return []

    data = df.tail(lookback).copy()
    if len(data) < pivot_window * 2 + 5:
        return []

    lows = pd.to_numeric(data["Low"], errors="coerce")
    highs = pd.to_numeric(data["High"], errors="coerce")
    closes = pd.to_numeric(data["Close"], errors="coerce")
    volumes = (
        pd.to_numeric(data["Volume"], errors="coerce")
        if "Volume" in data.columns else None
    )

    candidates = []

    for i in range(pivot_window, len(data) - pivot_window):
        low = lows.iloc[i]
        if pd.isna(low) or low >= current_price:
            continue

        left = lows.iloc[i - pivot_window:i]
        right = lows.iloc[i + 1:i + pivot_window + 1]

        if low <= left.min() and low <= right.min():
            high = highs.iloc[i]
            close = closes.iloc[i]
            candle_range = max(high - low, 1e-9)
            rejection = max(0.0, min(1.0, (close - low) / candle_range))

            volume_factor = 1.0
            if volumes is not None:
                recent_vol = volumes.iloc[max(0, i - 20):i + 1]
                median_vol = recent_vol.median()
                if pd.notna(median_vol) and median_vol > 0 and pd.notna(volumes.iloc[i]):
                    volume_factor = min(1.5, max(0.7, volumes.iloc[i] / median_vol))

            recency_days = len(data) - 1 - i
            recency_factor = 1.0 / (1.0 + recency_days / 45.0)

            candidates.append({
                "price": float(low),
                "rejection": rejection,
                "volume_factor": float(volume_factor),
                "recency_factor": float(recency_factor),
                "date": data.index[i],
            })

    if not candidates:
        return []

    # Cluster lows within 1.2% into one support zone.
    zones = []
    for c in sorted(candidates, key=lambda x: x["price"], reverse=True):
        assigned = None
        for zone in zones:
            if abs(c["price"] - zone["price"]) / zone["price"] <= tolerance_pct:
                assigned = zone
                break
        if assigned:
            assigned["members"].append(c)
            assigned["price"] = sum(x["price"] for x in assigned["members"]) / len(assigned["members"])
        else:
            zones.append({"price": c["price"], "members": [c]})

    scored = []
    for zone in zones:
        members = zone["members"]
        touches = len(members)
        recency = max(x["recency_factor"] for x in members)
        rejection = sum(x["rejection"] for x in members) / touches
        volume_factor = sum(x["volume_factor"] for x in members) / touches

        score = (
            2.5 * min(touches, 4)
            + 2.0 * recency
            + 1.5 * rejection
            + 1.0 * volume_factor
        )

        distance_pct = (current_price - zone["price"]) / current_price

        scored.append({
            "price": zone["price"],
            "score": score,
            "touches": touches,
            "distance_pct": distance_pct,
            "last_test": max(x["date"] for x in members),
        })

    scored = [x for x in scored if x["price"] < current_price]
    scored.sort(key=lambda x: (x["score"], -x["distance_pct"]), reverse=True)

    return scored[:8]


def select_best_support(df, current_price):
    candidates = find_strong_support_levels(df, current_price)

    if candidates:
        return candidates[0]["price"], candidates

    # EMA is now only a safety fallback, not the primary support engine.
    ema20 = float(df["EMA20"].iloc[-1])
    ema50 = float(df["EMA50"].iloc[-1])
    below = [x for x in (ema20, ema50) if x < current_price]

    if below:
        return max(below), []

    return float(df["Low"].tail(20).min()), []


# --- STREAMLIT WEB APP UI INTERFACE ---
st.set_page_config(page_title="Entry Matrix Terminal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .stApp { background-color: #121212; color: #ffffff; }
        div[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: bold !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Entry Matrix Terminal")

with st.sidebar:
    st.header("⚙️ Configuration Engine")
    ticker_input = st.text_input("Ticker Symbol", value="", placeholder="e.g. AAPL").strip().upper()
    trading_capital = st.number_input("Trading Capital ($)", value=DEFAULT_GLOBAL_CAPITAL, step=500.0)
    st.markdown("---")

if ticker_input:
    with st.spinner(f"Analyzing {ticker_input} profiles safely from multi-layer data channels..."):
        dataset, error_msg = fetch_stock_data_cached(ticker_input)
        
        if error_msg:
            st.error(error_msg)
            st.stop()
        if dataset is None:
            st.warning("No structural profile returned from core cache pool layer. Try again in a brief moment.")
            st.stop()
            
        try:
            full_df = dataset["df"]
            info = dataset["info"]
            calendar = dataset["calendar"]
            quarterly_income = dataset["quarterly_income"]
            earnings_dates = dataset.get("earnings_dates")
            earnings_history = dataset.get("earnings_history")
            
            chart_df = full_df.tail(63).copy()
            
            ema20 = float(full_df["EMA20"].iloc[-1])
            ema50 = float(full_df["EMA50"].iloc[-1])
            ema200 = float(full_df["EMA200"].iloc[-1])
            
            tr = pd.concat([
                full_df["High"] - full_df["Low"], 
                (full_df["High"] - full_df["Close"].shift()).abs(), 
                (full_df["Low"] - full_df["Close"].shift()).abs()
            ], axis=1).max(axis=1)
            
            extracted_atr = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
            current_price = float(full_df["Close"].iloc[-1])

            # Initialize support variables BEFORE any UI code can reference them.
            # EMA20/EMA50 are used only as a fallback inside select_best_support().
            support_candidates = []
            default_support = float(full_df["Low"].tail(20).min())
            default_support, support_candidates = select_best_support(full_df, current_price)

            # Singular parsing pipeline for fundamentals and last earnings date
            finviz_data = scrape_finviz_fallback_data(ticker_input)
            stockanalysis_data = scrape_stockanalysis_earnings(ticker_input)
            mb_data = scrape_marketbeat_fallback_data(ticker_input)
            
            sector_name = info.get('sector', 'N/A') if info else 'N/A'
            industry_name = info.get('industry', 'N/A') if info else 'N/A'
            detailed_sector_str = f"{sector_name} - {industry_name}" if industry_name != 'N/A' else sector_name
                
            trailing_pe = info.get("trailingPE", "N/A") if info else "N/A"
            if trailing_pe == "N/A": trailing_pe = finviz_data["trailing_pe"]
            if trailing_pe == "N/A": trailing_pe = mb_data["trailing_pe"]
                
            forward_pe = info.get("forwardPE", "N/A") if info else "N/A"
            if forward_pe == "N/A": forward_pe = finviz_data["forward_pe"]
                
            peg_ratio = info.get("pegRatio", "N/A") if info else "N/A"
            if peg_ratio == "N/A": peg_ratio = finviz_data["peg_ratio"]
                
            target_mean_price = info.get("targetMeanPrice") if info else None
            scraped_matp = mb_data["post_earnings_median_matp"] or target_mean_price or current_price
            
            earn = get_earnings_profile(
                ticker_input,
                calendar,
                earnings_dates,
                earnings_history,
                quarterly_income,
                finviz_data,
                stockanalysis_data,
                mb_data
            )
            
            def style_metric_val(val, threshold, is_peg=False):
                if val == "N/A" or not isinstance(val, (int, float)): return f"`{val}`"
                formatted_val = f"{val:.2f}"
                is_good = (val <= threshold) if not is_peg else (0 < val <= threshold)
                return f'<span style="color:{"#00e676" if is_good else "#ff5252"}; font-weight:bold;">{formatted_val}</span>'

            pe_styled = style_metric_val(trailing_pe, 30.0)
            fwd_pe_styled = style_metric_val(forward_pe, 30.0)
            peg_styled = style_metric_val(peg_ratio, 2.0, is_peg=True)
            
            def style_earnings_date(date_str, label, days_val):
                if date_str == "N/A" or days_val is None: return f"`{date_str}`"
                if abs(days_val) <= 7: return f'<span style="color:#ff5252; font-weight:bold;">{date_str} ({label})</span>'
                return f'`{date_str}` ({label})'

            last_earn_styled = style_earnings_date(earn["past_date"], earn["past_elapsed"], earn["past_days_val"])
            next_earn_styled = style_earnings_date(earn["next_date"], earn["next_days"], earn["next_days_val"])

            workspace_left, workspace_right = st.columns([1, 1.2])
            
            with workspace_left:
                st.subheader("📊 Core Market Analysis Profile")
                st.metric("Current Price", f"${current_price:.2f}")
                st.markdown(f"**Sector Info:** `{detailed_sector_str}`")
                trend_status = "🟩 **PERFECT UPTREND (EMA STACK)**" if (ema20 > ema50 > ema200) else "🟥 **NO CLEAR TREND / CONSOLIDATION**"
                st.markdown(f"**Trend State:** {trend_status}")

                if support_candidates:
                    best = support_candidates[0]
                    st.markdown(
                        f"**Auto Support:** `${best['price']:.2f}` "
                        f"({best['touches']} swing-low touch(es), "
                        f"{best['distance_pct']*100:.1f}% below price)"
                    )
                else:
                    st.markdown("**Auto Support:** `EMA fallback`")
                
                st.markdown(f"**Trailing P/E:** {pe_styled}", unsafe_allow_html=True)
                st.markdown(f"**Forward P/E:** {fwd_pe_styled}", unsafe_allow_html=True)
                st.markdown(f"**PEG Ratio:** {peg_styled}", unsafe_allow_html=True)
                st.markdown(f"**MATP Price:** `${scraped_matp:.2f}`")
                st.markdown(
                    f"**Last Earnings:** {last_earn_styled} "
                    f"<small>[{earn['past_source']}]</small>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Next Earnings:** {next_earn_styled} <small>[{earn["next_source"]}]</small>", unsafe_allow_html=True)
                
                qh_text = "🟢 **3Q Continuous Growth Uptrend**" if earn['is_3q_uptrend'] else "📋 **Mixed Growth Matrix**"
                st.markdown(f"**Quarterly Income Health:** {qh_text} {earn['trend_str']}")
                st.markdown("---")
                
                st.subheader("⚙️ Interactive Formula Adjustments")
                
                default_resistance = scraped_matp
                
                # Setup session state metrics explicitly on ticker switch
                if "prev_ticker" not in st.session_state or st.session_state.prev_ticker != ticker_input:
                    st.session_state.prev_ticker = ticker_input
                    st.session_state.val_support = float(default_support)
                    st.session_state.val_resistance = float(default_resistance)
                    st.session_state.val_entry = float(default_support * (1 + OFFSET_PCT))
                    st.session_state.val_target = float(default_resistance * (1 - 0.002))
                    st.session_state.val_atr_mult = 1.5  # Initialized ATR multiplier state
                    st.session_state.val_stop = float(default_support - (1.5 * extracted_atr))

                # --- INSTANT SYNCHRONIZATION CALLBACK LAYERS ---
                def on_support_change():
                    st.session_state.val_entry = st.session_state.val_support * (1 + OFFSET_PCT)
                    st.session_state.val_stop = st.session_state.val_support - (st.session_state.val_atr_mult * extracted_atr)

                def on_resistance_change():
                    st.session_state.val_target = st.session_state.val_resistance * (1 - 0.002)

                def on_mult_change():
                    # Recalculates stop loss directly when the multiplier number changes
                    st.session_state.val_stop = st.session_state.val_support - (st.session_state.val_atr_mult * extracted_atr)

                grid_col1, grid_col2 = st.columns(2)
                with grid_col1:
                    st.number_input("Support Level", key="val_support", step=0.5, on_change=on_support_change)
                    st.number_input("ATR Multiplier", key="val_atr_mult", step=0.1, min_value=0.1, on_change=on_mult_change)
                    st.number_input("Entry Price", key="val_entry", step=0.5)
                    st.number_input("Stop Loss", key="val_stop", step=0.5)
                with grid_col2:
                    st.number_input("Resistance Level (MATP Source)", key="val_resistance", step=0.5, on_change=on_resistance_change)
                    st.number_input("Profit Target", key="val_target", step=0.5)
                
                entry_final = st.session_state.val_entry
                stop_final = st.session_state.val_stop
                target_final = st.session_state.val_target
                atr_mult_final = st.session_state.val_atr_mult
                
                unit_risk = abs(entry_final - stop_final)
                unit_reward = abs(target_final - entry_final)
                ror = unit_reward / unit_risk if unit_risk > 0 else 0.0
                
                max_allowed_risk_dollars = trading_capital * RISK_PERCENT
                units = int(max_allowed_risk_dollars / unit_risk) if unit_risk > 0 else 0
                potential_profit = unit_reward * units
                potential_loss = unit_risk * units
                
                st.markdown("---")
                st.subheader("🏆 Expected Formula Execution Output")
                
                st.markdown(f"• **Entry Price:** `${entry_final:.2f}`")
                st.markdown(f"• **Profit Target:** `${target_final:.2f}`")
                st.markdown(f"• **Stop Loss:** `${stop_final:.2f}` *(Using {atr_mult_final:.1f}x ATR)*")
                st.markdown(f"• **ATR (14d Volatility):** `{extracted_atr:.2f}`")
                
                ror_indicator = "✅ Safe Metric" if ror >= 2.5 else ("⚠️ Moderate" if ror >= 2.0 else "❌ Warning Low")
                st.markdown(f"• **Reward over Risk (RoR):** `{ror:.2f}` ({ror_indicator})")
                st.markdown(f"• **Max Units (1% Risk Allocation):** `{units}` shares *(Allocated risk: ${units*unit_risk:.2f})*")
                
                st.success(f"Potential Profit: **+${potential_profit:.2f}**")
                st.error(f"Potential Loss: **-${potential_loss:.2f}**")
                
            with workspace_right:
                st.subheader("📈 Strategic Entry Matrix Visualization")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=chart_df.index, open=chart_df['Open'], high=chart_df['High'],
                    low=chart_df['Low'], close=chart_df['Close'],
                    increasing_line_color='#00e676', decreasing_line_color='#ff5252',
                    name="Price"
                ))
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA20'], line=dict(color='#ff5252', width=1.5), name="EMA20"))
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA50'], line=dict(color='#00e676', width=1.5), name="EMA50"))
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA200'], line=dict(color='#e040fb', width=1.8), name="EMA200"))
                
                fig.add_trace(go.Scatter(
                    x=[chart_df.index.min(), chart_df.index.max()], y=[target_final, target_final],
                    mode="lines", line=dict(color="#00e5ff", width=2), name=f"Target: ${target_final:.2f}"
                ))
                fig.add_trace(go.Scatter(
                    x=[chart_df.index.min(), chart_df.index.max()], y=[entry_final, entry_final],
                    mode="lines", line=dict(color="#2196F3", width=2), name=f"Entry Price: ${entry_final:.2f}"
                ))
                fig.add_trace(go.Scatter(
                    x=[chart_df.index.min(), chart_df.index.max()], y=[stop_final, stop_final],
                    mode="lines", line=dict(color="#ff9800", width=2, dash="dash"), name=f"Stop Loss: ${stop_final:.2f}"
                ))
                
                fig.update_layout(
                    title=f"{ticker_input} Technical Matrix", template="plotly_dark",
                    paper_bgcolor="#121212", plot_bgcolor="#1e1e1e", xaxis_rangeslider_visible=False,
                    height=700, margin=dict(l=10, r=10, t=40, b=10), showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Execution Error Parsing Parameters: {str(e)}")
else:
    st.info("💡 Enter a stock ticker symbol in the configuration sidebar to initialize the real-time visual web entry terminal.")