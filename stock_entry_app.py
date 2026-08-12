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
                parsed = datetime.strptime(
                    match.group(1), "%b %d, %Y"
                ).date()

                if parsed > datetime.now(timezone.utc).date():
                    fallback["next_earnings_date"] = parsed
                else:
                    fallback["last_earnings_date"] = parsed
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
                    parsed = datetime.strptime(
                        match.group(1), "%b %d, %Y"
                    ).date()

                    if parsed > datetime.now(timezone.utc).date():
                        fallback["next_earnings_date"] = parsed
                    else:
                        fallback["last_earnings_date"] = parsed
                except ValueError:
                    pass

    except Exception:
        pass

    return fallback


# --- SCRAPER FALLBACK ENGINE 2: MARKETBEAT TARGETS ---
@st.cache_data(ttl=1800)
@st.cache_data(ttl=1800)
def scrape_marketbeat_fallback_data(ticker):
    """
    Scrape MarketBeat analyst target history.

    The raw analyst revisions are retained so MATP can be calculated using
    only revisions made after the latest earnings announcement.
    """
    fallback = {
        "trailing_pe": "N/A",
        "next_earnings_date": None,
        "analyst_target_history": [],
        "post_earnings_median_matp": None
    }

    url = f"https://www.marketbeat.com/stocks/NYSE/{ticker}/forecast/"
    scraper = cloudscraper.create_scraper()

    try:
        response = scraper.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        if response.status_code != 200:
            alt_url = f"https://www.marketbeat.com/stocks/NASDAQ/{ticker}/forecast/"
            response = scraper.get(
                alt_url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )

        if response.status_code != 200:
            return fallback

        soup = BeautifulSoup(response.text, "html.parser")
        text_content = soup.get_text(" ")

        pe_match = re.search(
            r'P/E\s+ratio\s+of\s+(\d+(?:\.\d+)?)',
            text_content,
            re.IGNORECASE
        )
        if pe_match:
            fallback["trailing_pe"] = float(pe_match.group(1))

        history_table = None
        for table in soup.find_all("table"):
            first_row = table.find("tr")
            if first_row:
                header_cells = [
                    cell.text.lower().strip()
                    for cell in first_row.find_all(["th", "td"])
                ]
                if (
                    any("date" in h for h in header_cells)
                    and any("brokerage" in h for h in header_cells)
                ):
                    history_table = table
                    break

        if history_table:
            header_cells = [
                cell.text.lower().strip()
                for cell in history_table.find("tr").find_all(["th", "td"])
            ]

            date_idx = next(
                (i for i, h in enumerate(header_cells) if "date" in h),
                0
            )
            brokerage_idx = next(
                (i for i, h in enumerate(header_cells) if "brokerage" in h),
                None
            )
            target_idx = next(
                (i for i, h in enumerate(header_cells) if "target" in h),
                3
            )

            for row in history_table.find_all("tr"):
                cols = row.find_all(["td", "th"])

                if len(cols) <= max(
                    date_idx,
                    target_idx,
                    brokerage_idx if brokerage_idx is not None else 0
                ):
                    continue

                raw_date_str = cols[date_idx].text.strip()
                raw_target_str = cols[target_idx].text.strip()

                if (
                    "date" in raw_date_str.lower()
                    or "brokerage" in raw_date_str.lower()
                ):
                    continue

                cleaned_date_str = re.sub(
                    r'^[A-Za-z]+,\s+',
                    '',
                    raw_date_str
                )
                cleaned_date_str = (
                    re.sub(r'\s+', ' ', cleaned_date_str)
                    .replace(",", "")
                    .replace(".", "")
                    .strip()
                )

                row_date = None
                for fmt in (
                    "%m/%d/%Y",
                    "%b %d %Y",
                    "%B %d %Y",
                    "%m/%d/%y"
                ):
                    try:
                        row_date = datetime.strptime(
                            cleaned_date_str,
                            fmt
                        ).date()
                        break
                    except ValueError:
                        continue

                if not row_date:
                    continue

                final_target_segment = (
                    raw_target_str.split("➝")[-1].strip()
                    if "➝" in raw_target_str
                    else raw_target_str
                )

                numeric_match = re.search(
                    r'\d+(?:\.\d+)?',
                    final_target_segment.replace(",", "")
                )

                if not numeric_match:
                    continue

                target_value = float(numeric_match.group(0))

                brokerage = (
                    cols[brokerage_idx].text.strip()
                    if brokerage_idx is not None
                    else "Unknown"
                )

                fallback["analyst_target_history"].append({
                    "date": row_date,
                    "brokerage": brokerage,
                    "target": target_value
                })

            # The MarketBeat page also contains upcoming earnings information
            # in some layouts. Preserve the existing future-date extraction.
            scraped_dates = [
                x["date"] for x in fallback["analyst_target_history"]
            ]
            today = datetime.now(timezone.utc).date()
            futures = [d for d in scraped_dates if d > today]
            if futures:
                fallback["next_earnings_date"] = min(futures)

    except Exception:
        pass

    return fallback


def calculate_post_earnings_matp(marketbeat_data, last_earnings_date):
    """
    Calculate MATP using only the latest analyst target from each brokerage
    after the latest earnings announcement.

    This avoids:
      - using pre-earnings targets
      - counting multiple revisions from the same analyst repeatedly

    Returns (MATP, number_of_analysts_used).
    """
    if not last_earnings_date:
        return None, 0

    history = marketbeat_data.get("analyst_target_history", [])
    if not history:
        return None, 0

    post_earnings = [
        row for row in history
        if row.get("date") and row["date"] > last_earnings_date
    ]

    if not post_earnings:
        return None, 0

    # Latest target per brokerage after earnings.
    latest_by_brokerage = {}

    for row in post_earnings:
        brokerage = row.get("brokerage") or "Unknown"
        existing = latest_by_brokerage.get(brokerage)

        if existing is None or row["date"] > existing["date"]:
            latest_by_brokerage[brokerage] = row

    targets = [
        row["target"]
        for row in latest_by_brokerage.values()
        if isinstance(row.get("target"), (int, float))
    ]

    if not targets:
        return None, 0

    return statistics.median(targets), len(targets)

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
        "stockanalysis_next": (
            stockanalysis_data.get("next_earnings_date")
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
        candidate = _to_date(
            stockanalysis_data.get("last_earnings_date")
        )
        # StockAnalysis' overview "Earnings Date" can represent the next
        # scheduled earnings event. Never treat a future date as "Last Earnings".
        if candidate and candidate <= today_date:
            pst_dt = candidate
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


def _support_touch_strength(df, level, tolerance_pct=0.012):
    if df is None or df.empty:
        return 0, 0.0
    recent = df.tail(180)
    lows = pd.to_numeric(recent["Low"], errors="coerce").dropna()
    if lows.empty:
        return 0, 0.0

    hits = lows[((lows - level).abs() / level) <= tolerance_pct]
    recency_score = 0.0
    for idx in hits.index:
        try:
            days_ago = max(0, (recent.index[-1] - idx).days)
        except Exception:
            days_ago = 180
        recency_score += max(0.0, 1.0 - days_ago / 180.0)
    return int(len(hits)), recency_score


def _build_price_support_candidates(df, current_price):
    candidates = []
    recent = df.tail(180).copy()
    lows = pd.to_numeric(recent["Low"], errors="coerce")

    for window in (3, 5):
        rolling_min = lows.rolling(
            window * 2 + 1, center=True, min_periods=1
        ).min()
        swing_idx = recent.index[lows.eq(rolling_min)]

        for idx in swing_idx:
            level = float(recent.loc[idx, "Low"])
            if level >= current_price:
                continue

            touches, recency = _support_touch_strength(
                recent, level, tolerance_pct=0.012
            )
            distance_pct = (current_price - level) / current_price

            try:
                row = recent.loc[idx]
                candle_range = max(
                    float(row["High"] - row["Low"]), 1e-9
                )
                rejection = max(
                    0.0,
                    min(
                        1.0,
                        (float(row["Close"]) - float(row["Low"]))
                        / candle_range
                    )
                )
            except Exception:
                rejection = 0.5

            candidates.append({
                "price": level,
                "type": "Price Support",
                "touches": touches,
                "recency": recency,
                "rejection": rejection,
                "distance_pct": distance_pct,
            })

    # Cluster price levels within 1.5%.
    candidates.sort(key=lambda x: x["price"], reverse=True)
    zones = []

    for c in candidates:
        matched = None
        for z in zones:
            if abs(c["price"] - z["price"]) / z["price"] <= 0.015:
                matched = z
                break

        if matched:
            old_touches = matched["touches"]
            new_touches = old_touches + c["touches"]
            if new_touches:
                matched["price"] = (
                    matched["price"] * old_touches
                    + c["price"] * c["touches"]
                ) / new_touches
            matched["touches"] = new_touches
            matched["recency"] = max(
                matched["recency"], c["recency"]
            )
            matched["rejection"] = max(
                matched["rejection"], c["rejection"]
            )
            matched["distance_pct"] = (
                current_price - matched["price"]
            ) / current_price
        else:
            zones.append(c)

    return zones


def _score_support_candidate(
    candidate, current_price, ema20, ema50, ema200, atr
):
    level = candidate["price"]
    atr_distance = (
        (current_price - level) / atr
        if atr > 0 else 999.0
    )

    # Proximity is deliberately dominant for ENTRY support.
    if atr_distance <= 1.5:
        proximity = max(
            0.35,
            1.0 - abs(atr_distance - 0.75) / 1.5
        )
    elif atr_distance <= 2.5:
        proximity = 0.30
    else:
        proximity = 0.03

    touch_score = min(candidate["touches"], 5) / 5.0
    recency_score = min(candidate["recency"], 5.0) / 5.0
    rejection_score = candidate.get("rejection", 0.5)

    ema_bonus = 0.0
    if abs(level - ema20) / current_price <= 0.012:
        ema_bonus += 0.20
    if abs(level - ema50) / current_price <= 0.012:
        ema_bonus += 0.10
    if abs(level - ema200) / current_price <= 0.015:
        ema_bonus += 0.05

    candidate["atr_distance"] = atr_distance
    candidate["ema_confluence"] = ema_bonus
    candidate["score"] = (
        0.40 * proximity
        + 0.25 * touch_score
        + 0.20 * recency_score
        + 0.10 * rejection_score
        + 0.05 * min(1.0, ema_bonus * 2.0)
    )
    return candidate


def find_three_tier_supports(df, current_price):
    """
    Three support tiers:
      Entry    = nearest credible support suitable for a current swing entry
      Secondary = next meaningful lower support
      Major    = deeper structural support
    """
    ema20 = float(df["EMA20"].iloc[-1])
    ema50 = float(df["EMA50"].iloc[-1])
    ema200 = float(df["EMA200"].iloc[-1])

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)

    atr = float(
        tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    )

    candidates = _build_price_support_candidates(
        df, current_price
    )

    # Add dynamic supports as candidates, but let the scoring decide.
    for name, value in (
        ("EMA20", ema20),
        ("EMA50", ema50),
        ("EMA200", ema200),
    ):
        if value < current_price:
            touches, recency = _support_touch_strength(
                df.tail(180), value, tolerance_pct=0.012
            )
            candidates.append({
                "price": value,
                "type": name,
                "touches": touches,
                "recency": recency,
                "rejection": 0.5,
                "distance_pct": (
                    current_price - value
                ) / current_price,
            })

    candidates = [
        _score_support_candidate(
            c, current_price, ema20, ema50, ema200, atr
        )
        for c in candidates
        if c["price"] < current_price
    ]

    # Deduplicate close levels.
    candidates.sort(key=lambda x: x["price"], reverse=True)
    deduped = []
    for c in candidates:
        if not any(
            abs(c["price"] - x["price"]) / x["price"] <= 0.008
            for x in deduped
        ):
            deduped.append(c)

    # ENTRY: prefer a credible level within 1.5 ATR.
    near = [c for c in deduped if c["atr_distance"] <= 1.5]

    if near:
        near.sort(
            key=lambda x: (x["score"], -x["distance_pct"]),
            reverse=True
        )
        entry = near[0]
    else:
        medium = [
            c for c in deduped
            if c["atr_distance"] <= 2.5
        ]
        medium.sort(
            key=lambda x: (x["score"], -x["distance_pct"]),
            reverse=True
        )
        entry = medium[0] if medium else None

    # SECONDARY: nearest meaningful level below entry.
    secondary = None
    if entry:
        lower = [
            c for c in deduped
            if c["price"] < entry["price"] * 0.985
        ]
        lower.sort(
            key=lambda x: (x["distance_pct"], -x["score"])
        )
        secondary = lower[0] if lower else None

    # MAJOR: strongest repeated price support, irrespective of distance.
    structural = [
        c for c in deduped
        if c["type"] == "Price Support"
        and c["touches"] >= 2
    ]
    structural.sort(
        key=lambda x: (
            x["touches"] * 2
            + x["recency"]
            + x["rejection"]
        ),
        reverse=True
    )
    major = structural[0] if structural else None

    if major and entry:
        if abs(major["price"] - entry["price"]) / entry["price"] <= 0.015:
            alternatives = [
                c for c in structural
                if abs(c["price"] - entry["price"])
                / entry["price"] > 0.015
            ]
            major = alternatives[0] if alternatives else None

    return {
        "entry": entry,
        "secondary": secondary,
        "major": major,
        "atr": atr,
        "all_candidates": deduped[:12],
    }


def find_strong_support_levels(
    df, current_price, lookback=180,
    pivot_window=3, tolerance_pct=0.012
):
    """Backward-compatible wrapper."""
    return find_three_tier_supports(
        df, current_price
    )["all_candidates"]


def select_best_support(df, current_price):
    tiers = find_three_tier_supports(df, current_price)

    if tiers["entry"]:
        return (
            tiers["entry"]["price"],
            tiers["all_candidates"]
        )

    ema20 = float(df["EMA20"].iloc[-1])
    ema50 = float(df["EMA50"].iloc[-1])
    below = [x for x in (ema20, ema50) if x < current_price]

    if below:
        return max(below), tiers["all_candidates"]

    return (
        float(df["Low"].tail(20).min()),
        tiers["all_candidates"]
    )


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
            default_support, support_candidates = select_best_support(
                full_df, current_price
            )
            support_tiers = find_three_tier_supports(
                full_df, current_price
            )

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

            target_mean_price = info.get("targetMeanPrice") if info else None

            # MATP is anchored to the actual last earnings announcement.
            # Only post-earnings analyst targets are considered.
            post_earnings_matp, matp_analyst_count = calculate_post_earnings_matp(
                mb_data,
                _to_date(earn["past_date"])
            )

            if post_earnings_matp is not None:
                scraped_matp = post_earnings_matp
                matp_source = f"MarketBeat post-earnings ({matp_analyst_count} analysts)"
            elif target_mean_price is not None:
                scraped_matp = target_mean_price
                matp_source = "Yahoo targetMeanPrice fallback"
            else:
                scraped_matp = current_price
                matp_source = "Current price fallback"
            
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
                    entry_tier = support_tiers.get("entry")
                    secondary_tier = support_tiers.get("secondary")
                    major_tier = support_tiers.get("major")

                    def _support_label(tier):
                        if not tier:
                            return "N/A"
                        return (
                            f"${tier['price']:.2f} "
                            f"[{tier['type']}]"
                        )

                    st.markdown(
                        f"**🟢 Entry Support:** `{
_support_label(entry_tier)}`"
                    )
                    st.markdown(
                        f"**🟡 Secondary Support:** `{
_support_label(secondary_tier)}`"
                    )
                    st.markdown(
                        f"**🔵 Major Support:** `{
_support_label(major_tier)}`"
                    )
                    st.caption(
                        f"ATR-based proximity + price structure + recency + "
                        f"touches + EMA confluence; "
                        f"{len(support_candidates)} candidates evaluated."
                    )
                else:
                    st.markdown("**Auto Support:** `EMA fallback`")
                
                st.markdown(f"**Trailing P/E:** {pe_styled}", unsafe_allow_html=True)
                st.markdown(f"**Forward P/E:** {fwd_pe_styled}", unsafe_allow_html=True)
                st.markdown(f"**PEG Ratio:** {peg_styled}", unsafe_allow_html=True)
                st.markdown(f"**MATP Price:** `${scraped_matp:.2f}` <small>[{matp_source}]</small>", unsafe_allow_html=True)
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