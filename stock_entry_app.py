import streamlit as st
import yfinance as yf
import pandas as pd
import re
import statistics
from datetime import datetime, timezone, timedelta
import cloudscraper
from bs4 import BeautifulSoup
import plotly.graph_objects as go

# Global system trading parameters
DEFAULT_GLOBAL_CAPITAL = 6000.00  
RISK_PERCENT = 0.01       # 1% max risk per trade
OFFSET_PCT = 0.005        # 0.5% offset for Entry Price

# --- STREAMLIT CACHING ENGINE TO BYPASS CLOUD RATE LIMITS ---
@st.cache_data(ttl=300)  # Caches results for 5 minutes (300 seconds)
def fetch_stock_data_cached(ticker_symbol):
    """
    Fetches historical data, calendar information, ticker info stats, and financials
    in a single cached block to protect the cloud IP from Yahoo rate limits.
    """
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. Fetch History
    full_df = ticker.history(period="1y", interval="1d")
    if full_df.empty or len(full_df) < 200:
        return None, "Insufficient historical market engine metrics returned."
    
    full_df.columns = [str(col).strip() for col in full_df.columns]
    full_df["EMA20"] = full_df["Close"].ewm(span=20, adjust=False).mean()
    full_df["EMA50"] = full_df["Close"].ewm(span=50, adjust=False).mean()
    full_df["EMA200"] = full_df["Close"].ewm(span=200, adjust=False).mean()
    
    # 2. Extract Key Info Stats Safely
    info_dict = {}
    try:
        info_dict = ticker.info
    except Exception:
        pass
        
    # 3. Pull Earnings Calendar Safely
    calendar_dict = {}
    try:
        calendar_dict = ticker.calendar
    except Exception:
        pass

    # 4. Pull Primary Earnings Dates Directly as fallback
    earnings_dates_df = None
    try:
        earnings_dates_df = ticker.get_earnings_dates(limit=20)
    except Exception:
        pass

    # 5. Pull Financials for Trend Stats Safely
    quarterly_income = None
    try:
        quarterly_income = ticker.quarterly_income_stmt
    except Exception:
        pass
        
    return {
        "df": full_df,
        "info": info_dict,
        "calendar": calendar_dict,
        "earnings_dates_df": earnings_dates_df,
        "quarterly_income": quarterly_income
    }, None

# --- CORE MATH & ENGINE DATA FUNCTIONS ---
def get_last_earnings_date_yf(ticker_symbol, cached_calendar, cached_df):
    try:
        # Priority Fallback 1: Use direct historical data table if pulled successfully
        if cached_df is not None and not cached_df.empty:
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            past_earnings = cached_df[cached_df.index <= now_utc]
            if not past_earnings.empty:
                return past_earnings.index.max().date()

        # Priority Fallback 2: Use pre-cached calendar data safely
        if cached_calendar and "Earnings Date" in cached_calendar:
            dates = cached_calendar["Earnings Date"]
            if isinstance(dates, list) and len(dates) > 0:
                return dates[0] if not isinstance(dates[0], datetime) else dates[0].date()
            elif isinstance(dates, datetime):
                return dates.date()
    except Exception: pass
    return None

def calculate_post_earnings_median_matp(ticker, cached_calendar, cached_df, default_target_mean, current_price):
    last_earnings_date = get_last_earnings_date_yf(ticker, cached_calendar, cached_df)
    if not last_earnings_date:
        last_earnings_date = datetime.now(timezone.utc).date() - timedelta(days=14)
        
    url = f"https://www.marketbeat.com/stocks/NYSE/{ticker}/forecast/"
    scraper = cloudscraper.create_scraper()
    try:
        response = scraper.get(url, timeout=10)
        if response.status_code != 200:
            alt_url = f"https://www.marketbeat.com/stocks/NASDAQ/{ticker}/forecast/"
            response = scraper.get(alt_url, timeout=10)
        if response.status_code != 200: 
            return float(default_target_mean or current_price)
            
        soup = BeautifulSoup(response.text, 'html.parser')
        history_table = None
        for table in soup.find_all("table"):
            first_row = table.find("tr")
            if first_row:
                header_cells = [cell.text.lower().strip() for cell in first_row.find_all(["th", "td"])]
                if any("date" in h for h in header_cells) and any("brokerage" in h for h in header_cells):
                    history_table = table
                    break
        if not history_table: return float(default_target_mean or current_price)
            
        header_cells = [cell.text.lower().strip() for cell in history_table.find("tr").find_all(["th", "td"])]
        date_idx = next((i for i, h in enumerate(header_cells) if "date" in h), 0)
        target_idx = next((i for i, h in enumerate(header_cells) if "target" in h), 3)
        
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

            if row_date and row_date >= last_earnings_date:
                final_target_segment = raw_target_str.split("➝")[-1].strip() if "➝" in raw_target_str else raw_target_str
                numeric_match = re.search(r'\d+(?:\.\d+)?', final_target_segment.replace(",", ""))
                if numeric_match:
                    post_earnings_targets.append(float(numeric_match.group(0)))
                    
        if post_earnings_targets:
            return statistics.median(post_earnings_targets)
    except Exception: pass
    return float(default_target_mean or current_price)

def get_earnings_profile(cached_calendar, cached_df, cached_financials):
    now = datetime.now(timezone.utc)
    today_date = now.date()  
    profile = {
        "past_date": "N/A", "past_elapsed": "N/A", "past_days_val": None,
        "next_date": "N/A", "next_days": "N/A", "next_days_val": None,
        "trend_str": "", "is_3q_uptrend": False
    }

    # Method 1: Process Direct Earnings Dataframe if Available
    try:
        if cached_df is not None and not cached_df.empty:
            df_copy = cached_df.copy()
            df_copy.index = df_copy.index.tz_localize(None)
            today_datetime = datetime.combine(today_date, datetime.min.time())
            
            past_earnings = df_copy[df_copy.index <= today_datetime]
            future_earnings = df_copy[df_copy.index > today_datetime]
            
            if not future_earnings.empty:
                next_event = future_earnings.index.min().date()
                profile["next_date"] = next_event.strftime("%b %d, %Y")
                profile["next_days_val"] = (next_event - today_date).days
                profile["next_days"] = "Today" if profile["next_days_val"] == 0 else f"{profile['next_days_val']}d away"
                
            if not past_earnings.empty:
                past_event = past_earnings.index.max().date()
                profile["past_date"] = past_event.strftime("%b %d, %Y")
                profile["past_days_val"] = (today_date - past_event).days
                profile["past_elapsed"] = f"{profile['past_days_val']}d ago"
    except Exception: pass

    # Method 2: Fallback onto Calendar Dictionary if Method 1 returned incomplete values
    try:
        if (profile["next_date"] == "N/A" or profile["past_date"] == "N/A") and cached_calendar and "Earnings Date" in cached_calendar:
            dates = cached_calendar["Earnings Date"]
            if isinstance(dates, list) and len(dates) > 0:
                parsed_dates = [d.date() if isinstance(d, datetime) else d for d in dates]
                parsed_dates.sort()
                
                futures = [d for d in parsed_dates if d >= today_date]
                pasts = [d for d in parsed_dates if d < today_date]
                
                if futures and profile["next_date"] == "N/A":
                    nxt = futures[0]
                    profile["next_date"] = nxt.strftime("%b %d, %Y")
                    profile["next_days_val"] = (nxt - today_date).days
                    profile["next_days"] = f"{profile['next_days_val']}d away" if profile["next_days_val"] > 0 else "Today"
                    
                if pasts and profile["past_date"] == "N/A":
                    pst = pasts[-1]
                    profile["past_date"] = pst.strftime("%b %d, %Y")
                    profile["past_days_val"] = (today_date - pst).days
                    profile["past_elapsed"] = f"{profile['past_days_val']}d ago"
    except Exception: pass

    try:
        q_income = cached_financials
        if q_income is not None and not q_income.empty and "Net Income" in q_income.index:
            net_incomes = q_income.loc["Net Income"].tolist()
            net_incomes = [float(x) for x in net_incomes if pd.notna(x)]
            pct_values = []
            for i in range(min(3, len(net_incomes) - 1)):
                prev = net_incomes[i + 1]
                pct_values.append(((net_incomes[i] - prev) / abs(prev)) * 100 if prev else 0.0)
            pct_values.reverse()
            trend_formatted = [f"{'▲' if p>0 else '▼' if p<0 else '►'} {int(p)}%" for p in pct_values]
            profile["trend_str"] = " | Trends: " + " -> ".join(trend_formatted)
            if len(pct_values) >= 3 and all(p > 0 for p in pct_values):
                profile["is_3q_uptrend"] = True
    except Exception: pass
    return profile


# --- STREAMLIT WEB APP UI INTERFACE ---
st.set_page_config(page_title="Entry Matrix Terminal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .stApp { background-color: #121212; color: #ffffff; }
        div[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: bold !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Entry Matrix Terminal")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration Engine")
    ticker_input = st.text_input("Ticker Symbol", value="", placeholder="e.g. AAPL").strip().upper()
    trading_capital = st.number_input("Trading Capital ($)", value=DEFAULT_GLOBAL_CAPITAL, step=500.0)
    st.markdown("---")

if ticker_input:
    with st.spinner(f"Analyzing {ticker_input} profiles safely from cache pool..."):
        dataset, error_msg = fetch_stock_data_cached(ticker_input)
        
        if error_msg:
            st.error(error_msg)
            st.stop()
            
        if dataset is None:
            st.warning("No structural profile was returned from the cache layer. Rate limit exceeded. Try again in a minute.")
            st.stop()
            
        try:
            full_df = dataset["df"]
            info = dataset["info"]
            calendar = dataset["calendar"]
            earnings_dates_df = dataset["earnings_dates_df"]
            quarterly_income = dataset["quarterly_income"]
            
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
            
            trailing_pe = info.get("trailingPE", "N/A") if info else "N/A"
            forward_pe = info.get("forwardPE", "N/A") if info else "N/A"
            peg_ratio = info.get("pegRatio", "N/A") if info else "N/A"
            sector_name = info.get('sector', 'N/A') if info else 'N/A'
            industry_name = info.get('industry', 'N/A') if info else 'N/A'
            target_mean_price = info.get("targetMeanPrice") if info else None
            
            def style_metric_val(val, threshold, is_peg=False):
                if val == "N/A" or not isinstance(val, (int, float)):
                    return f"`{val}`"
                formatted_val = f"{val:.2f}"
                is_good = (val <= threshold) if not is_peg else (0 < val <= threshold)
                color = "#00e676" if is_good else "#ff5252"
                return f'<span style="color:{color}; font-weight:bold;">{formatted_val}</span>'

            pe_styled = style_metric_val(trailing_pe, 30.0)
            fwd_pe_styled = style_metric_val(forward_pe, 30.0)
            peg_styled = style_metric_val(peg_ratio, 2.0, is_peg=True)
            
            detailed_sector_str = f"{sector_name} - {industry_name}" if industry_name != 'N/A' else sector_name
            
            scraped_matp = calculate_post_earnings_median_matp(ticker_input, calendar, earnings_dates_df, target_mean_price, current_price)
            earn = get_earnings_profile(calendar, earnings_dates_df, quarterly_income)
            
            def style_earnings_date(date_str, days_val, is_future=False):
                if date_str == "N/A" or days_val is None:
                    return f"`{date_str}`"
                label = earn["next_days"] if is_future else earn["past_elapsed"]
                if abs(days_val) <= 7:
                    return f'<span style="color:#ff5252; font-weight:bold;">{date_str} ({label})</span>'
                return f'`{date_str}` ({label})'

            last_earn_styled = style_earnings_date(earn["past_date"], earn["past_days_val"], is_future=False)
            next_earn_styled = style_earnings_date(earn["next_date"], earn["next_days_val"], is_future=True)

            workspace_left, workspace_right = st.columns([1, 1.2])
            
            with workspace_left:
                st.subheader("📊 Core Market Analysis Profile")
                st.metric("Current Price", f"${current_price:.2f}")
                
                st.markdown(f"**Sector Info:** `{detailed_sector_str}`")
                trend_status = "🟩 **PERFECT UPTREND (EMA STACK)**" if (ema20 > ema50 > ema200) else "🟥 **NO CLEAR TREND / CONSOLIDATION**"
                st.markdown(f"**Trend State:** {trend_status}")
                
                st.markdown(f"**Trailing P/E:** {pe_styled}", unsafe_allow_html=True)
                st.markdown(f"**Forward P/E:** {fwd_pe_styled}", unsafe_allow_html=True)
                st.markdown(f"**PEG Ratio:** {peg_styled}", unsafe_allow_html=True)
                st.markdown(f"**MATP Price:** `${scraped_matp:.2f}`")
                
                st.markdown(f"**Last Earnings:** {last_earn_styled}", unsafe_allow_html=True)
                st.markdown(f"**Next Earnings:** {next_earn_styled}", unsafe_allow_html=True)
                
                qh_text = "🟢 **3Q Continuous Growth Uptrend**" if earn['is_3q_uptrend'] else "📋 **Mixed Growth Matrix**"
                st.markdown(f"**Quarterly Income Health:** {qh_text} {earn['trend_str']}")
                
                st.markdown("---")
                st.subheader("⚙️ Interactive Formula Adjustments")
                
                default_support = ema20 if abs(current_price - ema20) < abs(current_price - ema50) else ema50
                default_resistance = scraped_matp
                
                default_entry = default_support * (1 + OFFSET_PCT)
                default_target = default_resistance * (1 - 0.002)
                default_stop = default_support - (1.5 * extracted_atr)
                
                if "prev_ticker" not in st.session_state or st.session_state.prev_ticker != ticker_input:
                    st.session_state.prev_ticker = ticker_input
                    st.session_state.val_support = float(default_support)
                    st.session_state.val_resistance = float(default_resistance)
                    st.session_state.val_entry = float(default_entry)
                    st.session_state.val_target = float(default_target)
                    st.session_state.val_stop = float(default_stop)

                def update_base_fields():
                    st.session_state.val_entry = st.session_state.val_support * (1 + OFFSET_PCT)
                    st.session_state.val_target = st.session_state.val_resistance * (1 - 0.002)
                    st.session_state.val_stop = st.session_state.val_support - (1.5 * extracted_atr)

                grid_col1, grid_col2 = st.columns(2)
                with grid_col1:
                    support_val = st.number_input("Support Level", key="val_support", step=0.5, on_change=update_base_fields)
                    entry_val = st.number_input("Entry Price", key="val_entry", step=0.5)
                    stop_val = st.number_input("Stop Loss", key="val_stop", step=0.5)
                with grid_col2:
                    resistance_val = st.number_input("Resistance Level (MATP Source)", key="val_resistance", step=0.5, on_change=update_base_fields)
                    target_val = st.number_input("Profit Target", key="val_target", step=0.5)
                
                unit_risk = abs(entry_val - stop_val)
                unit_reward = abs(target_val - entry_val)
                ror = unit_reward / unit_risk if unit_risk > 0 else 0.0
                
                max_allowed_risk_dollars = trading_capital * RISK_PERCENT
                units = int(max_allowed_risk_dollars / unit_risk) if unit_risk > 0 else 0
                potential_profit = unit_reward * units
                potential_loss = unit_risk * units
                
                st.markdown("---")
                st.subheader("🏆 Expected Formula Execution Output")
                
                st.markdown(f"• **Entry Price:** `${entry_val:.2f}`")
                st.markdown(f"• **Profit Target:** `${target_val:.2f}`")
                st.markdown(f"• **Stop Loss:** `${stop_val:.2f}`")
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
                
                # Snaps horizontal line annotations neatly onto the left side margin edge 
                fig.add_hline(y=target_val, line_color="#00e5ff", line_width=2, line_dash="solid", label=dict(text=f"Target (${target_val:.2f})", textposition="top left", font=dict(color="#00e5ff")))
                fig.add_hline(y=entry_val, line_color="#2196F3", line_width=2, line_dash="solid", label=dict(text=f"Entry (${entry_val:.2f})", textposition="top left", font=dict(color="#2196F3")))
                fig.add_hline(y=stop_val, line_color="#ff9800", line_width=2, line_dash="dash", label=dict(text=f"Stop (${stop_val:.2f})", textposition="bottom left", font=dict(color="#ff9800")))
                
                fig.update_layout(
                    title=f"{ticker_input} Technical Matrix",
                    template="plotly_dark",
                    paper_bgcolor="#121212",
                    plot_bgcolor="#1e1e1e",
                    xaxis_rangeslider_visible=False,
                    height=700,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Execution Error Parsing Parameters: {str(e)}")
else:
    st.info("💡 Enter a stock ticker symbol in the configuration sidebar to initialize the real-time visual web entry terminal.")