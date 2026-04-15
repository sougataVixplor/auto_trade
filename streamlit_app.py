import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
import feedparser
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import time

load_dotenv()
# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold & Silver Predictor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🥇 Gold & Silver Price Predictor")
st.caption("AI-powered analysis combining chart patterns, technical indicators, and live news sentiment")

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    metal = st.selectbox("Metal", ["Gold (GC=F)", "Silver (SI=F)", "Both"])
    period = st.selectbox("Chart Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
    interval = st.selectbox("Candle Interval", ["1d", "1h", "4h"], index=0)
    auto_refresh = st.checkbox("Auto-refresh (every 5 min)", value=False)
    run_analysis = st.button("🔍 Run Analysis", type="primary", use_container_width=True)

    # Load keys from .env
    gemini_key = os.getenv("GEMINI_API_KEY")
    news_key = os.getenv("NEWS_API_KEY")

# ── Helper functions ───────────────────────────────────────────────────────────

TICKERS = {
    "Gold (GC=F)": ["GC=F"],
    "Silver (SI=F)": ["SI=F"],
    "Both": ["GC=F", "SI=F"],
}

METAL_NAMES = {"GC=F": "Gold", "SI=F": "Silver"}


@st.cache_data(ttl=300)
def fetch_price_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df.dropna(inplace=True)
    # Technical indicators
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.atr(length=14, append=True)
    return df


@st.cache_data(ttl=300)
def fetch_news_newsapi(query: str, api_key: str, days_back: int = 3) -> list[dict]:
    try:
        from newsapi import NewsApiClient
        napi = NewsApiClient(api_key=api_key)
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        res = napi.get_everything(q=query, from_param=from_date, language="en",
                                  sort_by="publishedAt", page_size=10)
        return res.get("articles", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def fetch_news_rss(metal_name: str) -> list[dict]:
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={metal_name.lower()}&region=US&lang=en-US",
        "https://www.kitco.com/rss/feeds/news.xml",
        "https://www.investing.com/rss/news_14.rss",  # Commodities RSS
    ]
    articles = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                articles.append({
                    "title": entry.get("title", ""),
                    "description": entry.get("summary", ""),
                    "publishedAt": entry.get("published", ""),
                    "url": entry.get("link", ""),
                    "source": {"name": feed.feed.get("title", url)},
                })
        except Exception:
            continue
    return articles[:10]


def score_sentiment(text: str) -> float:
    """Simple keyword-based sentiment score -1 to +1."""
    bullish = ["surge", "rally", "rise", "gain", "bullish", "high", "demand",
               "inflation", "safe haven", "buy", "strong", "record"]
    bearish = ["fall", "drop", "decline", "bearish", "sell", "weak", "crash",
               "low", "plunge", "risk", "pressure", "hawkish"]
    text_lower = text.lower()
    score = sum(1 for w in bullish if w in text_lower)
    score -= sum(1 for w in bearish if w in text_lower)
    return max(-1.0, min(1.0, score / 5.0))


def build_chart(df: pd.DataFrame, ticker: str, show_indicators: bool = True) -> go.Figure:
    name = METAL_NAMES.get(ticker, ticker)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=(f"{name} Price (OHLCV)", "RSI(14)", "MACD"),
    )
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=name,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)
    # Bollinger Bands
    if "BBU_20_2.0" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BBU_20_2.0"], line=dict(color="rgba(100,100,200,0.4)", width=1), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BBL_20_2.0"], line=dict(color="rgba(100,100,200,0.4)", width=1), fill="tonexty", fillcolor="rgba(100,100,200,0.05)", name="BB Lower"), row=1, col=1)
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], line=dict(color="orange", width=1.2, dash="dot"), name="SMA20"), row=1, col=1)
    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], line=dict(color="cyan", width=1.2, dash="dot"), name="SMA50"), row=1, col=1)
    # RSI
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], line=dict(color="violet", width=1.5), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    # MACD
    if "MACD_12_26_9" in df.columns:
        macd_col = "MACD_12_26_9"
        sig_col = "MACDs_12_26_9"
        hist_col = "MACDh_12_26_9"
        colors = ["green" if v >= 0 else "red" for v in df[hist_col]]
        fig.add_trace(go.Bar(x=df.index, y=df[hist_col], marker_color=colors, name="MACD Hist"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[macd_col], line=dict(color="blue", width=1.2), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[sig_col], line=dict(color="orange", width=1.2), name="Signal"), row=3, col=1)
    fig.update_layout(
        template="plotly_dark",
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def build_ai_prompt(ticker: str, df: pd.DataFrame, articles: list[dict]) -> str:
    name = METAL_NAMES.get(ticker, ticker)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    change_pct = (last["Close"] - prev["Close"]) / prev["Close"] * 100

    # Recent price data
    price_summary = f"""
METAL: {name} ({ticker})
Current Price: ${last['Close']:.2f}
Change (1 candle): {change_pct:+.2f}%
High (period): ${df['High'].max():.2f}
Low (period): ${df['Low'].min():.2f}
"""

    # Technical indicators
    rsi = last.get("RSI_14", "N/A")
    macd = last.get("MACD_12_26_9", "N/A")
    macd_sig = last.get("MACDs_12_26_9", "N/A")
    macd_hist = last.get("MACDh_12_26_9", "N/A")
    bb_upper = last.get("BBU_20_2.0", "N/A")
    bb_lower = last.get("BBL_20_2.0", "N/A")
    sma20 = last.get("SMA_20", "N/A")
    sma50 = last.get("SMA_50", "N/A")
    atr = last.get("ATRr_14", "N/A")

    indicator_summary = f"""
TECHNICAL INDICATORS:
RSI(14): {f"{rsi:.2f}" if isinstance(rsi, float) else rsi}
MACD: {f"{macd:.4f}" if isinstance(macd, float) else macd}, Signal: {f"{macd_sig:.4f}" if isinstance(macd_sig, float) else macd_sig}, Hist: {f"{macd_hist:.4f}" if isinstance(macd_hist, float) else macd_hist}
Bollinger Bands: Upper={f"{bb_upper:.2f}" if isinstance(bb_upper, float) else bb_upper}, Lower={f"{bb_lower:.2f}" if isinstance(bb_lower, float) else bb_lower}
SMA20: {f"{sma20:.2f}" if isinstance(sma20, float) else sma20}, SMA50: {f"{sma50:.2f}" if isinstance(sma50, float) else sma50}
ATR(14): {f"{atr:.2f}" if isinstance(atr, float) else atr}
Price vs SMA20: {"ABOVE" if isinstance(sma20, float) and last["Close"] > sma20 else "BELOW"}
Price vs SMA50: {"ABOVE" if isinstance(sma50, float) and last["Close"] > sma50 else "BELOW"}
"""

    # News summary
    news_lines = []
    for a in articles[:8]:
        title = a.get("title", "")
        pub = a.get("publishedAt", "")[:10]
        sent = score_sentiment(title + " " + a.get("description", ""))
        news_lines.append(f"[{pub}] {title} (sentiment: {sent:+.1f})")
    news_summary = "RECENT NEWS:\n" + "\n".join(news_lines) if news_lines else "RECENT NEWS: None available"

    return f"""You are an expert commodities analyst and technical trader. Analyze the following data for {name} and provide a concise but thorough price prediction report.

{price_summary}
{indicator_summary}
{news_summary}

Please provide your analysis in this exact format:

## 📊 Technical Analysis
[Analyze RSI, MACD, Bollinger Bands, SMA crossovers. Identify key support/resistance levels from the data.]

## 📰 News Sentiment
[Summarize the news tone. Is the macro environment bullish or bearish for {name}?]

## 🎯 Price Prediction
**Short-term (1–3 days):** [BULLISH / BEARISH / NEUTRAL] — [brief reason]
**Medium-term (1–2 weeks):** [BULLISH / BEARISH / NEUTRAL] — [brief reason]
**Confidence level:** [LOW / MEDIUM / HIGH]
**Key price levels to watch:** Support at $X, Resistance at $Y

## ⚠️ Risk Factors
[2–3 key risks that could invalidate this prediction]

## 📝 Summary
[2–3 sentence summary for a trader to act on]
"""


def run_ai_analysis(prompt: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
      model="gemini-3-flash-preview",
      contents=prompt
      )
    return response.text


# ── Main app logic ─────────────────────────────────────────────────────────────

if run_analysis or (auto_refresh and "last_run" in st.session_state):
    if not gemini_key:
        st.error("⚠️ Gemini API Key not found in .env file. Please add it to run analysis.")
        st.stop()

    tickers = TICKERS[metal]
    tabs = st.tabs([f"{'🥇' if t == 'GC=F' else '🥈'} {METAL_NAMES[t]}" for t in tickers])

    for i, ticker in enumerate(tickers):
        with tabs[i]:
            name = METAL_NAMES[ticker]
            col1, col2 = st.columns([2, 1])

            with st.spinner(f"Fetching {name} data..."):
                df = fetch_price_data(ticker, period, interval)

            if df.empty:
                st.error(f"Could not fetch data for {ticker}. Check your internet connection.")
                continue

            # Key metrics
            last_price = df["Close"].iloc[-1]
            prev_price = df["Close"].iloc[-2]
            chg = last_price - prev_price
            chg_pct = chg / prev_price * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{name} Price", f"${last_price:.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
            m2.metric("Period High", f"${df['High'].max():.2f}")
            m3.metric("Period Low", f"${df['Low'].min():.2f}")
            rsi_val = df["RSI_14"].iloc[-1] if "RSI_14" in df.columns else None
            if rsi_val:
                rsi_label = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
                m4.metric("RSI(14)", f"{rsi_val:.1f}", rsi_label)

            # Chart
            st.plotly_chart(build_chart(df, ticker), use_container_width=True)

            # News
            with st.spinner("Fetching news..."):
                if news_key:
                    articles = fetch_news_newsapi(f"{name} price", news_key)
                else:
                    articles = fetch_news_rss(name)

            col_news, col_ai = st.columns([1, 1])

            with col_news:
                st.subheader("📰 Latest News")
                if articles:
                    for a in articles[:6]:
                        title = a.get("title", "No title")
                        url = a.get("url", "#")
                        pub = a.get("publishedAt", "")[:10]
                        src = a.get("source", {}).get("name", "")
                        sent = score_sentiment(title + " " + a.get("description", ""))
                        sent_emoji = "🟢" if sent > 0.1 else ("🔴" if sent < -0.1 else "⚪")
                        st.markdown(f"{sent_emoji} **[{title}]({url})**  \n`{src}` · {pub}")
                        st.divider()
                else:
                    st.info("No news found. Add a NewsAPI key for better coverage.")

            with col_ai:
                st.subheader("🤖 AI Analysis")
                with st.spinner("Running Gemini AI analysis..."):
                    prompt = build_ai_prompt(ticker, df, articles)
                    try:
                        analysis = run_ai_analysis(prompt, gemini_key)
                        st.markdown(analysis)
                    except Exception as e:
                        if "403" in str(e) or "API_KEY_INVALID" in str(e):
                            st.error("Invalid Gemini API key.")
                        elif "429" in str(e):
                            st.error("Rate limit hit. Please wait and try again.")
                        else:
                            st.error(f"AI analysis failed: {e}")

    st.session_state["last_run"] = time.time()
    if auto_refresh:
        time.sleep(300)
        st.rerun()

else:
    st.info("👈 Configure settings in the sidebar and click **Run Analysis** to start.")
    st.markdown("""
    ### What this app does
    - Fetches real-time Gold & Silver prices via Yahoo Finance
    - Computes technical indicators: RSI, MACD, Bollinger Bands, SMA
    - Pulls recent news headlines and scores sentiment
    - Sends all data to Gemini AI for a combined prediction report
    - Displays interactive candlestick charts with overlays

    ### Setup
    1. Ensure your **.env** file contains `GEMINI_API_KEY` and `NEWS_API_KEY`.
    2. Restart the app or click **Run Analysis** to start.
    """)