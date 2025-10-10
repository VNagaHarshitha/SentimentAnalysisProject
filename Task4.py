# ---------------- INSTALL NOTE ----------------
# pip install matplotlib pandas prophet textblob python-dotenv google-generativeai streamlit requests plotly

import os
import time
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from textblob import TextBlob
from dotenv import load_dotenv
from datetime import datetime, timedelta
import google.generativeai as genai
import streamlit as st
import plotly.express as px

# ---------------- SETUP ----------------
load_dotenv("keys.env")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_KEY = os.getenv("GNEWS_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

if not all([NEWSAPI_KEY, GNEWS_KEY, GEMINI_KEY]):
    print(" Missing API keys. Please configure them in your .env file.")
BATCH_SIZE = 20
MAX_ARTICLES = 300
SLEEP_BETWEEN_REQUESTS = 1

# ---------------- GEMINI INIT ----------------
try:
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
    else:
        st.warning(" GEMINI_KEY not found in .env")
        model = None
except Exception as e:
    st.warning(f"Gemini initialization failed: {e}")
    model = None


# ---------------- FETCH NEWS ----------------
def fetch_news(query="  ", max_articles=MAX_ARTICLES):
    articles = []
    to_date = datetime.today()
    from_date = to_date - timedelta(days=5)
    current_date = to_date

    while current_date > from_date and len(articles) < max_articles:
        day_from = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
        day_to = current_date.strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "pageSize": 100,
            "from": day_from,
            "to": day_to,
            "apiKey": NEWSAPI_KEY
        }
        try:
            r = requests.get(url, params=params, timeout=10).json()
            batch = r.get("articles") or []
        except Exception:
            batch = []
        for a in batch:
            articles.append({
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url", ""),
                "publishedAt": a.get("publishedAt", "")
            })
        current_date -= timedelta(days=1)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return articles[:max_articles]


# ---------------- SENTIMENT ----------------
def sentiment_fallback(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    score = round(polarity * 10, 2)
    if score > 0.1:
        sentiment = "Positive"
    elif score < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, score


def analyze_sentiments(texts, batch_size=BATCH_SIZE):
    sentiments, scores = [], []
    if not model:
        for text in texts:
            s, sc = sentiment_fallback(text)
            sentiments.append(s)
            scores.append(sc)
        return sentiments, scores

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        prompt = f"""
        You are a sentiment analysis AI for stock market/business news.
        Analyze each headline and return JSON only.
        Rules:
        - Sentiment: "Positive", "Negative", "Neutral"
        - Score: float -10.0 to 10.0
        Headlines: {json.dumps(batch)}
        """
        try:
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
                request_options={"timeout": 60}
            )
            data = json.loads(response.text.strip())
            for item in data:
                sentiments.append(item.get("sentiment", "Neutral"))
                scores.append(float(item.get("score", 0.0)))
        except Exception:
            for text in batch:
                s, sc = sentiment_fallback(text)
                sentiments.append(s)
                scores.append(sc)
    return sentiments, scores


# ---------------- STREAMLIT DASHBOARD ----------------
def run_dashboard():
    st.set_page_config(layout="wide", page_title="Sentiment Forecast Dashboard")

    # -------------- VISUAL THEME --------------
    st.markdown("""
        <style>
        body {
            background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        }
        .main {
            background-color: rgba(255,255,255,0.9);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #1e3a8a;
        }
        </style>
    """, unsafe_allow_html=True)

    # -------------- SIDEBAR NAVIGATION --------------
    st.sidebar.title(" Dashboard Navigation")
    selected_page = st.sidebar.radio(
        "Go to:",
        [" Input & Data Fetch", " Forecast & Sentiment Charts", " Alert System"]
    )

    if "df" not in st.session_state:
        st.session_state["df"] = None

    # -------------- PAGE 1: INPUT ----------------
    if selected_page == " Input & Data Fetch":
        st.header("Input & Data Fetch")
        query = st.text_input("Enter Topic (e.g., 'stock market', 'AI'):", " Topic ")
        fetch_btn = st.button("Fetch Data")

        if fetch_btn:
            with st.spinner("Fetching news..."):
                articles = fetch_news(query)
                texts = [a["title"] for a in articles if a["title"]]
                sentiments, scores = analyze_sentiments(texts)
                for i in range(len(articles)):
                    articles[i]["sentiment"] = sentiments[i]
                    articles[i]["sentiment_score"] = scores[i]
                df = pd.DataFrame(articles)
                df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
                df = df.dropna(subset=["publishedAt"])
                st.session_state["df"] = df
                st.success(f" Fetched {len(df)} articles for '{query}'")
                st.dataframe(df[["title", "sentiment", "sentiment_score"]])
                st.download_button(
                    " Download CSV",
                    df.to_csv(index=False),
                    file_name=f"{query}_sentiments.csv"
                )

    # -------------- PAGE 2: FORECAST --------------
    elif selected_page == " Forecast & Sentiment Charts":
        st.header(" Forecast & Sentiment Charts")

        if st.session_state["df"] is not None:
            df = st.session_state["df"]

            # ---- Prepare data ----
            daily_df = df.resample("D", on="publishedAt")["sentiment_score"].mean().reset_index()
            daily_df["y"] = daily_df["sentiment_score"].rolling(window=3, min_periods=1).mean()
            daily_df = daily_df.rename(columns={"publishedAt": "ds"})
            daily_df["ds"] = pd.to_datetime(daily_df["ds"]).dt.tz_localize(None)

            # ---- Forecast ----
            prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True)
            prophet_model.fit(daily_df[["ds", "y"]])
            future = prophet_model.make_future_dataframe(periods=7)
            forecast = prophet_model.predict(future)

            # ---- KPI Cards ----
            pos_pct = round((df["sentiment_score"] > 0).mean() * 100, 1)
            neg_pct = round((df["sentiment_score"] < 0).mean() * 100, 1)
            avg_score = round(df["sentiment_score"].mean(), 2)
            forecast_trend = round(forecast["yhat"].iloc[-1], 2)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(" Positive Sentiment", f"{pos_pct}%", f"{pos_pct - 50:+.1f}% vs neutral")
            col2.metric(" Negative Sentiment", f"{neg_pct}%", f"{neg_pct - 50:+.1f}% vs neutral")
            col3.metric(" Avg Sentiment Score", f"{avg_score}", "7-day rolling avg")
            col4.metric(" Forecast Trend", f"{forecast_trend}", "Next 7 days")

            # ---- ORIGINAL FORECAST GRAPH ----
            st.subheader(" Sentiment Forecast (Prophet Model)")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="black")
            ax.scatter(daily_df["ds"], daily_df["y"], color="blue", s=20, label="Actual")
            ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                            alpha=0.3, color="skyblue")
            ax.legend()
            ax.set_xlabel("Date")
            ax.set_ylabel("Sentiment Score")
            st.pyplot(fig)

            # ---- SENTIMENT DISTRIBUTION ----
            st.subheader(" Sentiment Breakdown")
            col5, col6 = st.columns(2)

            pos_count = (df["sentiment_score"] > 0).sum()
            neg_count = (df["sentiment_score"] < 0).sum()
            neu_count = (df["sentiment_score"] == 0).sum()

            with col5:
                pie_chart = px.pie(
                    values=[pos_count, neu_count, neg_count],
                    names=["Positive", "Neutral", "Negative"],
                    title="Sentiment Distribution",
                    color_discrete_sequence=px.colors.sequential.Blues,
                )
                st.plotly_chart(pie_chart, use_container_width=True)

            with col6:
                bar_chart = px.bar(
                    x=["Positive", "Negative"],
                    y=[
                        df.loc[df["sentiment_score"] > 0, "sentiment_score"].mean(),
                        df.loc[df["sentiment_score"] < 0, "sentiment_score"].mean(),
                    ],
                    title="Average Sentiment Score",
                    labels={"x": "Sentiment", "y": "Average Score"},
                )
                st.plotly_chart(bar_chart, use_container_width=True)

        else:
            st.info("Please fetch data first to view charts.")

    # -------------- PAGE 3: ALERTS --------------
    elif selected_page == " Alert System":
        st.header(" Alert System")

        if st.session_state["df"] is not None:
            df = st.session_state["df"]
            strong_pos = df[df["sentiment_score"] > 5]
            strong_neg = df[df["sentiment_score"] < -5]

            if not strong_pos.empty:
                st.warning(f"{len(strong_pos)} Strong Positive Articles Detected!")
                for _, row in strong_pos.iterrows():
                    st.markdown(f"**[ {row['title']}]({row['url']})** — Score: {row['sentiment_score']:.2f}")

            if not strong_neg.empty:
                st.error(f" {len(strong_neg)} Strong Negative Articles Detected!")
                for _, row in strong_neg.iterrows():
                    st.markdown(f"**[ {row['title']}]({row['url']})** — Score: {row['sentiment_score']:.2f}")

            if strong_pos.empty and strong_neg.empty:
                st.info("Sentiment levels are within the normal range.")
        else:
            st.info("Fetch data to activate alerts.")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    run_dashboard()
