import os
import streamlit as st
import pandas as pd
import torch
import datetime
import numpy as np
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import yfinance as yf
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from newspaper import Article
import feedparser
from time import sleep

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# --- LSTM Model Class (Matches your trained model) ---
class LSTM(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=3, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- Configuration ---
MODEL_PATH = "project_weights_NSEEMANT_RAMARAOV_SADHANAS.pt"
DEVICE = torch.device("cpu")
STOCK_OPTIONS = {
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "RELIANCE": "RELIANCE.NS",
    "HDFC": "HDFCBANK.NS",
    "WIPRO": "WIPRO.NS"
}

tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model_sent = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
summarizer = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)

# --- Helper Functions ---
def load_stock_sequence(stock_ticker):
    df = yf.download(stock_ticker, period="120d", interval="1d")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    if len(df) < 30:
        st.error(f"❌ Only {len(df)} valid rows found for {stock_ticker}. Need at least 30.")
        st.stop()
    last_30 = df[-30:]
    data = last_30.values
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)

def predict_future(model, initial_sequence, days=7):
    model.eval()
    preds = []
    seq = initial_sequence.clone()
    for _ in range(days):
        with torch.no_grad():
            out = model(seq)
            preds.append(out.item())
            new_step = torch.tensor([[[out.item()] * 5]], dtype=torch.float32)
            seq = torch.cat((seq[:, 1:, :], new_step), dim=1)
    return preds

def get_sentiment_summary(stock_ticker):
    try:
        url = f"https://finance.yahoo.com/quote/{stock_ticker}?p={stock_ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}
        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")
        headlines = soup.find_all("h3", limit=5)
        sentiments = []
        for h in headlines:
            text = h.get_text()
            polarity = TextBlob(text).sentiment.polarity
            sentiments.append(polarity)
        if not sentiments:
            return "⚠️ No recent news found."
        avg = sum(sentiments) / len(sentiments)
        if avg > 0.1:
            return f"🟢 Positive sentiment with average polarity: {avg:.2f}"
        elif avg < -0.1:
            return f"🔴 Negative sentiment with average polarity: {avg:.2f}"
        else:
            return f"🟡 Neutral sentiment with average polarity: {avg:.2f}"
    except Exception as e:
        return f"⚠️ Error fetching sentiment: {str(e)}"

def advanced_sentiment_analysis(stock_ticker):
    feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_ticker}&region=US&lang=en-US"
    feed = feedparser.parse(feed_url)
    links = [(entry.title, entry.link) for entry in feed.entries][:10]
    summaries = []
    for title, link in links:
        try:
            article = Article(link)
            article.download()
            article.parse()
            if len(article.text.strip()) < 300:
                continue
            prompt = f"Summarize this article for stock movement insights about {stock_ticker}:\n\n{article.text[:4000]}"
            summary = summarizer(prompt, max_length=200, min_length=50, do_sample=False)[0]['generated_text']
            summaries.append(summary)
        except:
            continue
    if not summaries:
        return "No relevant summaries found."
    combined = "\n\n".join(summaries)[:4000]
    final_prompt = f"Give a human-like financial summary and recommendation for {stock_ticker}:\n\n{combined}"
    result = summarizer(final_prompt, max_length=300, min_length=60, do_sample=False)[0]['generated_text']
    return result

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Prediction App", layout="centered")
st.title("📈 Stock Price Forecast with Sentiment Analysis")

with st.form("prediction_form"):
    stock_label = st.selectbox("Select Stock", list(STOCK_OPTIONS.keys()))
    start_date = st.date_input("Prediction Start Date", datetime.date.today())
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    stock_ticker = STOCK_OPTIONS[stock_label]
    st.success(f"Predicting for {stock_label} ({stock_ticker}) from {start_date}")

    model = LSTM()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    input_seq = load_stock_sequence(stock_ticker)
    predictions = predict_future(model, input_seq, days=7)
    future_dates = pd.date_range(start=start_date, periods=7)
    df_pred = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions})

    st.subheader("📊 7-Day Stock Forecast")
    st.dataframe(df_pred)

    max_row = df_pred.loc[df_pred["Predicted Price"].idxmax()]
    min_row = df_pred.loc[df_pred["Predicted Price"].idxmin()]
    avg_price = df_pred["Predicted Price"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Highest", f"{max_row['Predicted Price']:.2f}", f"on {max_row['Date'].date()}")
    col2.metric("📉 Lowest", f"{min_row['Predicted Price']:.2f}", f"on {min_row['Date'].date()}")
    col3.metric("📋 Average", f"{avg_price:.2f}")

    st.line_chart(df_pred.set_index("Date"))

    st.markdown("### 🧠 Market Sentiment from Yahoo Finance")
    sentiment_summary = get_sentiment_summary(stock_ticker)
    st.info(sentiment_summary)

    if st.button("🔎 Deep Sentiment Analysis"):
        with st.spinner("Fetching advanced financial analysis..."):
            result = advanced_sentiment_analysis(stock_ticker)
            st.subheader("🧠 Human-Like Investment Summary")
            st.success(result)

if st.button("🔄 Reset All"):
    st.experimental_rerun()
