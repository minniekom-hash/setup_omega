
import os
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import requests
from datetime import datetime, timedelta
import pytz
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np
from scipy.stats import norm

nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
CORS(app)
sia = SentimentIntensityAnalyzer()

# ใส่ API Key จาก newsapi.org เพื่อให้ระบบวิเคราะห์ข่าวได้แม่นยำ
NEWS_API_KEY = 'YOUR_NEWS_API_KEY' 
TRACKED_STOCKS = ['SOFI', 'PLTR', 'NIO', 'MARA', 'AMD', 'F', 'PENN', 'LCID', 'TSLA', 'NVDA']

def calculate_black_scholes(S, K, T, r, sigma, option_type='Call'):
    if T <= 0: return 0, 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'Call':
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return round(delta, 2), round(gamma, 4), round(theta, 3), round(sigma, 2)

def monte_carlo_simulation(S, K, T, r, sigma, option_type='Call', iterations=10000):
    dt = T
    z = np.random.standard_normal(iterations)
    price_at_expiry = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    if option_type == 'Call':
        win_count = np.sum(price_at_expiry > K)
    else:
        win_count = np.sum(price_at_expiry < K)
    return (win_count / iterations) * 100

def get_sentiment_score(ticker):
    try:
        url = f'https://newsapi.org/v2/everything?q={ticker}+stock&sortBy=publishedAt&apiKey={NEWS_API_KEY}'
        response = requests.get(url).json()
        articles = response.get('articles', [])
        if not articles: return 0, "ไม่มีข่าวเด่น"
        scores = [sia.polarity_scores(art['title'])['compound'] for art in articles[:5]]
        return sum(scores) / len(scores), articles[0]['title']
    except: return 0, "ไม่สามารถดึงข่าวได้"

def omega_singularity_analysis(ticker, price):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="6mo")
        if len(history) < 120: return None
        closes = history['Close'].tolist()
        log_returns = np.log(np.array(closes[1:]) / np.array(closes[:-1]))
        volatility = np.std(log_returns) * np.sqrt(252)
        ema21 = sum(closes[-21:]) / 21
        vwap = sum(closes[-1:]) / 1 
        sentiment_score, headline = get_sentiment_score(ticker)
        
        final_direction = 'Call' if (price > ema21 and sentiment_score >= -0.1) or (sentiment_score > 0.3) else 'Put'
        
        today = datetime.now()
        days_until_friday = (4 - today.weekday()) % 7
        T = (days_until_friday + 1) / 365
        expected_move = price * volatility * np.sqrt(T)
        strike = round(price + (0.4 * expected_move), 2) if final_direction == 'Call' else round(price - (0.4 * expected_move), 2)
        tp = round(price + (1.2 * expected_move), 2) if final_direction == 'Call' else round(price - (1.2 * expected_move), 2)
        
        delta, gamma, theta, iv = calculate_black_scholes(price, strike, T, 0.05, volatility, final_direction)
        mc_win_rate = monte_carlo_simulation(price, strike, T, 0.05, volatility, final_direction)
        
        avg_vol = info.get('averageVolume', 1000000)
        current_vol = info.get('volume', 0)
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        
        final_score = mc_win_rate
        if vol_ratio > 2.0: final_score += 10
        if (final_direction == 'Call' and price > vwap) or (final_direction == 'Put' and price < vwap):
            final_score += 10
        if abs(sentiment_score) > 0.3: final_score += 10

        final_win_rate = min(final_score, 99.0)
        if final_win_rate >= 85: verdict = "🔥 ULTIMATE ALPHA"
        elif final_win_//_rate >= 70: verdict = "⚡ HIGH POTENTIAL"
        else: verdict = "❌ NO EDGE"

        return {
            "ticker": ticker, "price": price, "type": final_direction,
            "strike": strike, "tp": tp, "expiry": (today + timedelta(days=days_until_friday)).strftime('%Y-%m-%d'),
            "winRate": f"{round(final_win_rate, 1)}%", "reason": f"MC-Sim: {round(mc_win_rate,1)}% | Whale Flow: {round(vol_ratio,1)}x", 
            "verdict": verdict, "news": headline, "delta": delta, "gamma": gamma, "theta": theta, "iv": f"{round(iv*100, 2)}%"
        }
    except: return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    tz_us = pytz.timezone('America/New_York')
    tz_th = pytz.timezone('Asia/Bangkok')
    now_us = datetime.now(tz_us).strftime('%Y-%m-%d %H:%M:%S')
    now_th = datetime.now(tz_th).strftime('%Y-%m-%d %H:%M:%S')
    results = []
    for symbol in TRACKED_STOCKS:
        try:
            ticker_data = yf.Ticker(symbol)
            current_price = ticker_data.fast_info['last_price']
            data = omega_singularity_analysis(symbol, round(current_price, 2))
            if data: results.append(data)
        except: continue 
    return jsonify({"status": "open", "time_us": now_us, "time_th": now_th, "data": results})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
