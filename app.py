# ======================================
# Standard Library Imports
# ======================================
from decimal import Decimal, ROUND_HALF_UP # For rounding up to 2 decimal places
from concurrent.futures import ThreadPoolExecutor, as_completed # For concurrent requests
import json # For JSON data manipulation
import time # For time-related functions

# ======================================
# Third-Party Library Imports
# ======================================
import tweepy # For Twitter API
import requests # For making HTTP requests
from requests.exceptions import RequestException # For handling request exceptions
from textblob import TextBlob # For sentiment analysis
import pandas as pd # For data manipulation
import numpy as np # For numerical operations
from sklearn.ensemble import RandomForestRegressor # For Random Forest model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # For model evaluation
from ta.volatility import BollingerBands # For Bollinger Bands
from ta.momentum import RSIIndicator, StochasticOscillator # For RSI and Stochastic Oscillator
from ta.trend import MACD # For MACD
from ta.volume import OnBalanceVolumeIndicator # For On-Balance Volume
import plotly # For interactive plots
import plotly.graph_objs as go # For plotly graph objects
from markupsafe import Markup # For rendering HTML content
import yfinance as yf # For fetching stock data
from statsmodels.tsa.statespace.sarimax import SARIMAX # For SARIMA model
from dotenv import load_dotenv  # For loading environment variables
import os # For accessing environment variables

# ======================================
# Flask and Flask Extensions
# ======================================
from flask import Flask, render_template, request, make_response, session, jsonify, abort # For Flask app and rendering templates and handling requests
from flask_caching import Cache # For caching data in Flask app 

load_dotenv()

app = Flask(__name__)

# For Prices
COINCAP_ASSETS_URL = "https://api.coincap.io/v2/assets?limit=50"
COINCAP_EXCHANGES_URL = "https://api.coincap.io/v2/exchanges"
COINCAP_MARKETS_URL = "https://api.coincap.io/v2/markets"
EXCHANGE_RATES_API_URL = "https://open.er-api.com/v6/latest/USD"
BINANCE_CANDLES_URL = "https://api.binance.com/api/v3/klines"
COINGECKO_COINS_URL = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&limit=2000&sparkline=false"

# For News
# NEWS_ORG_API_KEY = os.getenv('NEWS_ORG_API_KEY')
# CRYPTOPANIC_API_TOKEN = os.getenv('CRYPTOPANIC_API_TOKEN')

# Initialize Cache
cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(app)

# Cache dictionary and cache expiration time (600 seconds = 10 minutes)
CACHE = {}
CACHE_TIMEOUT = 600  # 10 minutes

#===========================
# MAIN ROUTE
#===========================
@app.route('/')
def index():
    return render_template('index.html')


#===========================
# CRYPTO PRICE DATA
#===========================
# Constants
REQUEST_TIMEOUT = 10  # seconds

def fetch_data_concurrently(api_urls):
    def fetch_data(url):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # This will raise an HTTPError if the response returned a non-200 status code
            return response.json()
        except RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=len(api_urls)) as executor:
        futures = {executor.submit(fetch_data, url): url for url in api_urls}
    
    results = {}
    for future in as_completed(futures):
        url = futures[future]
        # Store the result in the cache
        cache.set(url, future.result(), timeout=CACHE_TIMEOUT)
        results[url] = future.result()
    return results

def convert_coin_data(coins, conversion_rate):
    for coin in coins:
        coin['priceUsd'] = round(float(coin.get('current_price') or 0) * conversion_rate, 2)
        coin['changePercent24Hr'] = round((float(coin.get('price_change_24h', 0)) / (float(coin.get('current_price', 0)) - float(coin.get('price_change_24h', 0)))) * 100, 1)
        coin['volumeUsd24Hr'] = round(float(coin.get('total_volume') or 0) * conversion_rate, 2)
        coin['marketCapUsd'] = round(float(coin.get('market_cap') or 0) * conversion_rate, 2)
    return coins

def fetch_candles_data(symbol='BTCUSDT', interval='1h', limit=500):
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(BINANCE_CANDLES_URL, params=params)
    if response.status_code == 200:
        data = []
        for item in response.json():
            data.append({
                'openTime': item[0],
                'open': item[1],
                'high': item[2],
                'low': item[3],
                'close': item[4]
            })
        return data
    else:
        print(f"Error fetching data from Binance: {response.json()}")
        return []

def create_candlestick_chart(data, current_symbol):
    trace = go.Candlestick(
        x=[item['openTime'] for item in data],
        open=[item['open'] for item in data],
        high=[item['high'] for item in data],
        low=[item['low'] for item in data],
        close=[item['close'] for item in data],
        name="Candles",
        increasing_line_color='green',
        decreasing_line_color='red'
    )
    
    layout = go.Layout(
        title='Candlestick Chart',
        xaxis=dict(title='Time', automargin=True),
        yaxis=dict(title='Price', tickprefix=current_symbol, automargin=True),
        hovermode='x',
        autosize=True
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.route('/price')
def price():
    # List of API endpoints to fetch data from
    api_urls = [
        EXCHANGE_RATES_API_URL,
        COINGECKO_COINS_URL,
        COINCAP_EXCHANGES_URL,
        COINCAP_MARKETS_URL,
        BINANCE_CANDLES_URL
    ]
    
    # Try to get the cached data
    cached_results = {url: cache.get(url) for url in api_urls}
    if all(cached_results.values()):
        results = cached_results
    else:
        results = fetch_data_concurrently(api_urls)

    # Extract data based on the API endpoint
    raw_rates = results[EXCHANGE_RATES_API_URL].get('rates', {})
    coins_data = results[COINGECKO_COINS_URL]
    exchanges_data = results[COINCAP_EXCHANGES_URL].get('data', [])
    markets_data = results[COINCAP_MARKETS_URL].get('data', [])

    rates = {
        'USD': {'rate': raw_rates['USD'], 'symbol': '$', 'name': 'United States Dollar'},
        'EUR': {'rate': raw_rates['EUR'], 'symbol': '€', 'name': 'Euro'},
        'GBP': {'rate': raw_rates['GBP'], 'symbol': '£', 'name': 'British Pound Sterling'},
        # Add other currencies as needed
    }

    # Initial conversions (based on USD)
    currency = request.args.get('currency', 'USD')
    current_symbol = rates[currency]['symbol']
    conversion_rate = rates[currency]['rate']

    # Convert coin data based on the selected currency
    for coin in coins_data:
        coin['priceUsd'] = round(float(coin.get('current_price') or 0) * conversion_rate, 2)
        coin['changePercent24Hr'] = round((float(coin.get('price_change_24h', 0)) / (float(coin.get('current_price', 0)) - float(coin.get('price_change_24h', 0)))) * 100, 1)
        coin['volumeUsd24Hr'] = round(float(coin.get('total_volume') or 0) * conversion_rate, 2)
        coin['marketCapUsd'] = round(float(coin.get('market_cap') or 0) * conversion_rate, 2)

    # Convert exchange data based on the selected currency
    for exchange in exchanges_data:
        exchange['volumeUsd'] = round(float(exchange.get('volumeUsd') or 0) * conversion_rate, 2)
        exchange['percentTotalVolume'] = '{:.2f}'.format(float(exchange.get('percentTotalVolume') or 0))
        exchange['status'] = "Online" if exchange.get('socket', False) else "Offline"

    # Convert market data based on the selected currency
    for market in markets_data:
        market['priceUsd'] = round(float(market.get('priceUsd') or 0) * conversion_rate, 2)
        market['volumeUsd24Hr'] = round(float(market.get('volumeUsd24Hr') or 0) * conversion_rate, 2)
        market['percentExchangeVolume'] = '{:.2f}'.format(float(market.get('percentExchangeVolume') or 0))

    # Fetching and creating the candlestick chart
    candles_data = fetch_candles_data()  # This remains separate as it might need params
    candlestick_chart = create_candlestick_chart(candles_data, current_symbol)

    # Check if the request is AJAX (XHR)
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'coins': coins_data,
            'exchanges': exchanges_data,
            'markets': markets_data,
            'current_symbol': current_symbol
        })

    return render_template('price.html', coins=coins_data, exchanges=exchanges_data, markets=markets_data, candlestick_chart=candlestick_chart, rates=rates, current_currency=currency, current_symbol=current_symbol)


# ===========================
# NEWS
# ===========================
@app.route('/news')
@cache.cached(timeout=7200)  # Cache the output for 2 hours (7200 seconds)
def news():
    """
    Fetches the latest cryptocurrency news from the CryptoCompare API.
    The news articles are then displayed on the 'news.html' template.

    Returns:
        Flask template: 'news.html' with news article data.
    """
    try:
        # Make a GET request to the CryptoCompare API to get the latest cryptocurrency news
        response = requests.get('https://min-api.cryptocompare.com/data/v2/news/', params={
            'lang': 'EN',
        })

        # Check if the request was successful
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as e:
        # Log the exception for debugging
        app.logger.error(f"An error occurred while fetching news: {e}")
        return render_template('error.html', error_message="Could not fetch news data."), 500

    # Initialize lists to store article details
    titles = []
    descriptions = []
    thumbnails = []
    links = []

    # Define the maximum description length
    max_description_length = 900

    # Extract the titles, descriptions, and thumbnails of the first 20 articles
    for article in data['Data'][:20]:
        title = article['title']
        description = article['body']

        # Check if the description exceeds the maximum length
        if len(description) > max_description_length:
            # Truncate the description to the maximum length and add "Continue to Article"
            description = description[:max_description_length] + '... Continue to Article'

        thumbnail = article['imageurl']
        link = article['url']

        titles.append(title)
        descriptions.append(description)
        thumbnails.append(thumbnail)
        links.append(link)

    return render_template('news.html', titles=titles, descriptions=descriptions, thumbnails=thumbnails, links=links)



#===========================
# SENTIMENTS
#===========================
# Cache the analysis function
NEWS_ORG_API_KEY = "2077a0a22bbe494eac7b129497c4cc86"  # KAN_API_KEY
CRYPTOPANIC_API_TOKEN = "8a66ae87eef26809144838e675db665202b2685c"  # KAN_API_KEY

@cache.memoize(3600)
def fetch_and_analyze_detailed(url):
    response = requests.get(url)
    data = response.json()
    positive, negative, neutral = 0, 0, 0
    positive_articles, negative_articles, neutral_articles = [], [], []
    
    if 'articles' in data:
        articles = data['articles']
    elif 'results' in data:
        articles = data['results']
    else:
        return None, None, None, None, None, None
    
    for article in articles:
        text = article.get('title', '') + ' ' + article.get('description', '')
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            positive += 1
            positive_articles.append(article)
        elif analysis.sentiment.polarity < 0:
            negative += 1
            negative_articles.append(article)
        else:
            neutral += 1
            neutral_articles.append(article)
    
    # Limit to 5 articles for each sentiment
    positive_articles = positive_articles
    negative_articles = negative_articles
    neutral_articles = neutral_articles
    
    return positive, negative, neutral, positive_articles, negative_articles, neutral_articles

def generate_plotly_bar_chart(positive, negative, neutral):
    data = [
        go.Bar(name='Positive', x=['Positive'], y=[positive], marker_color='green'),
        go.Bar(name='Negative', x=['Negative'], y=[negative], marker_color='red'),
        go.Bar(name='Neutral', x=['Neutral'], y=[neutral], marker_color='blue')
    ]

    layout = go.Layout(
        barmode='group',
        title='Sentiment Analysis',
        paper_bgcolor='rgba(245, 246, 249, 1)',  # Light blue background
        plot_bgcolor='rgba(245, 246, 249, 1)',   # Light blue background
        font=dict(color='black'),                # Font color
        xaxis=dict(title='Sentiment Categories'),# X-axis title
        yaxis=dict(title='Count', range=[0, 100]), # Y-axis title and range set to 0-100
        bargap=0.2                               # Gap between bars
    )

    # Convert to dictionaries
    data_dict = [bar.to_plotly_json() for bar in data]
    layout_dict = layout.to_plotly_json()

    return data_dict, layout_dict

# Cache the sentiment analysis route
@app.route('/sentiment', defaults={'coin': 'bitcoin'})
@app.route('/<coin>')
@cache.cached(timeout=3600, query_string=True)  # Cache with respect to query parameters
def sentiment(coin):
    try:
        # News API
        news_url = f"https://newsapi.org/v2/everything?q={coin}&apiKey={NEWS_ORG_API_KEY}"
        news_positive, news_negative, news_neutral, pos_articles, neg_articles, neu_articles = fetch_and_analyze_detailed(news_url)
        
        # CryptoPanic API
        cryptopanic_url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_TOKEN}&currencies={coin}&public=true"
        crypto_positive, crypto_negative, crypto_neutral, _, _, _ = fetch_and_analyze_detailed(cryptopanic_url)
        
        # Check for None values
        if any(val is None for val in [news_positive, news_negative, news_neutral, crypto_positive, crypto_negative, crypto_neutral]):
            raise ValueError("Failed to fetch or analyze some data")
        
        # Aggregation
        total_positive = news_positive + crypto_positive
        total_negative = news_negative + crypto_negative
        total_neutral = news_neutral + crypto_neutral
        
        # Generate Plotly Bar Chart
        data, layout = generate_plotly_bar_chart(total_positive, total_negative, total_neutral)
        
        # Check if it's an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                            'plotly_chart_data': data,
                            'plotly_chart_layout': layout,
                            'positive_articles': pos_articles,
                            'negative_articles': neg_articles,
                            'neutral_articles': neu_articles,
                        })
                
        return render_template("sentiment.html", 
                        plotly_chart_data=data, 
                        plotly_chart_layout=layout, 
                        positive_articles=pos_articles,
                        negative_articles=neg_articles,
                        neutral_articles=neu_articles,
                        selected_coin=coin)
        
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return make_response(jsonify(error=str(e)), 500)
    
    

#===========================
# CONVERTER
#===========================
# Dictionary of well-known cryptocurrencies and their symbols
# Updated crypto_symbols dictionary with 20 more cryptocurrencies
crypto_symbols = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'LTC': 'Litecoin',
    'XRP': 'Ripple',
    'ADA': 'Cardano',
    'DOGE': 'Dogecoin',
    'XLM': 'Stellar',
    'EOS': 'EOS',
    'NEO': 'NEO',
    'XMR': 'Monero',
    'DASH': 'Dash',
    'ZEC': 'Zcash',
    'XTZ': 'Tezos',
    'ATOM': 'Cosmos',
    'LINK': 'Chainlink',
    'BNB': 'Binance Coin',
    'TRX': 'Tron',
    'XVG': 'Verge',
    'IOTA': 'IOTA',
    'VET': 'VeChain',
    # Add more cryptocurrencies here
}

world_currencies = [
    'United States of America - USD - $',
    'Argentina - ARS - $',
    'Australia - AUD - $',
    'Bangladesh - BDT - ৳',
    'Brazil - BRL - R$',
    'Canada - CAD - $',
    'Chile - CLP - $',
    'China - CNY - ¥',
    'Colombia - COP - $',
    'Czech Republic - CZK - Kč',
    'Denmark - DKK - kr',
    'Egypt - EGP - £',
    'Europe - EUR - €',
    'Ghana - GHS - ₵',
    'Hong Kong - HKD - $',
    'Hungary - HUF - Ft',
    'India - INR - ₹',
    'Indonesia - IDR - Rp',
    'Israel - ILS - ₪',
    'Japan - JPY - ¥',
    'Kenya - KES - KSh',
    'Kuwait - KWD - د.ك',
    'Malaysia - MYR - RM',
    'Mexico - MXN - $',
    'Morocco - MAD - د.م.',
    'New Zealand - NZD - $',
    'Nigeria - NGN - ₦',
    'Norway - NOK - kr',
    'Oman - OMR - ﷼',
    'Peru - PEN - S/',
    'Philippines - PHP - ₱',
    'Poland - PLN - zł',
    'Qatar - QAR - ر.ق',
    'Romania - RON - lei',
    'Russia - RUB - ₽',
    'Saudi Arabia - SAR - ر.س',
    'Singapore - SGD - $',
    'South Africa - ZAR - R',
    'South Korea - KRW - ₩',
    'Sri Lanka - LKR - Rs',
    'Sweden - SEK - kr',
    'Switzerland - CHF - Fr',
    'Taiwan - TWD - NT$',
    'Thailand - THB - ฿',
    'Turkey - TRY - ₺',
    'Tunisia - TND - د.ت',
    'Uganda - UGX - USh',
    'United Arab Emirates - AED - د.إ',
    'United Kingdom - GBP - £',
    'Ukraine - UAH - ₴',
    'Vietnam - VND - ₫',
]

@app.route('/convert', methods=['GET', 'POST'])
def convert():
    return render_template('convert.html', crypto_symbols=crypto_symbols, world_currencies=world_currencies)

@app.route('/get_conversion', methods=['GET'])
def get_conversion():
    try:
        amount = Decimal(request.args.get('amount'))
        from_currency = request.args.get('from_currency')
        to_currency = request.args.get('to_currency')
        
        # Handle empty precision
        precision_str = request.args.get('precision')
        precision = int(precision_str) if precision_str else 2  # Default to 2 if precision is not provided
        
        direction = request.args.get('direction')

        # Use caching to fetch conversion data
        cache_key = f"{from_currency}_{to_currency}_{amount}_{direction}_{precision}"
        conversion_data = cache.get(cache_key)

        if conversion_data is None:
            conversion_data = fetch_conversion_data(amount, from_currency, to_currency, direction, precision)
            cache.set(cache_key, conversion_data, timeout=3600)  # Cache the result for 1 hour

        result = conversion_data['result']

        response_data = {
            'result': result
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"Error: {str(e)}")
        return abort(400, str(e))

def fetch_conversion_data(amount, from_currency, to_currency, direction, precision):
    try:
        rate = get_rate_from_api(from_currency, to_currency)

        if direction == 'fiat_to_crypto':
            raw_result = amount / rate
        else:
            raw_result = amount * rate

        # Format the result
        result = format(raw_result, f'.{precision}f')
        return {'result': result}
    except Exception as e:
        raise Exception(f"Error fetching conversion data: {str(e)}")

def get_rate_from_api(from_currency, to_currency):
    try:
        return get_rate_from_cryptocompare(from_currency, to_currency)
    except Exception as e:
        raise Exception(f"Error getting rate from API: {str(e)}")

def get_rate_from_cryptocompare(from_currency, to_currency):
    try:
        url = f'https://min-api.cryptocompare.com/data/price?fsym={from_currency}&tsyms={to_currency}'
        response = requests.get(url)
        data = response.json()

        if to_currency in data:
            return Decimal(data[to_currency])
        else:
            raise Exception("Conversion rate not found")
    except Exception as e:
        raise Exception(f"Error fetching CryptoCompare API: {str(e)}")


#===========================
# PRICE PREDICTION
#===========================
def enhanced_generate_plot(df, predictions, selected_coin, model_type):
    # Historical Open Prices
    trace_actual = go.Scatter(
        x=df.index,
        y=df['open'],
        mode='lines',
        name='Open Prices',
        hoverinfo='x+y',
        hovertemplate='Date: %{x}<br>Open Price: $%{y:.2f}<extra></extra>',
        visible=True  # This makes the trace visible by default
    )

    # Predicted Prices as a Green Line
    predicted_dates = [date for date, _ in predictions]
    predicted_prices = [price for _, price in predictions]
    trace_predicted = go.Scatter(
        x=predicted_dates, 
        y=predicted_prices, 
        mode='lines+markers',
        name='Predicted Prices', 
        line=dict(color='green'),
        marker=dict(color='green', size=6),
        hoverinfo='x+y', 
        hovertemplate='Predicted Date: %{x}<br>Predicted Price: $%{y:.2f}<extra></extra>',
        visible=True  # This makes the trace visible by default
    )
    
    # Mock Confidence Intervals
    upper_bound = [round(price + (0.05 * price), 2) for price in predicted_prices]  # 5% above prediction
    lower_bound = [round(price - (0.05 * price), 2) for price in predicted_prices]  # 5% below prediction

    trace_confidence_upper = go.Scatter(
        x=predicted_dates,
        y=upper_bound,
        mode='lines',
        fill=None,
        name='Upper Confidence',
        line=dict(color='rgba(0,100,80,0.2)'),
        visible=True
    )
    trace_confidence_lower = go.Scatter(
        x=predicted_dates,
        y=lower_bound,
        mode='lines',
        fill='tonexty',
        name='Lower Confidence',
        line=dict(color='rgba(0,100,80,0.2)'),
        visible=True
    )

    # The vertical line shape
    shapes = [{
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': df.index[-1],
        'y0': 0,
        'x1': df.index[-1],
        'y1': 1,
        'line': {
            'color': 'gray',
            'width': 2,
            'dash': 'dashdot'
        }
    }]
    
    coin_name = {
        'BTC': 'Bitcoin',
        'ETH': 'Ethereum'
        # Add more coin mappings if necessary
    }.get(selected_coin, selected_coin)
    
    model_name = {
        'RF': 'Random Forest',
        'SARIMA': 'SARIMA'
        # Add more model mappings if necessary
    }.get(model_type, model_type)
    
    # Construct the title
    title = f'Historical and Predicted Prices of {coin_name} with Confidence Intervals using {model_name} Model'

    layout = go.Layout(
    title=title,
    xaxis=dict(
        title='Date',
        rangeslider=dict(
            visible=True,
            autorange=True
        )
    ),
        yaxis=dict(title='Price in USD', tickformat='$,'),
        showlegend=True,
        autosize=True,
        hovermode='x',
        shapes=shapes,
    )

    fig = go.Figure(data=[trace_actual, trace_predicted, trace_confidence_upper, trace_confidence_lower], layout=layout)
    return fig

def fetch_data_from_cryptocompare(selected_coin):
    days = 1095 # 3 years
    url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={selected_coin}&tsym=USD&limit={days}' # Construct the URL
    response = requests.get(url) # Fetch data from the API
    data = json.loads(response.text) # Convert the response to a JSON object
    return data['Data']['Data'] # Return the historical price data

@cache.memoize(3600)  # cache for 1 hour
def get_data(selected_coin):
    return fetch_data_from_cryptocompare(selected_coin)

@cache.memoize(3600)  # cache for 1 hour
def predict_with_rf(selected_coin='BTC', forecast_days=10):
    # Fetch data using the new get_data function
    prices = get_data(selected_coin)

    # Convert data to DataFrame and preprocess
    df = pd.DataFrame(prices, columns=['time', 'low', 'high', 'open', 'close', 'volume', 'conversionType', 'conversionSymbol'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low']]
    
    # Check for missing values and handle them
    if df.isnull().any().any():
        df.dropna(inplace=True)
    
    # Feature engineering
    for i in range(1, 6):
        df[f'lag{i}'] = df['open'].shift(i)
    df['price_diff'] = df['open'].diff()
    df['volatility'] = df['high'] - df['low']
    df.dropna(inplace=True)

    # Splitting into training and validation sets
    train_size = int(0.8 * len(df))
    X_train, X_val = df.drop(columns=['open', 'high', 'low']).iloc[:train_size], df.drop(columns=['open', 'high', 'low']).iloc[train_size:]
    y_train, y_val = df['open'].iloc[:train_size], df['open'].iloc[train_size:]

    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)

    predictions = []

    for _ in range(forecast_days):
        last_date = df.index[-1]
        next_date = last_date + pd.DateOffset(days=1)

        next_features = {
            'lag1': df['open'].iloc[-1],
            'lag2': df['open'].iloc[-2],
            'lag3': df['open'].iloc[-3],
            'lag4': df['open'].iloc[-4],
            'lag5': df['open'].iloc[-5],
            'price_diff': df['price_diff'].iloc[-1],
            'volatility': df['volatility'].iloc[-1]
        }

        next_features_df = pd.DataFrame([next_features], columns=X_train.columns)
        prediction = model.predict(next_features_df)[0]
        predictions.append((next_date, prediction))

        new_row = {
            'open': prediction,
            'high': prediction,  # Placeholder
            'low': prediction,   # Placeholder
            'lag1': df['open'].iloc[-1],
            'lag2': df['open'].iloc[-2],
            'lag3': df['open'].iloc[-3],
            'lag4': df['open'].iloc[-4],
            'lag5': df['open'].iloc[-5],
            'price_diff': df['open'].iloc[-1] - df['open'].iloc[-2],
            'volatility': df['volatility'].iloc[-1]  # Placeholder
        }
        df.loc[next_date] = new_row


    return {
        'selected_coin': selected_coin,
        'predictions': predictions,
        'df': df[:-forecast_days]
    }


    
    
    
    
@cache.memoize(3600)  # cache for 1 hour
def predict_with_sarima(selected_coin='BTC', forecast_days=7):
    # Fetch data using the new get_data function
    prices = get_data(selected_coin)

    # Convert data to DataFrame and preprocess
    df = pd.DataFrame(prices, columns=['time', 'low', 'high', 'open', 'close', 'volume', 'conversionType', 'conversionSymbol'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['open']]

    # Check for missing values and handle them
    if df.isnull().any().any():
        df.dropna(inplace=True)
        
    # For the sake of simplicity, let's use fixed SARIMA parameters
    # Ideally, these parameters should be determined using a method like grid search
    p, d, q, P, D, Q, s = 1, 1, 1, 1, 1, 1, 7  # These are just sample parameters
    
    # Train the SARIMA model
    model = SARIMAX(df['open'], order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = model.fit(disp=False)
    
    # Predict future values
    forecast = results.get_forecast(steps=forecast_days)
    predictions = list(zip(forecast.predicted_mean.index, forecast.predicted_mean.values))

    return {
        'selected_coin': selected_coin,
        'predictions': predictions,
        'df': df
    }

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    graphJSON = None
    prediction = None
    selected_coin = None
    
    # Check if it's a GET request and if so, generate the default graph for Bitcoin with RF
    if request.method == 'GET':
        result = predict_with_rf("BTC")
        graphJSON = enhanced_generate_plot(result['df'], result['predictions'], "BTC", "RF").to_json()
        prediction = result['predictions'][-1][1]
        selected_coin = "BTC"
            
    if request.method == 'POST':
        coin = request.form.get('coin')
        model_type = request.form.get('model_type')

        # Depending on the model type, call the appropriate function
        if model_type == "RF":
            result = predict_with_rf(coin)
        elif model_type == "SARIMA":
            result = predict_with_sarima(coin)
        
        # Call the function with the additional parameters
        graphJSON = enhanced_generate_plot(result['df'], result['predictions'], coin, model_type).to_json()
        prediction = result['predictions'][-1][1]
        selected_coin = result['selected_coin']

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'graphJSON': graphJSON,
                'predicted_price': prediction,
                'selected_coin': selected_coin
            })

    return render_template('prediction.html', graphJSON=graphJSON, prediction=prediction, selected_coin=selected_coin)

@app.errorhandler(500)
def internal_error(error):
    return "Sorry, an unexpected error occurred. Please try again later.", 500


#===========================
# TRADE SIGNAL
#===========================
CRYPTOPANIC_API_ENDPOINT = "https://cryptopanic.com/api/v1/posts/"
@app.route('/tradesignal')
def trade_signal():
        return render_template('signal.html')

@app.route('/get_signal', methods=['POST'])
def get_trade_signal():
    symbol = request.form.get('symbol', 'BTC-USD')
    short_window = int(request.form.get('short_window', 5))
    long_window = int(request.form.get('long_window', 20))
    signal = generate_trade_signal(symbol, short_window, long_window)
    return signal

ROWS_PER_PAGE = 40

@app.route('/get_data', methods=['POST'])
def get_trade_data():
    symbol = request.form.get('symbol', 'BTC-USD')
    page = int(request.form.get('page', 1))
    data = fetch_trade_data(symbol)
    
    start_row = (page - 1) * ROWS_PER_PAGE
    end_row = start_row + ROWS_PER_PAGE
    paginated_data = data.iloc[start_row:end_row]

    formatted_data = [{
        "date": index.strftime('%d/%m/%Y'),
        "open": format_value(row["Open"]),
        "high": format_value(row["High"]),
        "low": format_value(row["Low"]),
        "close": format_value(row["Close"])
    } for index, row in paginated_data.iterrows()]
    
    return jsonify({
        "data": formatted_data,
        "total_pages": -(-len(data) // ROWS_PER_PAGE)
    })

def format_value(val):
    return "{:,.2f}".format(val)

@app.route('/get_chart', methods=['POST'])
def get_trade_chart():
    symbol = request.form.get('symbol', 'BTC-USD')
    data = fetch_trade_data(symbol)
    selectedIndicator = request.form.get('indicator', 'all')
    
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig_main.add_trace(go.Scatter(x=data.index, y=data['BB_High'], mode='lines', name='Bollinger High'))
    fig_main.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], mode='lines', name='Bollinger Low'))
    fig_main.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis=dict(
            title="Price",
            tickformat="$,.2f"  # Format y-axis ticks as "$12,345.67"
        ),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
    fig_rsi.update_layout(
        xaxis_title="Date",
        yaxis_title="RSI Value",
        yaxis=dict(
            title='RSI',
            range=[0, 100],
            tickformat=".2f"  # Format y-axis ticks as "12.34"
        ),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    charts = {
        'main': json.loads(json.dumps(fig_main, cls=plotly.utils.PlotlyJSONEncoder)),
        'rsi': json.loads(json.dumps(fig_rsi, cls=plotly.utils.PlotlyJSONEncoder))
    }
    return jsonify(charts)


# Add caching with timeout of 1 hour (3600 seconds)
@cache.cached(key_prefix='fetch_data', timeout=3600)
def fetch_trade_data(symbol, period='1y'):
    data = yf.download(symbol, period=period)
    if not data.index.tz:
        data.index = data.index.tz_localize(None)
    add_technical_indicators(data)
    return data

def add_technical_indicators(data):
    bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
    data["BB_High"] = bb.bollinger_hband()
    data["BB_Low"] = bb.bollinger_lband()

    rsi = RSIIndicator(close=data["Close"], window=14)
    data["RSI"] = rsi.rsi()

    macd = MACD(close=data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_signal"] = macd.macd_signal()

    stoch = StochasticOscillator(data["High"], data["Low"], data["Close"])
    data["Stoch_Oscillator"] = stoch.stoch()

    obv = OnBalanceVolumeIndicator(data["Close"], data["Volume"])
    data["OBV"] = obv.on_balance_volume()

news_cache = {}

# Add caching with timeout of 1 hour (3600 seconds)
@cache.cached(key_prefix='fetch_news', timeout=3600)
def fetch_trade_news(symbol):
    params = {
        "currencies": symbol,
        "public": "true"
    }
    response = requests.get(CRYPTOPANIC_API_ENDPOINT, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        return []

def analyze_trade_sentiment(articles):
    sentiments = []
    for article in articles:
        # Combine title, description, and content if available
        content_combined = article['title'] + " " + (article['description'] or "") + " " + (article.get('body', ''))  # Assuming 'body' is the key for the full content
        
        # Analyze sentiment
        analysis = TextBlob(content_combined)
        sentiment_polarity = analysis.sentiment.polarity
        
        # Adjust sentiment based on votes
        bullish_votes = article['votes']['bullish']
        bearish_votes = article['votes']['bearish']
        
        if bullish_votes > bearish_votes:
            sentiment_polarity += 0.5
        elif bearish_votes > bullish_votes:
            sentiment_polarity -= 0.5
        
        sentiments.append(sentiment_polarity)
    
    average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return average_sentiment


def fetch_and_analyze_news(symbol):
    current_time = time.time()
    if symbol in news_cache and (current_time - news_cache[symbol]['timestamp']) < 3600:
        return news_cache[symbol]['sentiment']
    articles = fetch_trade_news(symbol)
    sentiment = analyze_trade_sentiment(articles)
    news_cache[symbol] = {
        'timestamp': current_time,
        'sentiment': sentiment
    }
    return sentiment

# Add caching with timeout of 1 hour (3600 seconds)
@cache.memoize(timeout=3600)
def generate_trade_signal(symbol, short_window=5, long_window=20):
    data = fetch_trade_data(symbol)
    
    short_signals = moving_average_crossover(data, short_window, short_window * 2)
    long_signals = moving_average_crossover(data, long_window, long_window * 2)
    
    short_signal_latest = short_signals.iloc[-1]['Signal']
    long_signal_latest = long_signals.iloc[-1]['Signal']

    news_sentiment = fetch_and_analyze_news(symbol)

    # Classify sentiment based on thresholds
    if news_sentiment > 0.7:
        sentiment_description = "strongly positive"
    elif 0.3 < news_sentiment <= 0.7:
        sentiment_description = "positive"
    elif -0.3 <= news_sentiment <= 0.3:
        sentiment_description = "neutral"
    elif -0.7 <= news_sentiment < -0.3:
        sentiment_description = "negative"
    else:
        sentiment_description = "strongly negative"

    # Detailed explanations
    intro = f"Based on the recent price movements and news sentiment for {symbol}, "
    
    short_analysis = ""
    if short_signal_latest == 1:
        short_analysis += f"the price trend over the last {short_window} days suggests an upward momentum, indicating a potential buying opportunity. "
    elif short_signal_latest == -1:
        short_analysis += f"the price trend over the last {short_window} days suggests a downward momentum, indicating a potential selling point. "
    else:
        short_analysis += f"the price trend over the last {short_window} days is relatively stable, suggesting a holding position might be advisable. "
    
    long_analysis = ""
    if long_signal_latest == 1:
        long_analysis += f"Looking at a broader perspective over the last {long_window} days, the price trend indicates an upward trajectory, supporting a buy recommendation. "
    elif long_signal_latest == -1:
        long_analysis += f"From a long-term perspective spanning {long_window} days, the price trend shows a downward trajectory, supporting a sell recommendation. "
    else:
        long_analysis += f"Over the longer horizon of {long_window} days, the price seems stable, suggesting a neutral stance might be best. "
    
    sentiment_analysis = f"\nNews Sentiment: The recent news sentiment for {symbol} is {sentiment_description}. "
    if sentiment_description in ["strongly positive", "positive"]:
        sentiment_analysis += "This means most of the recent news articles and discussions about this cryptocurrency have been positive, which can be a good sign for potential growth."
    elif sentiment_description in ["negative", "strongly negative"]:
        sentiment_analysis += "This indicates that the majority of recent news and discussions around this cryptocurrency have been negative. It's essential to be cautious and research more about these negative sentiments before making decisions."
    else:
        sentiment_analysis += "This suggests that recent news and discussions are neither predominantly positive nor negative. It's always a good idea to delve deeper into the recent news to understand the market better."

    disclaimer = "\n\nPlease note: Trading cryptocurries is risky. It's essential to do your research, understand the market, and preferably consult with a financial expert before making any trading decisions."
    
    return {
        'shortAnalysis': intro + short_analysis,
        'longAnalysis': long_analysis,
        'sentimentAnalysis': sentiment_analysis + disclaimer
    }

def moving_average_crossover(data, short_window, long_window):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0

    signals['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    signals['Signal'][short_window:] = np.where(signals['Short_MA'][short_window:] > signals['Long_MA'][short_window:], 1.0, -1.0)
    signals['Positions'] = signals['Signal'].diff()
    signals['Buy_Signal'] = np.where(signals['Positions'] == 1, data['Close'], np.nan)
    signals['Sell_Signal'] = np.where(signals['Positions'] == -1, data['Close'], np.nan)

    return signals


#===========================
# PORTFOLIO
#===========================


#===========================
# ABOUT US
#===========================
@app.route('/about')
def about():
    return render_template('aboutus.html')

@app.route('/login')
def loginpage():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)





























"""
#Portfolio
app.config['SECRET_KEY'] = 'your_secret_key'

class CryptoForm(FlaskForm):
    asset_id = StringField('Asset', validators=[DataRequired()])
    quantity = DecimalField('Quantity', validators=[DataRequired()])
    purchase_price = DecimalField('Purchase Price', validators=[DataRequired()])

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    form = CryptoForm()

    if not session.get('portfolio_data'):
        session['portfolio_data'] = []

    portfolio_data = session['portfolio_data']
    
    if request.method == "POST" and form.validate():
        asset_id = form.asset_id.data
        quantity = form.quantity.data
        purchase_price = form.purchase_price.data

        lower_bound = purchase_price * Decimal('0.8')
        upper_bound = purchase_price * Decimal('1.2')
        current_price = random.uniform(float(lower_bound), float(upper_bound))
        value = current_price * float(quantity)

        new_item = {
            "asset": asset_id,
            "quantity": str(quantity),
            "purchase_price": str(purchase_price),
            "current_price": current_price,
            "value": value
        }
        portfolio_data.append(new_item)

        # Update the session data
        session['portfolio_data'] = portfolio_data

    total_value = round(sum(item["value"] for item in portfolio_data), 2)

    return render_template('portfolio.html', form=form, portfolio_data=portfolio_data, total_value=total_value)

@app.route('/download', methods=['GET'])
def download():
    portfolio_data = session.get('portfolio_data', [])

    json_data = json.dumps(portfolio_data, indent=4)

    response = make_response(json_data)
    response.headers['Content-Type'] = 'application/json'
    response.headers['Content-Disposition'] = 'attachment; filename=portfolio.json'

    return response
"""





"""
# Replace these with your own API keys and access tokens
consumer_key = 'Vic4eGhLaUXWXGw7GenQYZe3c' # KAN_API_KEY
consumer_secret = 'sFj8arTkPoNd2mqAKXsBVibw7knWH95c7yMvxVh7ay1Ve0emRC' # KAN_API_KEY
access_token = '1165642044418584578-QxPgAeQ29LEOzrqaYlh1Ixy8sZ622Y' # KAN_API_KEY
access_token_secret = '8f2qWJds9r7bRlYZlfieR9vHP6vtbQYSLbzvFzaEseVzx' # KAN_API_KEY

auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

def get_sentiment(crypto):
    # Search for tweets about the given cryptocurrency
    tweets = api.search_tweets(q='#cryptocurrency', lang='en')
    
    # Initialize counters for positive, negative, and neutral tweets
    positive = 0
    negative = 0
    neutral = 0
    
    # Iterate through the tweets and perform sentiment analysis
    for tweet in tweets:
        analysis = TextBlob(tweet.text)
        if analysis.sentiment.polarity > 0:
            positive += 1
        elif analysis.sentiment.polarity < 0:
            negative += 1
        else:
            neutral += 1
            
    # Calculate the total number of tweets
    total = positive + negative + neutral
    
    # Calculate the percentage of positive, negative, and neutral tweets
    positive_percent = positive / total * 100
    negative_percent = negative / total * 100
    neutral_percent = neutral / total * 100
    
    # Return the results as a dictionary
    return {
        'positive': positive_percent,
        'negative': negative_percent,
        'neutral': neutral_percent
    }

def show_sentiment_chart(sentiment):
    # Extract the positive, negative, and neutral percentages from the sentiment dictionary
    positive = sentiment['positive']
    negative = sentiment['negative']
    neutral = sentiment['neutral']

    # Set up the bar chart
    data = [go.Bar(
                x=['Positive', 'Negative', 'Neutral'],
                y=[positive, negative, neutral],
                marker=dict(color=['green', 'red', 'blue'])
           )]
    layout = go.Layout(title='Sentiment Analysis', yaxis=dict(title='Percentage'))
    fig = go.Figure(data=data, layout=layout)

    # Return the chart object
    return fig

@app.route('/sentiment')
def sentiment():
    # Get the cryptocurrency from the request query string
    crypto = request.args.get('crypto')
    
    # Perform sentiment analysis on the cryptocurrency
    sentiment = get_sentiment(crypto)
    
    # Generate the sentiment chart
    show_sentiment_chart(sentiment)
    
    # Render the template and pass the sentiment data to it
    return render_template('sentiment.html', crypto=crypto, sentiment=sentiment)


@app.route('/tweets')
def get_tweets():
    # Search for tweets about cryptocurrency
    tweets = api.search_tweets(q='#cryptocurrency', lang='en')
    
    # Extract the text and time of the first 10 tweets
    tweet_data = []
    
    for tweet in tweets[:10]:  # Get the first 10 tweets
        tweet_text = tweet.text
        tweet_time = tweet.created_at.strftime('%d/%m/%Y %H:%M:%S')
        tweet_data.append({'text': tweet_text, 'time': tweet_time})

    # Render the tweets.html template and pass in the tweet data
    return render_template('tweets.html', tweets=tweet_data)
"""