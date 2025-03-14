from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import io
import base64
import json
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Global variables to store data
stock_data = None
ticker_symbol = "BTC-USD"

def get_stock_data(period="7y"):
    """Download and process stock data"""
    global stock_data, ticker_symbol
    
    # Download data
    data = yf.download(ticker_symbol, period=period, group_by='column')
    
    # Flatten MultiIndex columns
    data.columns = data.columns.get_level_values(0)
    
    # Calculate technical indicators
    # SMA
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    
    # RSI
    period = 14
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    data['RSI14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    rolling_std = data['Close'].rolling(window=20).std()
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    data['BB_upper'] = data['BB_middle'] + (rolling_std * 2)
    data['BB_lower'] = data['BB_middle'] - (rolling_std * 2)
    
    # Trading Strategy 
    data['Signal'] = 0
    data.loc[data['SMA20'] > data['SMA50'], 'Signal'] = 1
    data.loc[data['SMA20'] < data['SMA50'], 'Signal'] = -1
    data['Trade_Signal'] = data['Signal'].shift(1)
    data['Market_Return'] = data['Close'].pct_change()
    data['Position_Change'] = data['Trade_Signal'].diff().abs()
    transaction_cost = 0.001
    data['Strategy_Return'] = data['Trade_Signal'] * data['Market_Return'] - transaction_cost * data['Position_Change']
    data['Cumulative_Market'] = (1 + data['Market_Return'].fillna(0)).cumprod()
    data['Cumulative_Strategy'] = (1 + data['Strategy_Return'].fillna(0)).cumprod()
    data['Trade_Change'] = data['Trade_Signal'].diff()
    
    stock_data = data
    return data

def create_price_volume_chart():
    """Create price and volume chart"""
    fig = Figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    
    ax1.plot(stock_data.index, stock_data['Close'], label='Close Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.bar(stock_data.index, stock_data['Volume'], alpha=0.3, color='gray', label='Volume')
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper right')
    
    fig.suptitle(f'{ticker_symbol} Price & Volume')
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the image to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def create_sma_chart():
    """Create chart with moving averages"""
    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    ax.plot(stock_data.index, stock_data['Close'], label='Close Price', alpha=0.7)
    ax.plot(stock_data.index, stock_data['SMA20'], label='20-day SMA', linestyle='--')
    ax.plot(stock_data.index, stock_data['SMA50'], label='50-day SMA', linestyle='-.')
    ax.plot(stock_data.index, stock_data['SMA200'], label='200-day SMA', linestyle=':')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{ticker_symbol} Price & Moving Averages')
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def create_rsi_chart():
    """Create RSI chart"""
    fig = Figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    
    ax.plot(stock_data.index, stock_data['RSI14'], label='RSI(14)', color='purple')
    ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI Value')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{ticker_symbol} RSI(14)')
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def create_bollinger_chart():
    """Create Bollinger Bands chart"""
    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    ax.plot(stock_data.index, stock_data['Close'], label='Close Price', alpha=0.7)
    ax.plot(stock_data.index, stock_data['BB_upper'], label='Upper Band', color='red', linestyle='--')
    ax.plot(stock_data.index, stock_data['BB_middle'], label='Middle Band', color='blue', linestyle='-')
    ax.plot(stock_data.index, stock_data['BB_lower'], label='Lower Band', color='green', linestyle='--')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{ticker_symbol} Bollinger Bands')
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def create_strategy_chart():
    """Create trading strategy performance chart"""
    fig = Figure(figsize=(12, 10))
    
    # Price and signals
    ax1 = fig.add_subplot(311)
    ax1.plot(stock_data.index, stock_data['Close'], label='Close Price', alpha=0.7)
    ax1.plot(stock_data.index, stock_data['SMA20'], label='20-day SMA', linestyle='--')
    ax1.plot(stock_data.index, stock_data['SMA50'], label='50-day SMA', linestyle='-.')
    
    # Buy/Sell signals
    buy_signals = stock_data[stock_data['Trade_Change'] == 2].index
    sell_signals = stock_data[stock_data['Trade_Change'] == -2].index
    
    if len(buy_signals) > 0:
        ax1.scatter(buy_signals, stock_data.loc[buy_signals, 'Close'], marker='^', color='g', s=80, label='Buy Signal')
    if len(sell_signals) > 0:
        ax1.scatter(sell_signals, stock_data.loc[sell_signals, 'Close'], marker='v', color='r', s=80, label='Sell Signal')
    
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Moving Average Crossover Strategy')
    
    # Cumulative returns
    ax2 = fig.add_subplot(312)
    ax2.plot(stock_data.index, stock_data['Cumulative_Market'], label='Market Returns')
    ax2.plot(stock_data.index, stock_data['Cumulative_Strategy'], label='Strategy Returns')
    ax2.set_ylabel('Cumulative Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Cumulative Returns Comparison')
    
    # Position
    ax3 = fig.add_subplot(313)
    ax3.plot(stock_data.index, stock_data['Signal'], label='Position')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.fill_between(stock_data.index, stock_data['Signal'], 0, where=stock_data['Signal'] > 0, color='g', alpha=0.3, label='Long')
    ax3.fill_between(stock_data.index, stock_data['Signal'], 0, where=stock_data['Signal'] < 0, color='r', alpha=0.3, label='Short')
    ax3.set_ylabel('Position')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Position Status')
    
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def calculate_performance_metrics():
    """Calculate performance metrics for trading strategy"""
    valid_data = stock_data.dropna()
    
    total_days = (valid_data.index[-1] - valid_data.index[0]).days
    annual_return_strategy = (valid_data['Cumulative_Strategy'].iloc[-1] ** (365 / total_days)) - 1
    annual_return_market = (valid_data['Cumulative_Market'].iloc[-1] ** (365 / total_days)) - 1
    
    daily_vol_strategy = valid_data['Strategy_Return'].std()
    annual_vol_strategy = daily_vol_strategy * np.sqrt(252)
    
    daily_vol_market = valid_data['Market_Return'].std()
    annual_vol_market = daily_vol_market * np.sqrt(252)
    
    sharpe_ratio_strategy = annual_return_strategy / annual_vol_strategy
    sharpe_ratio_market = annual_return_market / annual_vol_market
    
    cumulative_returns_strategy = valid_data['Cumulative_Strategy']
    running_max_strategy = cumulative_returns_strategy.cummax()
    drawdown_strategy = (cumulative_returns_strategy - running_max_strategy) / running_max_strategy
    max_drawdown_strategy = drawdown_strategy.min()
    
    cumulative_returns_market = valid_data['Cumulative_Market']
    running_max_market = cumulative_returns_market.cummax()
    drawdown_market = (cumulative_returns_market - running_max_market) / running_max_market
    max_drawdown_market = drawdown_market.min()
    
    metrics = {
        'start_date': valid_data.index[0].strftime('%Y-%m-%d'),
        'end_date': valid_data.index[-1].strftime('%Y-%m-%d'),
        'total_days': len(valid_data),
        'strategy': {
            'annual_return': f"{annual_return_strategy:.2%}",
            'annual_volatility': f"{annual_vol_strategy:.2%}",
            'sharpe_ratio': f"{sharpe_ratio_strategy:.2f}",
            'max_drawdown': f"{max_drawdown_strategy:.2%}",
            'final_return': f"{valid_data['Cumulative_Strategy'].iloc[-1]:.2f}"
        },
        'market': {
            'annual_return': f"{annual_return_market:.2%}",
            'annual_volatility': f"{annual_vol_market:.2%}",
            'sharpe_ratio': f"{sharpe_ratio_market:.2f}",
            'max_drawdown': f"{max_drawdown_market:.2%}",
            'final_return': f"{valid_data['Cumulative_Market'].iloc[-1]:.2f}"
        }
    }
    
    return metrics

@app.route('/')
def index():
    global stock_data
    
    if stock_data is None:
        get_stock_data()
    
    return render_template('index.html', ticker=ticker_symbol)

@app.route('/update_data', methods=['POST'])
def update_data():
    period = request.form.get('period', '7y')
    get_stock_data(period)
    
    return jsonify({'status': 'success'})

@app.route('/get_charts')
def get_charts():
    price_volume_chart = create_price_volume_chart()
    sma_chart = create_sma_chart()
    rsi_chart = create_rsi_chart()
    bollinger_chart = create_bollinger_chart()
    strategy_chart = create_strategy_chart()
    metrics = calculate_performance_metrics()
    
    return jsonify({
        'price_volume_chart': price_volume_chart,
        'sma_chart': sma_chart,
        'rsi_chart': rsi_chart,
        'bollinger_chart': bollinger_chart,
        'strategy_chart': strategy_chart,
        'metrics': metrics
    })

if __name__ == '__main__':
    app.run(debug=True) 