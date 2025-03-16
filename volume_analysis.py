import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
TICKER = "BTC-USD"
PERIOD = "7y"
OUTPUT_DIR = Path.home() / "Downloads"  # Cross-platform output directory

def setup_plot_style():
    """Configure the plot style for consistent visualization."""
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (14, 7)
    plt.rcParams['font.size'] = 12
    
    # Support Chinese characters if needed
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Show all columns in pandas DataFrames
    pd.set_option('display.max_columns', None)

def download_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """
    Download stock data using yfinance.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period to download
        
    Returns:
        DataFrame with downloaded stock data
    """
    try:
        logger.info(f"Downloading data for {ticker} over {period}")
        stock_data = yf.download(ticker, period=period, group_by='column')
        
        # Flatten MultiIndex columns for simpler access
        stock_data.columns = stock_data.columns.get_level_values(0)
        
        logger.info(f"Successfully downloaded {len(stock_data)} rows of data")
        return stock_data
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise

def save_data(data: pd.DataFrame, filename: str, directory: Path = OUTPUT_DIR) -> Path:
    """
    Save DataFrame to CSV file.
    
    Args:
        data: DataFrame to save
        filename: Filename to save to
        directory: Directory to save to
        
    Returns:
        Path to saved file
    """
    try:
        # Ensure directory exists
        directory.mkdir(parents=True, exist_ok=True)
        
        # Create full path
        filepath = directory / filename
        
        # Save data
        data.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        logger.info(f"Loading data from {filepath}")
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def calculate_basic_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic price metrics.
    
    Args:
        data: DataFrame with price data
        
    Returns:
        DataFrame with additional metrics
    """
    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()
    
    # Calculate previous close
    df['pre_close'] = df['Close'].shift(1)
    
    # Calculate change and percent change
    df['change'] = df['Close'] - df['pre_close']
    df['pct_chg'] = (df['change'] / df['pre_close'] * 100).round(2)
    
    # Calculate amount (volume * close price)
    df['amount'] = df['Volume'] * df['Close']
    
    # Add timestamp in Unix format
    df['ts'] = df.index.astype(np.int64) // 10**9
    
    # Add code column
    df['code'] = TICKER
    
    return df

def calculate_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate moving averages for price and volume.
    
    Args:
        data: DataFrame with price and volume data
        
    Returns:
        DataFrame with additional moving average metrics
    """
    df = data.copy()
    
    # Price moving averages
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma10'] = df['Close'].rolling(window=10).mean()
    df['ma20'] = df['Close'].rolling(window=20).mean()
    df['ma60'] = df['Close'].rolling(window=60).mean()
    
    # Volume moving averages
    df['vol_ma5'] = df['Volume'].rolling(window=5).mean()
    df['vol_ma10'] = df['Volume'].rolling(window=10).mean()
    df['vol_ma20'] = df['Volume'].rolling(window=20).mean()
    df['vol_ma'] = df['vol_ma5']  # Alias for consistency
    
    # Volume ratio
    df['vol_ratio'] = df['Volume'] / df['vol_ma']
    
    return df

def calculate_obv(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        data: DataFrame with price and volume data
        
    Returns:
        DataFrame with OBV metric
    """
    df = data.copy()
    
    # Initialize OBV column
    df['OBV'] = 0
    
    # Vectorized calculation where possible
    price_diff = df['Close'].diff()
    
    # Initial loop-based calculation (more readable)
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV'] + df.loc[df.index[i], 'Volume']
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV'] - df.loc[df.index[i], 'Volume']
        else:
            df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV']
    
    return df

def calculate_mfi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Money Flow Index (MFI).
    
    Args:
        data: DataFrame with price and volume data
        period: Period for MFI calculation
        
    Returns:
        DataFrame with MFI metric
    """
    df = data.copy()
    
    # Calculate typical price
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Calculate money flow
    df['money_flow'] = df['typical_price'] * df['Volume']
    
    # Initialize positive and negative flow columns
    df['positive_flow'] = 0.0
    df['negative_flow'] = 0.0
    
    # Calculate positive and negative money flow
    for i in range(1, len(df)):
        if df['typical_price'].iloc[i] > df['typical_price'].iloc[i-1]:
            df.loc[df.index[i], 'positive_flow'] = df.loc[df.index[i], 'money_flow']
        elif df['typical_price'].iloc[i] < df['typical_price'].iloc[i-1]:
            df.loc[df.index[i], 'negative_flow'] = df.loc[df.index[i], 'money_flow']
    
    # Calculate sums over the period
    df['positive_flow_sum'] = df['positive_flow'].rolling(window=period).sum()
    df['negative_flow_sum'] = df['negative_flow'].rolling(window=period).sum()
    
    # Calculate money ratio, avoiding division by zero
    df['money_ratio'] = np.where(
        df['negative_flow_sum'] > 0,
        df['positive_flow_sum'] / df['negative_flow_sum'],
        100  # Default value when negative flow sum is zero
    )
    
    # Calculate MFI
    df['MFI'] = 100 - (100 / (1 + df['money_ratio']))
    
    # Clean up intermediate columns to keep the DataFrame tidy
    df = df.drop(['typical_price', 'money_flow', 'positive_flow', 'negative_flow', 
                  'positive_flow_sum', 'negative_flow_sum', 'money_ratio'], axis=1)
    
    return df

def plot_price_volume_chart(data: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Create and save a price and volume chart.
    
    Args:
        data: DataFrame with price and volume data
        save_path: Path to save the chart
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                  gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price and MAs on the first subplot
    ax1.plot(data.index, data['Close'], color='black', linewidth=1.5, label='Closing Price')
    ax1.plot(data.index, data['ma5'], color='red', linewidth=1, label='MA5')
    ax1.plot(data.index, data['ma10'], color='blue', linewidth=1, label='MA10')
    ax1.plot(data.index, data['ma20'], color='green', linewidth=1, label='MA20')
    ax1.plot(data.index, data['ma60'], color='purple', linewidth=1.5, label='MA60')
    ax1.set_title(f'{TICKER} Price Trend and Moving Averages', fontsize=16)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    
    # Plot volume on the second subplot
    ax2.bar(data.index, data['Volume'], color='gray', alpha=0.6, label='Volume')
    ax2.set_title(f'{TICKER} Trading Volume', fontsize=16)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Adjust interval for readability
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Chart saved to {save_path}")
    
    plt.close(fig)

def plot_obv_comparison(data: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Create and save an OBV comparison chart.
    
    Args:
        data: DataFrame with price and OBV data
        save_path: Path to save the chart
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot closing price
    ax1.plot(data.index, data['Close'], color='blue', linewidth=1.5)
    ax1.set_title(f'{TICKER} Closing Price', fontsize=16)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot OBV indicator
    ax2.plot(data.index, data['OBV'], color='green', linewidth=1.5)
    ax2.set_title('On-Balance Volume (OBV) Indicator', fontsize=16)
    ax2.set_ylabel('OBV', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"OBV comparison chart saved to {save_path}")
    
    plt.close(fig)

def plot_comprehensive_chart(data: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Create and save a comprehensive analysis chart with multiple panels.
    
    Args:
        data: DataFrame with all calculated metrics
        save_path: Path to save the chart
    """
    # Create the comprehensive chart with 4 panels
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 20), sharex=True, 
                                           gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    
    # 1. Plot closing price with moving averages and OBV
    ax1.plot(data.index, data['Close'], color='blue', linewidth=1.5, label='Closing Price')
    ax1.plot(data.index, data['ma5'], color='red', linewidth=1, label='MA5')
    ax1.plot(data.index, data['ma10'], color='green', linewidth=1, label='MA10')
    ax1.plot(data.index, data['ma20'], color='purple', linewidth=1, label='MA20')
    ax1.set_title(f'{TICKER} Price Trend and OBV Comparison', fontsize=16)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    
    # Create a twin axis for OBV
    ax1_twin = ax1.twinx()
    ax1_twin.plot(data.index, data['OBV'], color='darkred', linewidth=1, label='OBV')
    ax1_twin.set_ylabel('OBV', color='darkred', fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor='darkred')
    ax1_twin.legend(loc='upper right')
    
    # 2. Plot Volume with MAs
    ax2.bar(data.index, data['Volume'], color='gray', alpha=0.6, label='Volume')
    ax2.plot(data.index, data['vol_ma5'], color='orange', linewidth=1.5, label='Volume MA5')
    ax2.plot(data.index, data['vol_ma10'], color='blue', linewidth=1.5, label='Volume MA10')
    ax2.plot(data.index, data['vol_ma20'], color='red', linewidth=1.5, label='Volume MA20')
    ax2.set_title('Volume with Moving Averages', fontsize=16)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left')
    
    # 3. Plot Volume Ratio with reference lines
    ax3.plot(data.index, data['vol_ratio'], color='darkblue', linewidth=1.5, label='Volume Ratio')
    ax3.axhline(y=1, color='green', linestyle='-', alpha=0.7, label='Normal Level (1.0)')
    ax3.axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='High Volume (2.0)')
    ax3.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='Very High Volume (3.0)')
    ax3.set_title('Volume Ratio (Relative to 5-day Average)', fontsize=16)
    ax3.set_ylabel('Ratio', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left')
    
    # 4. Plot MFI with overbought/oversold lines
    ax4.plot(data.index, data['MFI'], color='black', linewidth=1.5, label='MFI')
    ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Oversold (20)')
    ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Overbought (80)')
    ax4.set_title('Money Flow Index (MFI)', fontsize=16)
    ax4.set_ylabel('MFI Value', fontsize=12)
    ax4.set_ylim(0, 100)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(loc='upper left')
    ax4.set_xlabel('Date', fontsize=12)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Comprehensive chart saved to {save_path}")
    
    plt.close(fig)

def main():
    """Main function to orchestrate the analysis process."""
    try:
        # Set up plot style
        setup_plot_style()
        
        # Download data
        stock_data = download_stock_data(TICKER, PERIOD)
        
        # Save raw data
        raw_data_path = save_data(stock_data, f"{TICKER}_raw_data.csv")
        
        # Calculate all metrics in sequence
        enhanced_data = calculate_basic_metrics(stock_data)
        enhanced_data = calculate_moving_averages(enhanced_data)
        enhanced_data = calculate_obv(enhanced_data)
        enhanced_data = calculate_mfi(enhanced_data)
        
        # Save enhanced data
        enhanced_data_path = save_data(enhanced_data, f"{TICKER}_enhanced_data.csv")
        
        # Create and save all charts
        price_volume_chart_path = OUTPUT_DIR / f"{TICKER}_Price_Volume_Chart.png"
        plot_price_volume_chart(enhanced_data, price_volume_chart_path)
        
        obv_chart_path = OUTPUT_DIR / f"{TICKER}_OBV_Price_Comparison.png"
        plot_obv_comparison(enhanced_data, obv_chart_path)
        
        comprehensive_chart_path = OUTPUT_DIR / f"{TICKER}_Comprehensive_Analysis.png"
        plot_comprehensive_chart(enhanced_data, comprehensive_chart_path)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()