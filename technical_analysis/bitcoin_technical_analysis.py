# 基础库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta


# 设置绘图样式
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12

# 显示所有列
pd.set_option('display.max_columns', None)

ticker_symbol = "BTC-USD"

# 下载数据，虽然指定了 group_by='column'，返回的依然是 MultiIndex 列
stock_data = yf.download(ticker_symbol, period="7y", group_by='column')
print("原始数据：")
print(stock_data.head())

# 将 MultiIndex 列扁平化：取第一层作为新的列标签
stock_data.columns = stock_data.columns.get_level_values(0)
print("扁平化后的数据：")
print(stock_data.head())

# 保存为 CSV 文件
stock_data.to_csv(f"{ticker_symbol}_past_year_stock_data.csv")
print(f"{ticker_symbol} past 7 year stock data downloaded and saved to CSV file.")

# 查看数据基本信息
print("数据形状:", stock_data.shape)
print("\n数据类型:")
print(stock_data.dtypes)
print("\n基本统计信息:")
stock_data.describe()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# for f in fm.fontManager.ttflist:
#     # 只打印名称中包含 'PingFang' 或 'SC' 或 'Hei' 等关键字的字体
#     if "PingFang" in f.name or "Heiti" in f.name or "Songti" in f.name or "SC" in f.name:
#         print(f.name)

start_date = stock_data.index[0]
end_date = stock_data.index[-1]

plt.figure(figsize=(16, 8))

# 绘制收盘价
plt.plot(stock_data.index, stock_data['Close'], label='收盘价')

ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.bar(stock_data.index, stock_data['Volume'], alpha=0.3, color='gray', label='成交量')
ax2.set_ylabel('成交量')

plt.title(f'{ticker_symbol}股票价格走势 ({start_date.date()} 至 {end_date.date()})')
ax1.set_xlabel('日期')
ax1.set_ylabel('价格 (USD)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.grid(True, alpha=0.3)
plt.show()


# 计算移动平均线
stock_data['SMA20'] = stock_data['Close'].rolling(window=20).mean()  # 20日简单移动平均线
stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()  # 50日简单移动平均线
stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()  # 200日简单移动平均线

# 计算相对强弱指数 (RSI)
period=14
delta = stock_data['Close'].diff()
# 分离涨跌幅
gain = delta.clip(lower=0)         # 涨幅为正，其余为0
loss = -delta.clip(upper=0)        # 跌幅取绝对值，其余为0
# 使用 Wilder 平滑：alpha=1/period，min_periods=period，adjust=False
avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
# 计算RS及RSI
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
stock_data['RSI14'] = rsi

# 计算布林带
rolling_std = stock_data['Close'].rolling(window=20).std()
stock_data['BB_middle'] = stock_data['Close'].rolling(window=20).mean()
stock_data['BB_upper'] = stock_data['BB_middle'] + (rolling_std * 2)
stock_data['BB_lower'] = stock_data['BB_middle'] - (rolling_std * 2)


# 显示带有技术指标的数据
stock_data.tail()


# 绘制带有移动平均线的股票价格图
plt.figure(figsize=(16, 8))
plt.plot(stock_data.index, stock_data['Close'], label='收盘价', alpha=0.7)
plt.plot(stock_data.index, stock_data['SMA20'], label='20日均线', linestyle='--')
plt.plot(stock_data.index, stock_data['SMA50'], label='50日均线', linestyle='-.')
plt.plot(stock_data.index, stock_data['SMA200'], label='200日均线', linestyle=':')

plt.title(f'{ticker_symbol}股票价格和移动平均线')
plt.xlabel('日期')
plt.ylabel('价格 (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# 绘制RSI指标
plt.figure(figsize=(16, 5))
plt.plot(stock_data.index, stock_data['RSI14'], label='RSI(14)', color='purple')

# 添加超买超卖区域
plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='超买线 (70)')
plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='超卖线 (30)')
plt.axhline(y=50, color='gray', linestyle='-', alpha=0.3)

plt.title(f'{ticker_symbol} RSI(14)指标')
plt.xlabel('日期')
plt.ylabel('RSI值')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.show()


# 绘制布林带
plt.figure(figsize=(16, 8))
plt.plot(stock_data.index, stock_data['Close'], label='收盘价', alpha=0.7)
plt.plot(stock_data.index, stock_data['BB_upper'], label='上轨', color='red', linestyle='--')
plt.plot(stock_data.index, stock_data['BB_middle'], label='中轨', color='blue', linestyle='-')
plt.plot(stock_data.index, stock_data['BB_lower'], label='下轨', color='green', linestyle='--')

plt.title(f'{ticker_symbol}股票价格和布林带')
plt.xlabel('日期')
plt.ylabel('价格 (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据（使用随机游走模拟股票收盘价）
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=300, freq='B')
# 生成对数正态随机数据并累计作为价格
prices = np.random.lognormal(mean=0, sigma=0.02, size=len(dates)).cumprod() * 100
stock_data = pd.DataFrame({'Close': prices}, index=dates)

# 计算短期（20日）和长期（50日）移动平均线
stock_data['SMA_Fast'] = stock_data['Close'].rolling(window=20).mean()
stock_data['SMA_Slow'] = stock_data['Close'].rolling(window=50).mean()

# 根据均线交叉生成原始信号：20日均线大于50日均线时信号为1（买入），反之为-1（卖出）
stock_data['Signal'] = 0
stock_data.loc[stock_data['SMA_Fast'] > stock_data['SMA_Slow'], 'Signal'] = 1
stock_data.loc[stock_data['SMA_Fast'] < stock_data['SMA_Slow'], 'Signal'] = -1

# 为避免当日已知收盘价就进行交易，将信号延迟一天，模拟“下一日开盘”交易
stock_data['Trade_Signal'] = stock_data['Signal'].shift(1)

# 计算市场的日收益率（基于收盘价的百分比变化）
stock_data['Market_Return'] = stock_data['Close'].pct_change()

# 计算仓位变化（买卖时仓位的变化会产生交易成本）
stock_data['Position_Change'] = stock_data['Trade_Signal'].diff().abs()

# 设置交易成本比例（例如：每次交易成本为0.1%）
transaction_cost = 0.001

# 计算策略的日收益率：使用前一日的交易信号乘以当天市场收益，同时扣除因仓位变动产生的交易费用
stock_data['Strategy_Return'] = stock_data['Trade_Signal'] * stock_data['Market_Return'] - transaction_cost * stock_data['Position_Change']

# 计算累计收益，初始资金设为1
stock_data['Cumulative_Market'] = (1 + stock_data['Market_Return'].fillna(0)).cumprod()
stock_data['Cumulative_Strategy'] = (1 + stock_data['Strategy_Return'].fillna(0)).cumprod()

# 绘制结果
plt.figure(figsize=(16, 10))

# 子图1：收盘价、均线及买卖信号标记
plt.subplot(3, 1, 1)
plt.plot(stock_data.index, stock_data['Close'], label='收盘价', alpha=0.7)
plt.plot(stock_data.index, stock_data['SMA_Fast'], label='20日均线', linestyle='--')
plt.plot(stock_data.index, stock_data['SMA_Slow'], label='50日均线', linestyle='-.')
# 计算交易信号的变化，便于标记买入卖出点（当 Trade_Signal 从 -1 跃变到 1，diff 为2；反之为-2）
stock_data['Trade_Change'] = stock_data['Trade_Signal'].diff()
buy_signals = stock_data[stock_data['Trade_Change'] == 2].index
sell_signals = stock_data[stock_data['Trade_Change'] == -2].index

plt.scatter(buy_signals, stock_data.loc[buy_signals, 'Close'], marker='^', color='g', s=100, label='买入信号')
plt.scatter(sell_signals, stock_data.loc[sell_signals, 'Close'], marker='v', color='r', s=100, label='卖出信号')
plt.title('移动平均线交叉策略')
plt.xlabel('日期')
plt.ylabel('价格 (USD)')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：累计收益对比（市场 vs 策略）
plt.subplot(3, 1, 2)
plt.plot(stock_data.index, stock_data['Cumulative_Market'], label='市场累计收益')
plt.plot(stock_data.index, stock_data['Cumulative_Strategy'], label='策略累计收益')
plt.title('累计收益对比')
plt.xlabel('日期')
plt.ylabel('累计收益 (初始资金=1)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# # 绘制持仓状态
plt.subplot(3, 1, 3)
plt.plot(stock_data.index, stock_data['Signal'], label='持仓状态')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

plt.fill_between(stock_data.index, stock_data['Signal'], 0, where=stock_data['Signal'] > 0, color='g', alpha=0.3, label='多头')
plt.fill_between(stock_data.index, stock_data['Signal'], 0, where=stock_data['Signal'] < 0, color='r', alpha=0.3, label='空头')

plt.title('持仓状态')
plt.xlabel('日期')
plt.ylabel('仓位')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 选择有效数据（去除NaN）
valid_data = stock_data.dropna()

# 计算年化收益率
total_days = (valid_data.index[-1] - valid_data.index[0]).days
annual_return_strategy = (valid_data['Cumulative_Strategy'].iloc[-1] ** (365 / total_days)) - 1
annual_return_market = (valid_data['Cumulative_Market'].iloc[-1] ** (365 / total_days)) - 1

# 计算波动率（标准差）
daily_vol_strategy = valid_data['Strategy_Return'].std()
annual_vol_strategy = daily_vol_strategy * np.sqrt(252)  # 假设一年252个交易日

daily_vol_market = valid_data['Market_Return'].std()  # 使用计算好的市场日收益率
annual_vol_market = daily_vol_market * np.sqrt(252)

# 计算夏普比率（假设无风险收益率为0）
sharpe_ratio_strategy = annual_return_strategy / annual_vol_strategy
sharpe_ratio_market = annual_return_market / annual_vol_market

# 计算最大回撤
cumulative_returns_strategy = valid_data['Cumulative_Strategy']
running_max_strategy = cumulative_returns_strategy.cummax()
drawdown_strategy = (cumulative_returns_strategy - running_max_strategy) / running_max_strategy
max_drawdown_strategy = drawdown_strategy.min()

cumulative_returns_market = valid_data['Cumulative_Market']
running_max_market = cumulative_returns_market.cummax()
drawdown_market = (cumulative_returns_market - running_max_market) / running_max_market
max_drawdown_market = drawdown_market.min()

# 打印结果
print(f"评估时间段: {valid_data.index[0].date()} 至 {valid_data.index[-1].date()}")
print(f"总交易天数: {len(valid_data)}")
print("\n--- 策略表现 ---")
print(f"年化收益率: {annual_return_strategy:.2%}")
print(f"年化波动率: {annual_vol_strategy:.2%}")
print(f"夏普比率: {sharpe_ratio_strategy:.2f}")
print(f"最大回撤: {max_drawdown_strategy:.2%}")
print("\n--- 基准表现 (Buy & Hold) ---")
print(f"年化收益率: {annual_return_market:.2%}")
print(f"年化波动率: {annual_vol_market:.2%}")
print(f"夏普比率: {sharpe_ratio_market:.2f}")
print(f"最大回撤: {max_drawdown_market:.2%}")
