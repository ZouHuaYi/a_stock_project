# -*- coding: utf-8 -*-
"""股票技术指标计算工具模块"""

from venv import logger
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple

def calculate_technical_indicators(df: pd.DataFrame, ma_periods: List[int] = [5, 10, 20, 30, 60], 
                                 vol_periods: List[int] = [5, 10]) -> Tuple[pd.DataFrame, Dict]:
    """
    通用技术指标计算函数，计算常用的各种技术指标
    
    参数:
        df (pd.DataFrame): 原始股票数据，需要包含'open', 'high', 'low', 'close', 'volume'列
        ma_periods (List[int]): 移动平均线周期列表
        vol_periods (List[int]): 成交量移动平均线周期列表
        
    返回:
        pd.DataFrame: 增加了计算指标的DataFrame
    """
    if df.empty:
        return df
    
    # 确保索引已按日期排序
    # df = df.sort_index()
    
    indicators = {}
    # 创建结果DataFrame的副本
    result_df = df.copy()
    # 计算移动平均线
    for period in ma_periods:
        result_df[f'MA{period}'] = result_df['close'].rolling(window=period).mean()
        # 计算简单移动平均线
        indicators[f'SMA_{period}'] = result_df.ta.sma(length=period, close='close')

    # 计算成交量均线
    for period in vol_periods:
        result_df[f'VOL_MA{period}'] = result_df['volume'].rolling(window=period).mean()
    
    # 计算指数移动平均线
    result_df['EMA12'] = result_df['close'].ewm(span=12, adjust=False).mean()
    result_df['EMA26'] = result_df['close'].ewm(span=26, adjust=False).mean()
    
    # 计算MACD指标
    result_df['MACD_DIF'] = result_df['EMA12'] - result_df['EMA26']
    result_df['MACD_DEA'] = result_df['MACD_DIF'].ewm(span=9, adjust=False).mean()
    result_df['MACD_BAR'] = 2 * (result_df['MACD_DIF'] - result_df['MACD_DEA'])
    
    # 计算KDJ指标 - 使用浮点类型初始化
    result_df['KDJ_K'] = 50.0
    result_df['KDJ_D'] = 50.0
    result_df['KDJ_J'] = 50.0
    
    # 计算9日内的最低价和最高价
    low_9 = result_df['low'].rolling(window=9).min()
    high_9 = result_df['high'].rolling(window=9).max()
    
    # 计算KDJ指标，使用loc避免链式赋值警告
    for i in range(9, len(result_df)):
        # 计算RSV
        idx = result_df.index[i]
        prev_idx = result_df.index[i-1]
        
        if high_9.iloc[i] != low_9.iloc[i]:
            rsv = 100 * (result_df.loc[idx, 'close'] - low_9.iloc[i]) / (high_9.iloc[i] - low_9.iloc[i])
        else:
            rsv = 50.0
            
        # 计算K值
        result_df.loc[idx, 'KDJ_K'] = 2/3 * result_df.loc[prev_idx, 'KDJ_K'] + 1/3 * rsv
        
        # 计算D值
        result_df.loc[idx, 'KDJ_D'] = 2/3 * result_df.loc[prev_idx, 'KDJ_D'] + 1/3 * result_df.loc[idx, 'KDJ_K']
        
        # 计算J值
        result_df.loc[idx, 'KDJ_J'] = 3 * result_df.loc[idx, 'KDJ_K'] - 2 * result_df.loc[idx, 'KDJ_D']
    
    # 计算RSI指标
    delta = result_df['close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = -loss
    
    avg_gain_14 = gain.rolling(window=14).mean()
    avg_loss_14 = loss.rolling(window=14).mean()
    
    rs_14 = avg_gain_14 / avg_loss_14
    result_df['RSI_14'] = 100 - (100 / (1 + rs_14))
    
    # 计算布林带
    result_df['BOLL_MID'] = result_df['close'].rolling(window=20).mean()
    result_df['BOLL_STD'] = result_df['close'].rolling(window=20).std()
    result_df['BOLL_UP'] = result_df['BOLL_MID'] + 2 * result_df['BOLL_STD']
    result_df['BOLL_DOWN'] = result_df['BOLL_MID'] - 2 * result_df['BOLL_STD']
    
    # 计算涨跌幅
    result_df['change_pct'] = result_df['close'].pct_change() * 100
    
    # 计算成交量变化率
    result_df['volume_ratio'] = result_df['volume'] / result_df['VOL_MA5']
    
    # 计算振幅
    result_df['amplitude'] = (result_df['high'] - result_df['low']) / result_df['close'].shift(1) * 100

    # 相对强弱指数(RSI)
    indicators['RSI_6'] = result_df.ta.rsi(length=6, close='close')
    indicators['RSI_12'] = result_df.ta.rsi(length=12, close='close')
    indicators['RSI_14'] = result_df.ta.rsi(length=14, close='close')

    # MACD
    macd = result_df.ta.macd(fast=12, slow=26, signal=9, close='close')
    indicators['MACD'] = macd['MACD_12_26_9']
    indicators['MACD_signal'] = macd['MACDs_12_26_9']
    indicators['MACD_hist'] = macd['MACDh_12_26_9']

    # KDJ指标
    stoch = result_df.ta.stoch(high='high', low='low', close='close', k=9, d=3, smooth_k=3)
    indicators['KDJ_K'] = stoch['STOCHk_9_3_3']
    indicators['KDJ_D'] = stoch['STOCHd_9_3_3']
    indicators['KDJ_J'] = 3 * stoch['STOCHk_9_3_3'] - 2 * stoch['STOCHd_9_3_3']

    # 布林带
    bbands = result_df.ta.bbands(length=20, close='close')
    indicators['BB_upper'] = bbands['BBU_20_2.0']
    indicators['BB_middle'] = bbands['BBM_20_2.0']
    indicators['BB_lower'] = bbands['BBL_20_2.0']

    return result_df, indicators

def calculate_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算基本技术指标，包括移动平均线、成交量均线等
    
    参数:
        df (pd.DataFrame): 原始股票数据，需要包含'open', 'high', 'low', 'close', 'volume'列
        
    返回:
        pd.DataFrame: 增加了计算指标的DataFrame
    """
    # 调用通用计算函数实现
    return calculate_technical_indicators(df)

def calculate_fibonacci_levels(df: pd.DataFrame, use_swing: bool = False) -> Dict[str, float]:
    """
    计算斐波那契回调水平
    
    参数:
        df (pd.DataFrame): 股票数据，需要包含'high'和'low'列
        use_swing (bool): 是否使用波段高低点而不是简单的最高最低点
        
    返回:
        Dict[str, float]: 计算出的各级别斐波那契回调水平
    """
    if df.empty:
        return {}
    
    if use_swing:
        # 找到波段高低点（简单实现，可以进一步优化）
        # 此处算法可根据实际需求复杂化
        low_idx = df['low'].idxmin()
        # 确保高点在低点之后
        if low_idx == df.index[-1]:
            # 如果低点是最后一个点，则取之前的最低点
            low_idx = df.iloc[:-1]['low'].idxmin()
        
        # 只在低点之后找高点
        high_df = df.loc[low_idx:]
        high_idx = high_df['high'].idxmax()
        
        lowest_price = df.loc[low_idx, 'low']
        highest_price = df.loc[high_idx, 'high']
    else:
        # 简单地找出区间内的最高点和最低点
        lowest_price = df['low'].min()
        highest_price = df['high'].max()
    
    price_diff = highest_price - lowest_price
    
    if price_diff <= 0:
        return {}
    
    # 计算各级别的斐波那契回调位
    fib_levels = {
        'Fib 0.0% (High)': highest_price,
        'Fib 23.6%': highest_price - price_diff * 0.236,
        'Fib 38.2%': highest_price - price_diff * 0.382,
        'Fib 50.0%': highest_price - price_diff * 0.5,
        'Fib 61.8%': highest_price - price_diff * 0.618,
        'Fib 78.6%': highest_price - price_diff * 0.786,
        'Fib 100% (Low)': lowest_price,
        'Fib 161.8%': highest_price + price_diff * 0.618,
        'Fib 261.8%': highest_price + price_diff * 1.618
    }
    
    return fib_levels

def detect_patterns(df: pd.DataFrame) -> List[Dict]:
    """
    检测K线形态模式
    
    参数:
        df (pd.DataFrame): 股票数据，需要包含OHLC和技术指标
        
    返回:
        List[Dict]: 发现的形态模式列表
    """
    if df.empty or len(df) < 20:
        return []
    
    patterns = []
    
    # 计算20日均线方向
    ma20_direction = 1 if df['MA20'].iloc[-1] > df['MA20'].iloc[-5] else -1
    
    # 检测成交量变化
    vol_decrease = df['volume'].iloc[-1] < df['VOL_MA5'].iloc[-1]
    vol_increase = df['volume'].iloc[-1] > df['VOL_MA5'].iloc[-1] * 1.5
    
    # 检测价格横盘
    price_range = (df['high'].iloc[-10:].max() - df['low'].iloc[-10:].min()) / df['close'].iloc[-10] 
    price_stable = price_range < 0.05  # 5%以内认为是横盘
    
    # 检测下影线
    recent_data = df.iloc[-5:]
    lower_shadows = []
    
    for idx, row in recent_data.iterrows():
        body = abs(row['close'] - row['open'])
        if body == 0:
            body = 0.001  # 避免除以零
        
        lower_shadow = (min(row['open'], row['close']) - row['low']) / body
        if lower_shadow > 1.5:  # 下影线是实体的1.5倍以上
            lower_shadows.append({
                'date': idx, 
                'ratio': lower_shadow,
                'price': row['low']
            })
    
    # 添加发现的模式
    if vol_decrease and price_stable:
        patterns.append({
            'type': '横盘震荡',
            'description': f'价格在{price_range:.2%}范围内震荡，成交量萎缩',
            'confidence': 0.7
        })
    
    if vol_decrease:
        patterns.append({
            'type': '成交量萎缩',
            'description': f'成交量低于5日均量，当前成交量与5日均量比值为{df["volume"].iloc[-1]/df["VOL_MA5"].iloc[-1]:.2f}',
            'confidence': 0.6
        })
    
    if lower_shadows:
        patterns.append({
            'type': '下影线信号',
            'description': f'近期出现{len(lower_shadows)}次明显下影线',
            'shadows': lower_shadows,
            'confidence': 0.5
        })
    
    if vol_increase and df['close'].iloc[-1] > df['close'].iloc[-2] * 1.03:
        patterns.append({
            'type': '放量上涨',
            'description': f'成交量是5日均量的{df["volume"].iloc[-1]/df["VOL_MA5"].iloc[-1]:.2f}倍，股价上涨{(df["close"].iloc[-1]/df["close"].iloc[-2]-1)*100:.2f}%',
            'confidence': 0.8
        })
    
    return patterns

def calculate_support_resistance(df: pd.DataFrame, period: int = 20) -> Dict[str, List[float]]:
    """
    计算支撑位和阻力位
    
    参数:
        df (pd.DataFrame): 股票数据，需要包含OHLC
        period (int): 计算周期，默认20天
        
    返回:
        Dict[str, List[float]]: 支撑位和阻力位
    """
    if len(df) < period:
        return {'support': [], 'resistance': []}
    
    # 使用最近period天的数据
    recent_df = df.iloc[-period:]
    
    # 寻找局部极小值作为支撑位
    supports = []
    for i in range(1, len(recent_df) - 1):
        if recent_df['low'].iloc[i] < recent_df['low'].iloc[i-1] and recent_df['low'].iloc[i] < recent_df['low'].iloc[i+1]:
            supports.append(recent_df['low'].iloc[i])
    
    # 寻找局部极大值作为阻力位
    resistances = []
    for i in range(1, len(recent_df) - 1):
        if recent_df['high'].iloc[i] > recent_df['high'].iloc[i-1] and recent_df['high'].iloc[i] > recent_df['high'].iloc[i+1]:
            resistances.append(recent_df['high'].iloc[i])
    
    # 添加最低价和最高价
    if recent_df['low'].min() not in supports:
        supports.append(recent_df['low'].min())
    if recent_df['high'].max() not in resistances:
        resistances.append(recent_df['high'].max())
    
    # 添加均线作为动态支撑/阻力
    current_price = recent_df['close'].iloc[-1]
    
    for ma in ['MA5', 'MA10', 'MA20', 'MA60']:
        if ma in recent_df.columns:
            ma_value = recent_df[ma].iloc[-1]
            if not np.isnan(ma_value):
                if ma_value < current_price and ma_value not in supports:
                    supports.append(ma_value)
                elif ma_value > current_price and ma_value not in resistances:
                    resistances.append(ma_value)
    
    # 排序并去重
    supports = sorted(list(set(supports)))
    resistances = sorted(list(set(resistances)))
    
    return {'support': supports, 'resistance': resistances}

def plot_stock_chart(df: pd.DataFrame, 
                     indicators: Dict,
                     title: str = None, 
                     save_path: str = None, 
                   plot_ma: bool = True, 
                   plot_macd: bool = True,
                   plot_volume: bool = True, 
                   plot_kdj: bool = True,
                   plot_rsi: bool = True,
                   plot_boll: bool = True
                  ) -> bool:
    """
    通用股票图表绘制函数
    
    参数:
        df (pd.DataFrame): 股票数据，需要包含OHLC和技术指标
        indicators (Dict): 技术指标字典
        title (str): 图表标题
        save_path (str): 保存路径
        plot_ma (bool): 是否绘制移动平均线
        plot_volume (bool): 是否绘制成交量
        plot_macd (bool): 是否绘制MACD指标
        plot_kdj (bool): 是否绘制KDJ指标
        plot_rsi (bool): 是否绘制RSI指标
        plot_boll (bool): 是否绘制布林带
    返回:
        bool: 是否成功绘制图表
    """
    if df.empty:
        return False
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    plt.figure(figsize=(15, 25))
    plt.title(title)
    # 确保所有指标与数据索引对齐,空出
    data_indices = df.index
    
    if plot_ma:
        # 价格和移动平均线
        plt.subplot(6, 1, 1)
        # 画蜡烛图 
        plt.plot(data_indices, df['close'], label='收盘价')
        plt.plot(data_indices, df['MA5'].reindex(data_indices), label='5日均线')
        plt.plot(data_indices, df['MA10'].reindex(data_indices), label='10日均线')
        plt.plot(data_indices, df['MA20'].reindex(data_indices), label='20日均线')
        plt.plot(data_indices, df['MA60'].reindex(data_indices), label='60日均线')
        plt.title(f'{title} 价格走势')
        plt.legend()

    if plot_macd:
        # MACD
        plt.subplot(6, 1, 2)
        plt.plot(data_indices, indicators['MACD'].reindex(data_indices), label='MACD')
        plt.plot(data_indices, indicators['MACD_signal'].reindex(data_indices), label='信号线')
        plt.bar(data_indices, indicators['MACD_hist'].reindex(data_indices), label='MACD柱状图')
        plt.title('MACD指标')
        plt.legend()

    if plot_volume:
        # 成交量
        plt.subplot(6, 1, 3)
        plt.bar(data_indices, df['volume'], label='成交量')
        plt.title(f'{title} 成交量')
        plt.legend()

    if plot_kdj:
        # KDJ
        plt.subplot(6, 1, 4)
        plt.plot(data_indices, df['KDJ_K'].reindex(data_indices), label='K值')
        plt.plot(data_indices, df['KDJ_D'].reindex(data_indices), label='D值')
        plt.plot(data_indices, df['KDJ_J'].reindex(data_indices), label='J值')
        plt.axhline(y=80, color='r', linestyle='--')
        plt.axhline(y=20, color='g', linestyle='--')
        plt.title('KDJ指标')
        plt.legend()

    if plot_rsi:
        # RSI
        plt.subplot(6, 1, 5)
        plt.plot(data_indices, indicators['RSI_6'].reindex(data_indices), label='6日RSI')
        plt.plot(data_indices, indicators['RSI_12'].reindex(data_indices), label='12日RSI')
        plt.axhline(y=70, color='r', linestyle='--')
        plt.axhline(y=30, color='g', linestyle='--')
        plt.title('相对强弱指数(RSI)')
        plt.legend()

    if plot_boll:
        # 布林带
        plt.subplot(6, 1, 6)
        plt.plot(data_indices, indicators['BB_upper'].reindex(data_indices), label='BOLL上轨')
        plt.plot(data_indices, indicators['BB_middle'].reindex(data_indices), label='BOLL中轨')
        plt.plot(data_indices, indicators['BB_lower'].reindex(data_indices), label='BOLL下轨')
        plt.title('布林带')
        plt.legend()
    
    plt.tight_layout()
        
    if save_path:
        plt.savefig(save_path)
        return True
    else:
        plt.show()
        return False