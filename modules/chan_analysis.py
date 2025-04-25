# -*- coding: utf-8 -*-
"""
缠论T+0训练系统 - 信号分析模块

该模块负责缠论训练系统的信号分析和检测，包括：
1. 底背驰和顶背驰检测
2. 买卖信号生成
3. 缠论分析结果处理
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

# 导入项目模块
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger(__name__)

class ChanAnalyzer:
    """
    缠论分析器类
    
    该类负责分析数据并生成交易信号。
    """
    
    def __init__(self, buy_threshold: float = 0.5, sell_threshold: float = 0.5):
        """
        初始化分析器
        
        参数:
            buy_threshold (float, 可选): 买入阈值，用于判断买入条件，默认0.5
            sell_threshold (float, 可选): 卖出阈值，用于判断卖出条件，默认0.5
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.logger = logger

    def analyze_current_data(self, data_handler, level: str) -> Dict[str, Any]:
        """
        分析当前级别数据
        
        参数:
            data_handler: 数据处理器实例
            level (str): 数据级别
            
        返回:
            Dict[str, Any]: 分析结果
        """
        self.logger.info(f"分析{data_handler.stock_code} {level}级别数据...")
        
        try:
            if level not in data_handler.level_data or data_handler.level_data[level].empty:
                self.logger.warning(f"{level}级别数据不存在或为空")
                return {}
            
            # 使用缠论分析器分析当前级别数据
            if data_handler.chan_analyzer:
                data_handler.chan_analyzer.level_data[level] = data_handler.level_data[level]
                result = data_handler.chan_analyzer.analyze_level(level)
                
                if not result:
                    self.logger.warning(f"分析{level}级别数据失败")
                    return {}
                
                self.logger.info(f"成功分析{level}级别数据")
                return result
            else:
                self.logger.warning("缠论分析器未初始化")
                return {}
            
        except Exception as e:
            self.logger.error(f"分析数据时出错: {str(e)}")
            return {}

    def detect_buying_signal(self, level_data: pd.DataFrame) -> Dict[str, Any]:
        """
        检测买入信号
        
        参数:
            level_data (pd.DataFrame): 级别数据
            
        返回:
            Dict[str, Any]: 买入信号信息
        """
        self.logger.info("检测买入信号...")
        
        try:
            if level_data.empty:
                return {'signal': False, 'reason': "数据为空"}
            
            # 复制数据，避免修改原始数据
            df = level_data.copy()
            
            # 1. 检测底背驰
            # 底背驰特征：价格创新低，但MACD指标不创新低
            df['price_new_low'] = False
            df['macd_not_new_low'] = False
            
            # 检查窗口大小
            window_size = 20
            if len(df) < window_size:
                window_size = len(df) // 2
            
            # 对最近的K线数据进行检测
            for i in range(window_size, len(df)):
                window = df.iloc[i-window_size:i+1]
                current_low = df.iloc[i]['low']
                current_macd = df.iloc[i]['macd']
                
                # 判断是否创新低
                if current_low <= window['low'].min():
                    df.iloc[i, df.columns.get_loc('price_new_low')] = True
                
                # 判断MACD是否不创新低（当价格创新低时）
                if df.iloc[i]['price_new_low']:
                    if current_macd > window['macd'].min():
                        df.iloc[i, df.columns.get_loc('macd_not_new_low')] = True
            
            # 最新的几根K线中是否有底背驰特征
            recent_bars = df.iloc[-5:]
            bottom_divergence = (recent_bars['price_new_low'] & recent_bars['macd_not_new_low']).any()
            
            # 2. 检测1类买点
            # 特征：强势向上突破，成交量放大
            volume_increase = df['volume'].iloc[-1] > df['volume'].iloc[-6:-1].mean() * 1.3
            price_up_trend = df['close'].iloc[-1] > df['close'].iloc[-6:-1].mean() * 1.02
            
            # 3. 检测MACD金叉
            # 特征：DIFF线从下向上穿越DEA线
            macd_golden_cross = (df['diff'].iloc[-2] < df['dea'].iloc[-2]) and (df['diff'].iloc[-1] > df['dea'].iloc[-1])
            
            # 综合判断
            if bottom_divergence:
                return {
                    'signal': True,
                    'type': 'bottom_divergence',
                    'reason': "底背驰信号：价格创新低，但MACD不创新低",
                    'strength': 0.8,
                    'price': df['close'].iloc[-1]
                }
            elif macd_golden_cross and volume_increase:
                return {
                    'signal': True,
                    'type': 'macd_golden_cross',
                    'reason': "MACD金叉 + 成交量放大",
                    'strength': 0.7,
                    'price': df['close'].iloc[-1]
                }
            elif price_up_trend and volume_increase:
                return {
                    'signal': True,
                    'type': 'volume_price_up',
                    'reason': "价格上涨 + 成交量放大",
                    'strength': 0.6,
                    'price': df['close'].iloc[-1]
                }
            else:
                return {'signal': False, 'reason': "未检测到买入信号"}
                
        except Exception as e:
            self.logger.error(f"检测买入信号时出错: {str(e)}")
            return {'signal': False, 'reason': f"检测出错: {str(e)}"}

    def detect_selling_signal(self, level_data: pd.DataFrame) -> Dict[str, Any]:
        """
        检测卖出信号
        
        参数:
            level_data (pd.DataFrame): 级别数据
            
        返回:
            Dict[str, Any]: 卖出信号信息
        """
        self.logger.info("检测卖出信号...")
        
        try:
            if level_data.empty:
                return {'signal': False, 'reason': "数据为空"}
            
            # 复制数据，避免修改原始数据
            df = level_data.copy()
            
            # 1. 检测顶背驰
            # 顶背驰特征：价格创新高，但MACD指标不创新高
            df['price_new_high'] = False
            df['macd_not_new_high'] = False
            
            # 检查窗口大小
            window_size = 20
            if len(df) < window_size:
                window_size = len(df) // 2
            
            # 对最近的K线数据进行检测
            for i in range(window_size, len(df)):
                window = df.iloc[i-window_size:i+1]
                current_high = df.iloc[i]['high']
                current_macd = df.iloc[i]['macd']
                
                # 判断是否创新高
                if current_high >= window['high'].max():
                    df.iloc[i, df.columns.get_loc('price_new_high')] = True
                
                # 判断MACD是否不创新高（当价格创新高时）
                if df.iloc[i]['price_new_high']:
                    if current_macd < window['macd'].max():
                        df.iloc[i, df.columns.get_loc('macd_not_new_high')] = True
            
            # 最新的几根K线中是否有顶背驰特征
            recent_bars = df.iloc[-5:]
            top_divergence = (recent_bars['price_new_high'] & recent_bars['macd_not_new_high']).any()
            
            # 2. 检测3类卖点
            # 特征：突破中枢后回落，未能再次突破
            price_down_trend = df['close'].iloc[-1] < df['close'].iloc[-6:-1].mean() * 0.98
            volume_decrease = df['volume'].iloc[-1] < df['volume'].iloc[-6:-1].mean() * 0.7
            
            # 3. 检测MACD死叉
            # 特征：DIFF线从上向下穿越DEA线
            macd_death_cross = (df['diff'].iloc[-2] > df['dea'].iloc[-2]) and (df['diff'].iloc[-1] < df['dea'].iloc[-1])
            
            # 综合判断
            if top_divergence:
                return {
                    'signal': True,
                    'type': 'top_divergence',
                    'reason': "顶背驰信号：价格创新高，但MACD不创新高",
                    'strength': 0.8,
                    'price': df['close'].iloc[-1]
                }
            elif macd_death_cross:
                return {
                    'signal': True,
                    'type': 'macd_death_cross',
                    'reason': "MACD死叉",
                    'strength': 0.7,
                    'price': df['close'].iloc[-1]
                }
            elif price_down_trend and volume_decrease:
                return {
                    'signal': True,
                    'type': 'volume_price_down',
                    'reason': "价格下跌 + 成交量萎缩",
                    'strength': 0.6,
                    'price': df['close'].iloc[-1]
                }
            else:
                return {'signal': False, 'reason': "未检测到卖出信号"}
                
        except Exception as e:
            self.logger.error(f"检测卖出信号时出错: {str(e)}")
            return {'signal': False, 'reason': f"检测出错: {str(e)}"}
    
    def should_buy(self, signal: Dict[str, Any]) -> bool:
        """
        判断是否应该买入
        
        参数:
            signal (Dict[str, Any]): 买入信号
            
        返回:
            bool: 是否应该买入
        """
        return signal.get('signal', False) and signal.get('strength', 0) >= self.buy_threshold
    
    def should_sell(self, signal: Dict[str, Any]) -> bool:
        """
        判断是否应该卖出
        
        参数:
            signal (Dict[str, Any]): 卖出信号
            
        返回:
            bool: 是否应该卖出
        """
        return signal.get('signal', False) and signal.get('strength', 0) >= self.sell_threshold
    
    def set_thresholds(self, buy_threshold: float, sell_threshold: float) -> None:
        """
        设置买卖阈值
        
        参数:
            buy_threshold (float): 买入阈值
            sell_threshold (float): 卖出阈值
        """
        self.buy_threshold = max(0, min(1, buy_threshold))
        self.sell_threshold = max(0, min(1, sell_threshold))
        self.logger.info(f"更新阈值: 买入={self.buy_threshold}, 卖出={self.sell_threshold}") 