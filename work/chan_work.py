# -*- coding: utf-8 -*-
"""
缠论T+0训练系统

该模块实现了缠论T+0训练系统，用于通过模拟T+0交易进行缠论实践，主要功能包括：
1. 选择一只股票作为训练标的
2. 获取实时分钟级别行情数据
3. 基于缠论进行小级别（1分钟、5分钟）分析
4. 判断买卖点信号（特别是背驰信号）
5. 进行模拟T+0交易操作
6. 记录交易结果和分析
7. 提供复盘和训练功能

通过本模块可以：
- 对缠论在实时行情中的应用进行训练
- 提高小级别背驰判断能力
- 形成机械化的交易系统
- 在实盘前通过模拟训练降低风险
"""

import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import logging
import traceback

# 导入项目中的其他模块
from analyzer.chan_making_analyzer import ChanMakingAnalyzer
from utils.logger import get_logger, setup_logger
from utils.indicators import calculate_technical_indicators
from config import ANALYZER_CONFIG, PATH_CONFIG
from utils.akshare_api import AkshareAPI

# 创建日志记录器
logger = get_logger(__name__)

class ChanWorkTrainer:
    """缠论T+0训练系统类
    
    该类实现了基于缠论理论的T+0训练功能，专注于小级别（1分钟、5分钟）的分析和交易。
    主要用于提高用户对缠论理论的实践应用能力，特别是背驰判断和买卖点识别。
    """
    
    def __init__(self, stock_code: str, 
                 stock_name: str = None,
                 initial_capital: float = 100000.0,
                 levels: List[str] = None,
                 day_limit: int = 10,
                 focus_level: str = '1min'):
        """
        初始化缠论T+0训练系统
        
        参数:
            stock_code (str): 股票代码（6位数字，如'000001'）
            stock_name (str, 可选): 股票名称，不提供则自动获取
            initial_capital (float, 可选): 初始资金，默认为10万
            levels (List[str], 可选): 要分析的周期级别，默认为["daily", "30min", "5min", "1min"]
            day_limit (int, 可选): 模拟训练天数限制，默认为10天
            focus_level (str, 可选): 主要关注的级别，默认为1分钟
        """
        self.stock_code = stock_code
        # 格式化股票代码为6位数字
        if len(stock_code) > 6:
            self.stock_code = stock_code[-6:]
        
        # 尝试获取股票名称
        self.akshare_api = AkshareAPI()
        self.stock_name = stock_name if stock_name else self.akshare_api.get_stock_name(self.stock_code)
        
        # 设置分析周期级别
        self.levels = levels if levels else ["daily", "30min", "5min", "1min"]
        self.focus_level = focus_level
        
        # 模拟交易设置
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.current_position = 0
        self.position_price = 0.0
        self.day_limit = day_limit
        self.current_day = 0
        
        # 交易记录
        self.trade_history = []
        
        # 分析器和数据
        self.chan_analyzer = None
        self.level_data = {}
        self.real_time_data = pd.DataFrame()
        self.latest_price = 0.0
        
        # 交易统计
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
        # 交易规则参数
        self.buy_threshold = 0.5  # 买入阈值，用于判断买入条件
        self.sell_threshold = 0.5  # 卖出阈值，用于判断卖出条件
        self.stop_loss_rate = 0.02  # 止损比例，默认2%
        self.take_profit_rate = 0.03  # 止盈比例，默认3%
        
        # 保存路径
        self.base_save_path = os.path.join(PATH_CONFIG.get('output_dir'), 'chan_work', 
                                     datetime.now().strftime('%Y%m%d'), self.stock_code)
        os.makedirs(self.base_save_path, exist_ok=True)
        
        # 初始化日志
        self.setup_logger()
        
        logger.info(f"初始化缠论T+0训练系统，股票：{self.stock_code} ({self.stock_name})")
        logger.info(f"初始资金：{self.initial_capital}，分析周期：{', '.join(self.levels)}")
    
    def setup_logger(self) -> None:
        """设置专用日志记录器"""
        log_dir = os.path.join(self.base_save_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        log_file = os.path.join(log_dir, f'chan_work_{self.stock_code}.log')
        
        # 创建专用的文件处理器
        self.work_logger = get_logger(f'chan_work_{self.stock_code}')
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # 确保不会重复添加处理器
        for handler in self.work_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file:
                self.work_logger.removeHandler(handler)
                
        self.work_logger.addHandler(file_handler)
        
    def log_trade(self, action: str, price: float, quantity: int, 
                 reason: str, signal_type: str = None) -> None:
        """
        记录交易
        
        参数:
            action (str): 交易动作，'buy'或'sell'
            price (float): 交易价格
            quantity (int): 交易数量
            reason (str): 交易原因
            signal_type (str, 可选): 信号类型
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        trade_info = {
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'quantity': quantity,
            'value': price * quantity,
            'reason': reason,
            'signal_type': signal_type,
            'capital_after': self.current_capital,
            'position_after': self.current_position
        }
        
        self.trade_history.append(trade_info)
        
        # 记录到日志
        self.work_logger.info(
            f"交易: {action.upper()} {quantity}股 @ ¥{price:.2f}, "
            f"总值: ¥{price * quantity:.2f}, 原因: {reason}, "
            f"信号: {signal_type if signal_type else 'N/A'}, "
            f"资金: ¥{self.current_capital:.2f}, 持仓: {self.current_position}"
        )
    
    def get_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        获取历史数据
        
        返回:
            Dict[str, pd.DataFrame]: 各级别历史数据
        """
        self.work_logger.info(f"获取{self.stock_code} ({self.stock_name})的历史数据...")
        
        try:
            # 使用缠论分析器获取历史数据
            self.chan_analyzer = ChanMakingAnalyzer(
                stock_code=self.stock_code,
                stock_name=self.stock_name,
                levels=self.levels
            )
            
            # 获取多级别数据
            self.level_data = self.chan_analyzer.get_multi_level_data()
            
            if not self.level_data:
                self.work_logger.error(f"获取{self.stock_code}历史数据失败")
                return {}
            
            for level, data in self.level_data.items():
                self.work_logger.info(f"成功获取{level}级别数据: {len(data)}条")
            
            return self.level_data
            
        except Exception as e:
            self.work_logger.error(f"获取历史数据时出错: {str(e)}")
            return {}
    
    def get_real_time_data(self, level: str) -> pd.DataFrame:
        """
        获取实时数据（模拟）
        
        参数:
            level (str): 数据级别，如'1min', '5min'等
            
        返回:
            pd.DataFrame: 实时数据
        """
        self.work_logger.info(f"获取{self.stock_code} {level}级别实时数据...")
        
        try:
            # 这里使用AkShare获取最新分钟数据
            # 实际应用中可能需要使用实时行情API替代
            today = datetime.now().strftime('%Y-%m-%d')
            
            if level == '1min':
                period = '1'
                days = 1
            elif level == '5min':
                period = '5'
                days = 1
            elif level == '30min':
                period = '30'
                days = 3
            else:
                self.work_logger.warning(f"不支持的级别: {level}")
                return pd.DataFrame()
            
            minute_data = self.akshare_api.get_stock_history_min(
                stock_code=self.stock_code,
                period=period,
                days=days
            )
            
            if minute_data.empty:
                self.work_logger.warning(f"获取{level}级别实时数据失败")
                return pd.DataFrame()
            
            # 计算技术指标
            minute_data, _ = calculate_technical_indicators(minute_data)
            
            self.real_time_data = minute_data
            self.latest_price = minute_data['close'].iloc[-1] if not minute_data.empty else 0.0
            
            self.work_logger.info(f"成功获取{level}级别实时数据: {len(minute_data)}条，最新价: {self.latest_price}")
            
            return minute_data
            
        except Exception as e:
            self.work_logger.error(f"获取实时数据时出错: {str(e)}")
            return pd.DataFrame()
    
    def analyze_current_data(self, level: str) -> Dict[str, Any]:
        """
        分析当前级别数据
        
        参数:
            level (str): 数据级别，如'1min', '5min'等
            
        返回:
            Dict[str, Any]: 分析结果
        """
        self.work_logger.info(f"分析{self.stock_code} {level}级别数据...")
        
        try:
            if level not in self.level_data or self.level_data[level].empty:
                self.work_logger.warning(f"{level}级别数据不存在或为空")
                return {}
            
            # 使用缠论分析器分析当前级别数据
            self.chan_analyzer.level_data[level] = self.level_data[level]
            result = self.chan_analyzer.analyze_level(level)
            
            if not result:
                self.work_logger.warning(f"分析{level}级别数据失败")
                return {}
            
            self.work_logger.info(f"成功分析{level}级别数据")
            return result
            
        except Exception as e:
            self.work_logger.error(f"分析数据时出错: {str(e)}")
            return {}
    
    def detect_buying_signal(self, level_data: pd.DataFrame) -> Dict[str, Any]:
        """
        检测买入信号
        
        参数:
            level_data (pd.DataFrame): 特定级别的DataFrame数据，包含价格和指标数据
            
        返回:
            Dict[str, Any]: 包含信号信息的字典，包括:
                - signal (bool): 是否有买入信号
                - score (float): 信号强度分数 (0-1)
                - reason (str): 信号原因说明
                - type (str): 信号类型
        """
        # 默认无信号
        signal_result = {
            'signal': False,
            'score': 0,
            'reason': '',
            'type': ''
        }
        
        try:
            # 确保数据有至少30行
            if level_data is None or level_data.empty or len(level_data) < 30:
                return signal_result
            
            # 获取最近的数据 - 使用iloc获取最后10行数据
            recent_data = level_data.iloc[-10:].copy()
            last_row = recent_data.iloc[-1]
            prev_row = level_data.iloc[-2] if len(level_data) > 1 else None
            
            # 1. 检查MACD底背驰（MACD上穿零轴前）
            if all(col in level_data.columns for col in ['macd', 'diff', 'dea']):
                # 检查近期MACD走势
                last_macd = last_row['macd']
                last_diff = last_row['diff']
                last_dea = last_row['dea']
                
                # 检查是否有MACD金叉
                is_golden_cross = False
                for i in range(len(recent_data) - 1):
                    curr_diff = recent_data.iloc[i]['diff']
                    curr_dea = recent_data.iloc[i]['dea']
                    next_diff = recent_data.iloc[i+1]['diff']
                    next_dea = recent_data.iloc[i+1]['dea']
                    if curr_diff < curr_dea and next_diff > next_dea:
                        is_golden_cross = True
                        break
                
                if is_golden_cross:
                    signal_result['signal'] = True
                    signal_result['score'] += 0.3
                    signal_result['reason'] += "MACD金叉信号; "
                    signal_result['type'] = 'macd_golden_cross'
            
            # 2. 检查KDJ金叉
            if all(col in level_data.columns for col in ['k', 'd']):
                # 检查近期KDJ走势
                recent_k = recent_data['k'].values
                recent_d = recent_data['d'].values
                
                # 检查是否有KDJ金叉
                kdj_golden_cross = False
                for i in range(len(recent_k) - 1):
                    if recent_k[i] < recent_d[i] and recent_k[i+1] > recent_d[i+1]:
                        kdj_golden_cross = True
                        break
                
                # K值在低位（<30）金叉更有效
                if kdj_golden_cross and recent_k[-2] < 30:
                    signal_result['signal'] = True
                    signal_result['score'] += 0.3
                    signal_result['reason'] += "KDJ金叉信号; "
                    if 'type' not in signal_result or not signal_result['type']:
                        signal_result['type'] = 'kdj_golden_cross'
            
            # 3. 检查RSI低位转向
            if 'rsi' in level_data.columns:
                # 获取最近的RSI数据
                recent_rsi = recent_data['rsi'].values
                
                # RSI在低位（<30）并且开始向上
                if len(recent_rsi) >= 2 and recent_rsi[-1] < 30 and recent_rsi[-1] > recent_rsi[-2]:
                    signal_result['signal'] = True
                    signal_result['score'] += 0.2
                    signal_result['reason'] += "RSI低位回升信号; "
                    if 'type' not in signal_result or not signal_result['type']:
                        signal_result['type'] = 'rsi_low_turning'
            
            # 4. 检查均线支撑
            if all(col in level_data.columns for col in ['ma5', 'ma10', 'ma20']):
                last_close = last_row['close']
                last_ma5 = last_row['ma5']
                last_ma10 = last_row['ma10']
                last_ma20 = last_row['ma20']
                
                # 价格靠近均线或从均线下方上穿
                near_ma5 = abs(last_close - last_ma5) / last_ma5 < 0.01
                near_ma10 = abs(last_close - last_ma10) / last_ma10 < 0.01
                near_ma20 = abs(last_close - last_ma20) / last_ma20 < 0.01
                
                if near_ma5 or near_ma10 or near_ma20:
                    signal_result['signal'] = True
                    signal_result['score'] += 0.2
                    signal_result['reason'] += "均线支撑信号; "
                    if 'type' not in signal_result or not signal_result['type']:
                        signal_result['type'] = 'ma_support'
                        
            # 5. 检查成交量放大
            if 'volume_ratio' in level_data.columns:
                vol_ratio = last_row['volume_ratio']
                if vol_ratio > 1.5:  # 成交量明显放大
                    signal_result['score'] += 0.1
                    signal_result['reason'] += "成交量放大; "
                    
            return signal_result
            
        except Exception as e:
            self.work_logger.error(f"检测买入信号出错: {str(e)}")
            traceback.print_exc()
            return signal_result
    
    def detect_selling_signal(self, level_data: pd.DataFrame) -> Dict[str, Any]:
        """
        检测卖出信号
        
        参数:
            level_data (pd.DataFrame): 特定级别的DataFrame数据，包含价格和指标数据
            
        返回:
            Dict[str, Any]: 包含信号信息的字典，包括:
                - signal (bool): 是否有卖出信号
                - score (float): 信号强度分数 (0-1)
                - reason (str): 信号原因说明
                - type (str): 信号类型
        """
        # 默认无信号
        signal_result = {
            'signal': False,
            'score': 0,
            'reason': '',
            'type': ''
        }
        
        try:
            # 确保数据有至少30行
            if level_data is None or level_data.empty or len(level_data) < 30:
                return signal_result
            
            # 获取最近的数据 - 使用iloc获取最后10行数据
            recent_data = level_data.iloc[-10:].copy()
            last_row = recent_data.iloc[-1]
            prev_row = level_data.iloc[-2] if len(level_data) > 1 else None
            
            # 1. 检查MACD顶背驰（MACD死叉）
            if all(col in level_data.columns for col in ['macd', 'diff', 'dea']):
                # 检查近期MACD走势
                last_macd = last_row['macd']
                last_diff = last_row['diff']
                last_dea = last_row['dea']
                
                # 检查是否有MACD死叉
                is_death_cross = False
                for i in range(len(recent_data) - 1):
                    curr_diff = recent_data.iloc[i]['diff']
                    curr_dea = recent_data.iloc[i]['dea']
                    next_diff = recent_data.iloc[i+1]['diff']
                    next_dea = recent_data.iloc[i+1]['dea']
                    if curr_diff > curr_dea and next_diff < next_dea:
                        is_death_cross = True
                        break
                
                if is_death_cross:
                    signal_result['signal'] = True
                    signal_result['score'] += 0.3
                    signal_result['reason'] += "MACD死叉信号; "
                    signal_result['type'] = 'macd_death_cross'
            
            # 2. 检查KDJ死叉
            if all(col in level_data.columns for col in ['k', 'd']):
                # 检查近期KDJ走势
                recent_k = recent_data['k'].values
                recent_d = recent_data['d'].values
                
                # 检查是否有KDJ死叉
                kdj_death_cross = False
                for i in range(len(recent_k) - 1):
                    if recent_k[i] > recent_d[i] and recent_k[i+1] < recent_d[i+1]:
                        kdj_death_cross = True
                        break
                
                # K值在高位（>70）死叉更有效
                if kdj_death_cross and recent_k[-2] > 70:
                    signal_result['signal'] = True
                    signal_result['score'] += 0.3
                    signal_result['reason'] += "KDJ死叉信号; "
                    if 'type' not in signal_result or not signal_result['type']:
                        signal_result['type'] = 'kdj_death_cross'
            
            # 3. 检查RSI高位转向
            if 'rsi' in level_data.columns:
                # 获取最近的RSI数据
                recent_rsi = recent_data['rsi'].values
                
                # RSI在高位（>70）并且开始向下
                if len(recent_rsi) >= 2 and recent_rsi[-1] > 70 and recent_rsi[-1] < recent_rsi[-2]:
                    signal_result['signal'] = True
                    signal_result['score'] += 0.2
                    signal_result['reason'] += "RSI高位回落信号; "
                    if 'type' not in signal_result or not signal_result['type']:
                        signal_result['type'] = 'rsi_high_turning'
            
            # 4. 检查均线压力
            if all(col in level_data.columns for col in ['ma5', 'ma10', 'ma20']):
                last_close = last_row['close']
                last_ma5 = last_row['ma5']
                last_ma10 = last_row['ma10']
                last_ma20 = last_row['ma20']
                
                # 价格触及均线下方
                prev_close = prev_row['close'] if prev_row is not None else None
                cross_below_ma5 = prev_close is not None and last_close < last_ma5 and prev_close > level_data.iloc[-2]['ma5']
                cross_below_ma10 = prev_close is not None and last_close < last_ma10 and prev_close > level_data.iloc[-2]['ma10']
                cross_below_ma20 = prev_close is not None and last_close < last_ma20 and prev_close > level_data.iloc[-2]['ma20']
                
                if cross_below_ma5 or cross_below_ma10 or cross_below_ma20:
                    signal_result['signal'] = True
                    signal_result['score'] += 0.2
                    signal_result['reason'] += "均线压力信号; "
                    if 'type' not in signal_result or not signal_result['type']:
                        signal_result['type'] = 'ma_resistance'
                        
            # 5. 检查成交量萎缩
            if 'volume_ratio' in level_data.columns:
                vol_ratio = last_row['volume_ratio']
                if vol_ratio < 0.7:  # 成交量明显萎缩
                    signal_result['score'] += 0.1
                    signal_result['reason'] += "成交量萎缩; "
                    
            return signal_result
            
        except Exception as e:
            self.work_logger.error(f"检测卖出信号出错: {str(e)}")
            traceback.print_exc()
            return signal_result
    
    def execute_buy(self, price: float, reason: str, signal_type: str = None) -> bool:
        """
        执行买入操作
        
        参数:
            price (float): 买入价格
            reason (str): 买入原因
            signal_type (str, 可选): 信号类型
            
        返回:
            bool: 是否成功买入
        """
        # 检查当前资金是否足够
        if self.current_capital < price * 100:  # 最小买入单位100股
            self.work_logger.warning(f"资金不足，无法买入: 当前资金 {self.current_capital:.2f}, 需要 {price * 100:.2f}")
            return False
        
        # 计算可以买入的最大股数（必须是100的整数倍）
        max_shares = int(self.current_capital / price / 100) * 100
        if max_shares == 0:
            self.work_logger.warning(f"资金不足以买入最小单位(100股): 当前资金 {self.current_capital:.2f}")
            return False
        
        # 执行买入
        buy_shares = max_shares  # 默认全部买入
        buy_value = price * buy_shares
        
        # 更新资金和持仓
        self.current_capital -= buy_value
        self.current_position += buy_shares
        self.position_price = price
        
        # 记录交易
        self.log_trade('buy', price, buy_shares, reason, signal_type)
        
        # 更新统计信息
        self.total_trades += 1
        
        return True
    
    def execute_sell(self, price: float, reason: str, signal_type: str = None) -> bool:
        """
        执行卖出操作
        
        参数:
            price (float): 卖出价格
            reason (str): 卖出原因
            signal_type (str, 可选): 信号类型
            
        返回:
            bool: 是否成功卖出
        """
        # 检查当前持仓是否足够
        if self.current_position <= 0:
            self.work_logger.warning("当前无持仓，无法卖出")
            return False
        
        # 执行卖出
        sell_shares = self.current_position  # 默认全部卖出
        sell_value = price * sell_shares
        
        # 计算盈亏
        profit = (price - self.position_price) * sell_shares
        
        # 更新资金和持仓
        self.current_capital += sell_value
        self.current_position = 0
        
        # 更新统计信息
        if profit > 0:
            self.winning_trades += 1
            self.total_profit += profit
        else:
            self.losing_trades += 1
            self.total_loss += abs(profit)
        
        # 记录交易
        self.log_trade('sell', price, sell_shares, reason, signal_type)
        
        return True
    
    def check_stop_loss(self, current_price: float) -> bool:
        """
        检查止损条件
        
        参数:
            current_price (float): 当前价格
            
        返回:
            bool: 是否需要止损
        """
        if self.current_position <= 0 or self.position_price <= 0:
            return False
        
        # 计算当前亏损比例
        loss_rate = (self.position_price - current_price) / self.position_price
        
        # 如果亏损超过止损比例，执行止损
        if loss_rate >= self.stop_loss_rate:
            self.work_logger.info(f"触发止损: 买入价 {self.position_price:.2f}, 当前价 {current_price:.2f}, 亏损率 {loss_rate*100:.2f}%")
            return self.execute_sell(current_price, "止损", "stop_loss")
        
        return False
    
    def check_take_profit(self, current_price: float) -> bool:
        """
        检查止盈条件
        
        参数:
            current_price (float): 当前价格
            
        返回:
            bool: 是否需要止盈
        """
        if self.current_position <= 0 or self.position_price <= 0:
            return False
        
        # 计算当前盈利比例
        profit_rate = (current_price - self.position_price) / self.position_price
        
        # 如果盈利超过止盈比例，执行止盈
        if profit_rate >= self.take_profit_rate:
            self.work_logger.info(f"触发止盈: 买入价 {self.position_price:.2f}, 当前价 {current_price:.2f}, 盈利率 {profit_rate*100:.2f}%")
            return self.execute_sell(current_price, "止盈", "take_profit")
        
        return False
    
    def simulate_trading_day(self) -> Dict[str, Any]:
        """
        模拟一个交易日的T+0操作
        
        返回:
            Dict[str, Any]: 当日交易结果
        """
        self.current_day += 1
        self.work_logger.info(f"开始第{self.current_day}天的模拟交易...")
        
        # 更新数据
        self.get_real_time_data(self.focus_level)
        
        if self.real_time_data.empty:
            self.work_logger.error("获取实时数据失败，无法开始交易模拟")
            return {'success': False, 'message': "获取实时数据失败"}
        
        # 合并最新数据到历史数据中
        if self.focus_level in self.level_data:
            # 找出新数据中不在历史数据中的部分
            existing_dates = set(self.level_data[self.focus_level].index)
            new_data = self.real_time_data[~self.real_time_data.index.isin(existing_dates)]
            
            if not new_data.empty:
                self.level_data[self.focus_level] = pd.concat([self.level_data[self.focus_level], new_data])
                self.level_data[self.focus_level] = self.level_data[self.focus_level].sort_index()
                self.work_logger.info(f"添加{len(new_data)}条新数据到{self.focus_level}级别历史数据")
        else:
            self.level_data[self.focus_level] = self.real_time_data
            self.work_logger.info(f"初始化{self.focus_level}级别历史数据: {len(self.real_time_data)}条")
        
        # 模拟交易日内的操作
        # 实际交易中，可以每分钟或每5分钟执行一次
        # 这里简化为对当前所有数据进行一次性分析
        
        # 分析当前数据
        analysis_result = self.analyze_current_data(self.focus_level)
        
        # 记录交易前状态
        start_capital = self.current_capital
        start_position = self.current_position
        
        # 针对不同交易状态执行不同策略
        if self.current_position > 0:
            # 已有持仓，检查是否需要卖出
            # 1. 首先检查止损止盈
            if not self.check_stop_loss(self.latest_price) and not self.check_take_profit(self.latest_price):
                # 2. 如果没有触发止损止盈，检查卖出信号
                sell_signal = self.detect_selling_signal(self.level_data[self.focus_level])
                if sell_signal['signal'] and sell_signal['score'] >= self.sell_threshold:
                    self.execute_sell(
                        self.latest_price, 
                        sell_signal['reason'], 
                        sell_signal['type']
                    )
        else:
            # 无持仓，检查是否需要买入
            buy_signal = self.detect_buying_signal(self.level_data[self.focus_level])
            if buy_signal['signal'] and buy_signal['score'] >= self.buy_threshold:
                self.execute_buy(
                    self.latest_price, 
                    buy_signal['reason'], 
                    buy_signal['type']
                )
        
        # 计算当日交易结果
        day_profit = self.current_capital - start_capital
        if self.current_position > 0 and start_position == 0:
            day_position_value = self.current_position * self.latest_price
            day_result = "建仓"
        elif self.current_position == 0 and start_position > 0:
            day_position_value = 0
            day_result = "清仓"
        elif self.current_position > 0 and start_position > 0:
            day_position_value = self.current_position * self.latest_price
            day_result = "持仓"
        else:
            day_position_value = 0
            day_result = "空仓"
        
        total_asset = self.current_capital + day_position_value
        
        # 保存交易日结果
        day_trades = [t for t in self.trade_history if t['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))]
        
        result = {
            'day': self.current_day,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'success': True,
            'start_capital': start_capital,
            'end_capital': self.current_capital,
            'position': self.current_position,
            'position_price': self.position_price if self.current_position > 0 else 0.0,
            'position_value': day_position_value,
            'total_asset': total_asset,
            'day_profit': day_profit,
            'day_result': day_result,
            'trades_count': len(day_trades),
            'trades': day_trades
        }
        
        # 输出日结果
        self.work_logger.info(
            f"第{self.current_day}天交易结束: "
            f"资金 {self.current_capital:.2f}, "
            f"持仓 {self.current_position}股, "
            f"持仓价 {self.position_price if self.current_position > 0 else 0.0:.2f}, "
            f"总资产 {total_asset:.2f}, "
            f"日内交易 {len(day_trades)}笔, "
            f"状态: {day_result}"
        )
        
        # 保存交易日结果到文件
        self.save_daily_result(result)
        
        return result
    
    def save_daily_result(self, result: Dict[str, Any]) -> None:
        """
        保存每日交易结果
        
        参数:
            result (Dict[str, Any]): 交易结果
        """
        try:
            # 确保结果中包含必要的字段
            for field in ['date', 'day', 'start_capital', 'end_capital', 'day_profit', 'trades_count']:
                if field not in result:
                    self.work_logger.warning(f"交易结果缺少必要字段: {field}")
                    result[field] = "未知" if field in ['date'] else 0
                    
            # 添加持仓信息
            if 'start_position' not in result:
                result['start_position'] = 0
            if 'end_position' not in result:
                result['end_position'] = self.current_position
                
            if 'position_price' not in result and hasattr(self, 'position_price'):
                result['position_price'] = self.position_price
            elif 'position_price' not in result:
                result['position_price'] = 0.0
                
            if 'latest_price' not in result:
                result['latest_price'] = self.latest_price
                
            # 保存到 JSON 文件
            daily_result_dir = os.path.join(self.base_save_path, 'daily_results')
            os.makedirs(daily_result_dir, exist_ok=True)
            
            # 使用日期和序号作为文件名
            date_str = result['date'] if isinstance(result['date'], str) else result['date'].strftime('%Y%m%d')
            day = result['day']
            filename = f"day_{day}_{date_str}.json"
            
            filepath = os.path.join(daily_result_dir, filename)
            
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                # 需要处理不可序列化的对象，这里简单处理，实际应用可能需要更复杂的处理
                serializable_result = {}
                for key, value in result.items():
                    # 处理DataFrame等不可直接序列化的对象
                    if key == 'trades':
                        serializable_trades = []
                        for trade in value:
                            serializable_trade = {}
                            for k, v in trade.items():
                                if isinstance(v, pd.Timestamp) or isinstance(v, datetime):
                                    serializable_trade[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    serializable_trade[k] = v
                            serializable_trades.append(serializable_trade)
                        serializable_result[key] = serializable_trades
                    elif isinstance(value, pd.Timestamp) or isinstance(value, datetime):
                        serializable_result[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        serializable_result[key] = value
                
                json.dump(serializable_result, f, ensure_ascii=False, indent=2, default=str)
                
            self.work_logger.debug(f"交易结果已保存到 {filepath}")
        except Exception as e:
            self.work_logger.error(f"保存交易结果时出错: {str(e)}")
            import traceback
            self.work_logger.debug(traceback.format_exc())
    
    def plot_trade_chart(self, level: str = None, save_path: str = None) -> str:
        """
        绘制交易图表
        
        参数:
            level (str, 可选): 数据级别，默认为focus_level
            save_path (str, 可选): 保存路径，不提供则使用默认路径
            
        返回:
            str: 图表保存路径
        """
        if not level:
            level = self.focus_level
            
        if level not in self.level_data or self.level_data[level].empty:
            self.work_logger.warning(f"{level}级别数据不存在或为空，无法绘制图表")
            return ""
            
        try:
            # 设置matplotlib样式
            plt.style.use(ANALYZER_CONFIG.get('chart_style', 'seaborn-v0_8-darkgrid'))
            plt.rcParams['font.family'] = ANALYZER_CONFIG.get('font_family', 'SimHei')
            
            # 创建图形
            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, 
                figsize=ANALYZER_CONFIG.get('chart_figsize', (16, 12)),
                sharex=True,
                gridspec_kw={'height_ratios': [3, 1, 1]}
            )
            
            # 获取数据
            df = self.level_data[level].copy()
            
            # 获取交易记录
            buy_points = []
            sell_points = []
            
            for trade in self.trade_history:
                trade_time = datetime.strptime(trade['timestamp'], '%Y-%m-%d %H:%M:%S')
                if trade_time in df.index or trade_time.strftime('%Y-%m-%d') in df.index.astype(str):
                    if trade['action'] == 'buy':
                        buy_points.append((trade_time, trade['price'], trade['quantity']))
                    elif trade['action'] == 'sell':
                        sell_points.append((trade_time, trade['price'], trade['quantity']))
            
            # 绘制K线图
            if isinstance(df.index[0], str):
                # 如果索引是字符串，则转换为日期类型
                df.index = pd.to_datetime(df.index)
                
            # 获取x坐标
            x = np.arange(len(df))
            
            # 绘制K线
            candlestick_width = 0.6
            for i, (idx, row) in enumerate(df.iterrows()):
                color = 'red' if row['close'] >= row['open'] else 'green'
                # 绘制K线实体
                ax1.add_patch(Rectangle(
                    (i - candlestick_width/2, row['open']),
                    candlestick_width,
                    row['close'] - row['open'],
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.8
                ))
                # 绘制上下影线
                ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
            
            # 绘制均线
            if 'ma5' in df.columns:
                ax1.plot(x, df['ma5'], color='blue', linewidth=1, label='MA5')
            if 'ma10' in df.columns:
                ax1.plot(x, df['ma10'], color='orange', linewidth=1, label='MA10')
            if 'ma20' in df.columns:
                ax1.plot(x, df['ma20'], color='purple', linewidth=1, label='MA20')
            
            # 标记买卖点
            for buy_point in buy_points:
                time, price, quantity = buy_point
                if time in df.index:
                    idx = df.index.get_loc(time)
                    ax1.scatter(idx, price, color='red', marker='^', s=100, label='买入' if buy_point == buy_points[0] else "")
                    ax1.annotate(f"买入\n{quantity}股", (idx, price), xytext=(0, 20), 
                                textcoords='offset points', ha='center', va='bottom',
                                arrowprops=dict(arrowstyle='->', color='black'))
                elif time.strftime('%Y-%m-%d') in df.index.astype(str):
                    idx = df.index.astype(str).get_loc(time.strftime('%Y-%m-%d'))
                    ax1.scatter(idx, price, color='red', marker='^', s=100, label='买入' if buy_point == buy_points[0] else "")
                    ax1.annotate(f"买入\n{quantity}股", (idx, price), xytext=(0, 20), 
                                textcoords='offset points', ha='center', va='bottom',
                                arrowprops=dict(arrowstyle='->', color='black'))
            
            for sell_point in sell_points:
                time, price, quantity = sell_point
                if time in df.index:
                    idx = df.index.get_loc(time)
                    ax1.scatter(idx, price, color='green', marker='v', s=100, label='卖出' if sell_point == sell_points[0] else "")
                    ax1.annotate(f"卖出\n{quantity}股", (idx, price), xytext=(0, -20), 
                                textcoords='offset points', ha='center', va='top',
                                arrowprops=dict(arrowstyle='->', color='black'))
                elif time.strftime('%Y-%m-%d') in df.index.astype(str):
                    idx = df.index.astype(str).get_loc(time.strftime('%Y-%m-%d'))
                    ax1.scatter(idx, price, color='green', marker='v', s=100, label='卖出' if sell_point == sell_points[0] else "")
                    ax1.annotate(f"卖出\n{quantity}股", (idx, price), xytext=(0, -20), 
                                textcoords='offset points', ha='center', va='top',
                                arrowprops=dict(arrowstyle='->', color='black'))
            
            # 绘制成交量
            ax2.bar(x, df['volume'], color=['red' if row['close'] >= row['open'] else 'green' for _, row in df.iterrows()], alpha=0.7)
            ax2.set_ylabel('成交量')
            
            # 绘制MACD
            if all(col in df.columns for col in ['macd', 'diff', 'dea']):
                ax3.bar(x, df['macd'], color=['red' if x >= 0 else 'green' for x in df['macd']], alpha=0.7)
                ax3.plot(x, df['diff'], color='white', linewidth=1)
                ax3.plot(x, df['dea'], color='yellow', linewidth=1)
                ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
                ax3.set_ylabel('MACD')
            
            # 设置图表标题和标签
            fig.suptitle(f"{self.stock_code} {self.stock_name} - {level}级别缠论T+0训练图", fontsize=16)
            ax1.set_title(f"第{self.current_day}天 - 资金: ¥{self.current_capital:.2f}, 持仓: {self.current_position}股")
            ax1.set_ylabel('价格')
            ax1.legend(loc='upper left')
            ax1.grid(True)
            
            # 设置x轴标签
            plt.xticks(x[::len(x)//10], [d.strftime('%m-%d %H:%M') if isinstance(d, pd.Timestamp) else d for d in df.index[::len(x)//10]], rotation=45)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if not save_path:
                save_path = os.path.join(self.base_save_path, f"chart_{level}_day{self.current_day}.png")
                
            plt.savefig(save_path, dpi=ANALYZER_CONFIG.get('chart_dpi', 100))
            plt.close(fig)
            
            self.work_logger.info(f"交易图表已保存到: {save_path}")
            return save_path
            
        except Exception as e:
            self.work_logger.error(f"绘制交易图表时出错: {str(e)}")
            return ""

    def generate_summary(self) -> Dict[str, Any]:
        """
        生成训练总结
        
        返回:
            Dict[str, Any]: 训练总结
        """
        if self.total_trades == 0:
            win_rate = 0
        else:
            win_rate = self.winning_trades / self.total_trades * 100
            
        total_profit_loss = self.total_profit - self.total_loss
        if self.initial_capital == 0:
            profit_rate = 0
        else:
            profit_rate = total_profit_loss / self.initial_capital * 100
            
        total_asset = self.current_capital
        if self.current_position > 0:
            total_asset += self.current_position * self.latest_price
            
        summary = {
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'training_days': self.current_day,
            'focus_level': self.focus_level,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'current_position': self.current_position,
            'position_price': self.position_price if self.current_position > 0 else 0.0,
            'latest_price': self.latest_price,
            'total_asset': total_asset,
            'total_profit_loss': total_profit_loss,
            'profit_rate': profit_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss
        }
        
        # 记录总结
        self.work_logger.info("\n========== 训练总结 ==========")
        self.work_logger.info(f"股票: {self.stock_code} ({self.stock_name})")
        self.work_logger.info(f"训练周期: {self.current_day}天, 主要级别: {self.focus_level}")
        self.work_logger.info(f"初始资金: ¥{self.initial_capital:.2f}, 最终资金: ¥{self.current_capital:.2f}")
        self.work_logger.info(f"当前持仓: {self.current_position}股, 持仓价格: ¥{self.position_price if self.current_position > 0 else 0.0:.2f}")
        self.work_logger.info(f"最新价格: ¥{self.latest_price:.2f}, 总资产: ¥{total_asset:.2f}")
        self.work_logger.info(f"总盈亏: ¥{total_profit_loss:.2f} ({profit_rate:.2f}%)")
        self.work_logger.info(f"总交易次数: {self.total_trades}次, 盈利: {self.winning_trades}次, 亏损: {self.losing_trades}次, 胜率: {win_rate:.2f}%")
        self.work_logger.info(f"总盈利: ¥{self.total_profit:.2f}, 总亏损: ¥{self.total_loss:.2f}")
        self.work_logger.info("==============================\n")
        
        # 保存总结到文件
        try:
            summary_path = os.path.join(self.base_save_path, 'training_summary.json')
            import json
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=4)
            self.work_logger.info(f"训练总结已保存到: {summary_path}")
        except Exception as e:
            self.work_logger.error(f"保存训练总结时出错: {str(e)}")
            
        return summary

    def run_training(self) -> Dict[str, Any]:
        """
        运行T+0训练
        
        返回:
            Dict[str, Any]: 训练结果
        """
        self.work_logger.info(f"开始缠论T+0训练，股票：{self.stock_code} ({self.stock_name})")
        
        # 获取历史数据
        if not self.get_historical_data():
            self.work_logger.error("获取历史数据失败，无法开始训练")
            return {'success': False, 'message': "获取历史数据失败"}
        
        # 分析历史数据
        for level in self.levels:
            if level in self.level_data:
                self.analyze_current_data(level)
        
        try:
            # 执行每日模拟交易
            for day in range(self.day_limit):
                # 模拟交易日
                day_result = self.simulate_trading_day()
                
                # 如果模拟失败，则停止训练
                if not day_result.get('success', False):
                    self.work_logger.error(f"第{self.current_day}天交易模拟失败，停止训练")
                    break
                
                # 绘制当日交易图表
                self.plot_trade_chart()
                
                # 暂停一下，模拟实际间隔
                time.sleep(1)
                
                # 可以加入用户交互环节，例如让用户分析当前信号或决定是否继续
                
            # 生成训练总结
            summary = self.generate_summary()
            
            return {
                'success': True,
                'days': self.current_day,
                'summary': summary
            }
            
        except Exception as e:
            self.work_logger.error(f"训练过程中出错: {str(e)}")
            return {
                'success': False,
                'message': f"训练出错: {str(e)}",
                'days': self.current_day
            }

    def run_interactive_training(self) -> Dict[str, Any]:
        """
        运行交互式T+0训练，每天询问用户是否继续
        
        返回:
            Dict[str, Any]: 训练结果
        """
        self.work_logger.info(f"开始交互式缠论T+0训练，股票：{self.stock_code} ({self.stock_name})")
        
        # 获取历史数据
        if not self.get_historical_data():
            self.work_logger.error("获取历史数据失败，无法开始训练")
            return {'success': False, 'message': "获取历史数据失败"}
        
        # 分析历史数据
        for level in self.levels:
            if level in self.level_data:
                self.analyze_current_data(level)
        
        try:
            # 执行每日模拟交易，交互模式
            while self.current_day < self.day_limit:
                # 模拟交易日
                day_result = self.simulate_trading_day()
                
                # 如果模拟失败，则停止训练
                if not day_result.get('success', False):
                    self.work_logger.error(f"第{self.current_day}天交易模拟失败，停止训练")
                    break
                
                # 绘制当日交易图表
                chart_path = self.plot_trade_chart()
                
                # 交互部分：让用户决定是否继续、修改参数等
                print(f"\n========== 第{self.current_day}天交易结果 ==========")
                print(f"资金: ¥{self.current_capital:.2f}, 持仓: {self.current_position}股, 持仓价: ¥{self.position_price if self.current_position > 0 else 0.0:.2f}")
                print(f"日内交易: {day_result['trades_count']}笔, 结果: {day_result['day_result']}")
                print(f"交易图表已保存到: {chart_path}")
                
                choice = input("\n请选择操作：\n1. 继续下一天\n2. 调整交易参数\n3. 复盘当日交易\n4. 结束训练\n请输入 (1-4): ")
                
                if choice == "2":
                    # 调整参数
                    try:
                        print("\n当前参数:")
                        print(f"买入阈值: {self.buy_threshold}, 卖出阈值: {self.sell_threshold}")
                        print(f"止损比例: {self.stop_loss_rate*100}%, 止盈比例: {self.take_profit_rate*100}%")
                        
                        new_buy = float(input(f"新买入阈值 (0-1, 当前{self.buy_threshold}): ") or self.buy_threshold)
                        new_sell = float(input(f"新卖出阈值 (0-1, 当前{self.sell_threshold}): ") or self.sell_threshold)
                        new_stop_loss = float(input(f"新止损比例 (%, 当前{self.stop_loss_rate*100}): ") or self.stop_loss_rate*100) / 100
                        new_take_profit = float(input(f"新止盈比例 (%, 当前{self.take_profit_rate*100}): ") or self.take_profit_rate*100) / 100
                        
                        self.buy_threshold = max(0, min(1, new_buy))
                        self.sell_threshold = max(0, min(1, new_sell))
                        self.stop_loss_rate = max(0, min(0.5, new_stop_loss))
                        self.take_profit_rate = max(0, min(0.5, new_take_profit))
                        
                        self.work_logger.info(f"交易参数已更新: 买入阈值={self.buy_threshold}, 卖出阈值={self.sell_threshold}, "
                                           f"止损比例={self.stop_loss_rate*100}%, 止盈比例={self.take_profit_rate*100}%")
                    except ValueError:
                        print("输入无效，参数未更改")
                        
                elif choice == "3":
                    # 复盘当日交易
                    if day_result['trades']:
                        print("\n当日交易详情:")
                        for i, trade in enumerate(day_result['trades'], 1):
                            print(f"交易 {i}: {trade['action'].upper()} {trade['quantity']}股 @ ¥{trade['price']:.2f}, "
                                 f"原因: {trade['reason']}, 信号: {trade['signal_type'] if trade['signal_type'] else 'N/A'}")
                        
                        # 显示分析结果
                        if self.focus_level in self.level_data and not self.level_data[self.focus_level].empty:
                            # 打印MACD值
                            recent_data = self.level_data[self.focus_level].iloc[-5:]
                            print("\n最近的MACD值:")
                            for idx, row in recent_data.iterrows():
                                print(f"{idx}: MACD={row.get('macd', 'N/A'):.4f}, DIFF={row.get('diff', 'N/A'):.4f}, DEA={row.get('dea', 'N/A'):.4f}, "
                                     f"收盘价={row.get('close', 'N/A'):.2f}")
                    else:
                        print("当日无交易记录")
                        
                elif choice == "4":
                    # 结束训练
                    break
                
                # 否则继续下一天
                
            # 生成训练总结
            summary = self.generate_summary()
            
            return {
                'success': True,
                'days': self.current_day,
                'summary': summary
            }
            
        except Exception as e:
            self.work_logger.error(f"交互式训练过程中出错: {str(e)}")
            return {
                'success': False,
                'message': f"训练出错: {str(e)}",
                'days': self.current_day
            }

    def run_backtest(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        运行回测模式，使用历史数据评估策略表现
        
        参数:
            start_date (str, 可选): 回测开始日期，格式YYYY-MM-DD
            end_date (str, 可选): 回测结束日期，格式YYYY-MM-DD
            
        返回:
            Dict[str, Any]: 回测结果
        """
        self.work_logger.info(f"开始缠论T+0回测模式，股票：{self.stock_code} ({self.stock_name})")
        self.work_logger.info(f"回测区间: {start_date or '默认'} 至 {end_date or '默认'}")
        
        # 获取历史数据，指定日期范围
        try:
            # 使用缠论分析器获取历史数据，添加日期范围
            self.chan_analyzer = ChanMakingAnalyzer(
                stock_code=self.stock_code,
                stock_name=self.stock_name,
                levels=self.levels,
                start_date=start_date,
                end_date=end_date
            )
            
            # 获取多级别数据
            self.level_data = self.chan_analyzer.get_multi_level_data()
            
            if not self.level_data:
                self.work_logger.error(f"获取{self.stock_code}回测历史数据失败")
                return {'success': False, 'message': "获取历史数据失败"}
            
            for level, data in self.level_data.items():
                self.work_logger.info(f"成功获取{level}级别回测数据: {len(data)}条")
                
            # 分析历史数据
            for level in self.levels:
                if level in self.level_data:
                    self.work_logger.info(f"分析{self.stock_code} {level}级别数据...")
                    result = self.analyze_current_data(level)
                    if result:
                        self.work_logger.info(f"成功分析{level}级别数据")
                    else:
                        self.work_logger.warning(f"分析{level}级别数据失败")
            
            # 执行回测
            self.work_logger.info("开始执行回测...")
            
            # 获取交易日期列表
            trading_dates = []
            if 'daily' in self.level_data and not self.level_data['daily'].empty:
                trading_dates = self.level_data['daily'].index.unique().tolist()
                self.work_logger.info(f"回测期间共有{len(trading_dates)}个交易日")
            else:
                self.work_logger.error("无法获取回测交易日期")
                return {'success': False, 'message': "无法获取回测交易日期"}
            
            # 限制回测天数
            max_days = min(len(trading_dates), self.day_limit)
            trading_dates = trading_dates[-max_days:]
            self.work_logger.info(f"实际回测{len(trading_dates)}个交易日")
            
            # 创建回测专用目录
            backtest_dir = os.path.join(self.base_save_path, 'backtest')
            os.makedirs(backtest_dir, exist_ok=True)
            
            # 执行每日回测
            daily_results = []
            for date in trading_dates:
                self.current_day += 1
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                self.work_logger.info(f"回测第{self.current_day}天: {date_str}")
                
                # 准备当日数据（模拟实时数据）
                date_data = {}
                for level, data in self.level_data.items():
                    if level != 'daily':  # 分钟级别数据按日期过滤
                        # 处理日期过滤
                        try:
                            if isinstance(date, str):
                                date_obj = datetime.strptime(date, '%Y-%m-%d')
                            else:
                                date_obj = date
                                
                            # 检查索引类型并适当处理
                            filtered_data = pd.DataFrame()
                            if hasattr(data.index, 'date'):
                                # 如果索引是DatetimeIndex
                                filtered_data = data[data.index.date == date_obj.date()]
                            else:
                                # 如果是其他类型的索引，尝试根据日期列过滤
                                date_str_check = date_obj.strftime('%Y-%m-%d')
                                if 'trade_date' in data.columns:
                                    filtered_data = data[data['trade_date'].astype(str).str.startswith(date_str_check)]
                                elif 'date' in data.columns:
                                    filtered_data = data[data['date'].astype(str).str.startswith(date_str_check)]
                                elif 'datetime' in data.columns:
                                    filtered_data = data[data['datetime'].astype(str).str.startswith(date_str_check)]
                                else:
                                    # 无法过滤，记录警告
                                    self.work_logger.warning(f"无法根据日期过滤{level}级别数据，可能导致回测不准确")
                                    filtered_data = data  # 使用全部数据
                                    
                            date_data[level] = filtered_data
                            self.work_logger.info(f"{level}级别当日数据: {len(date_data[level])}条")
                        except Exception as e:
                            self.work_logger.error(f"过滤{level}级别数据出错: {str(e)}")
                            date_data[level] = pd.DataFrame()  # 使用空数据框
                    else:  # 日线级别数据保持不变
                        date_data[level] = data
                
                # 如果没有当日分钟数据，跳过这一天
                if all(len(data) == 0 for level, data in date_data.items() if level != 'daily'):
                    self.work_logger.warning(f"{date_str}无交易数据，跳过")
                    continue
                
                # 模拟当天交易
                day_result = self.backtest_trading_day(date_data, date_str)
                
                if day_result.get('success', False):
                    daily_results.append(day_result)
                    # 保存当日结果
                    self.save_daily_result(day_result)
                    # 绘制当日交易图表
                    chart_path = self.plot_trade_chart(
                        level=self.focus_level, 
                        save_path=os.path.join(backtest_dir, f"day_{self.current_day}_{date_str}.png")
                    )
                    self.work_logger.info(f"回测图表已保存至: {chart_path}")
                
            # 生成回测总结
            summary = self.generate_summary()
            summary['backtest'] = True
            summary['backtest_start'] = start_date or trading_dates[0]
            summary['backtest_end'] = end_date or trading_dates[-1]
            
            # 保存回测结果
            summary_file = os.path.join(backtest_dir, f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            import json
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            
            self.work_logger.info(f"回测结果已保存至: {summary_file}")
            
            return {
                'success': True,
                'days': self.current_day,
                'summary': summary,
                'summary_file': summary_file
            }
            
        except Exception as e:
            self.work_logger.error(f"回测过程中出错: {str(e)}")
            import traceback
            self.work_logger.error(traceback.format_exc())
            return {
                'success': False,
                'message': f"回测出错: {str(e)}",
                'days': self.current_day
            }
    
    def backtest_trading_day(self, date_data: Dict[str, pd.DataFrame], date_str: str) -> Dict[str, Any]:
        """
        针对指定日期进行回测
        
        参数:
            date_data (Dict[str, pd.DataFrame]): 按级别组织的DataFrame数据 
            date_str (str): 当前回测日期
        
        返回:
            Dict: 回测结果
        """
        self.work_logger.info(f"开始回测交易日: {date_str}")
        
        try:
            # 初始化结果
            result = {
                'date': date_str,
                'day': 1,  # 单日回测
                'start_capital': self.initial_capital,
                'end_capital': self.initial_capital,
                'day_profit': 0,
                'trades_count': 0,
                'trades': [],
                'start_position': 0,
                'end_position': 0,
                'position_price': 0,
                'latest_price': 0
            }
            
            # 获取5分钟级别数据
            minute_data = date_data.get(self.focus_level, pd.DataFrame())
            if minute_data.empty:
                self.work_logger.warning(f"无效的{self.focus_level}级别数据")
                return result
                
            # 确保技术指标已计算
            from utils.indicators import calculate_technical_indicators
            minute_data, _ = calculate_technical_indicators(minute_data)
            
            # 重置latest_price避免受到之前的价格影响
            latest_price = 0
            position = 0
            position_price = 0
            
            # 遍历日内数据
            for idx in minute_data.index:
                # 使用loc获取当前行数据
                current_data = minute_data.loc[idx]
                latest_price = current_data['close']
                
                # 当前时间之前的所有数据，用于信号检测
                current_view = minute_data.loc[:idx]
                
                # 买入信号检测
                if position == 0:  # 如果当前没有持仓
                    # 检查是否有买入信号
                    buy_signal = self.detect_buying_signal(current_view)
                    if buy_signal.get('signal', False) and buy_signal.get('score', 0) >= self.buy_threshold:
                        # 模拟买入
                        position = self.execute_buy(latest_price, buy_signal.get('reason', ''), buy_signal.get('type', ''))
                        position_price = latest_price
                        
                        # 记录交易
                        trade = {
                            'time': str(idx),
                            'action': 'buy',
                            'price': latest_price,
                            'amount': position,
                        }
                        result['trades'].append(trade)
                        result['trades_count'] += 1
                        self.work_logger.info(f"买入信号触发: 时间={idx}, 价格={latest_price}, 数量={position}")
                else:  # 如果已有持仓
                    # 检查是否有卖出信号
                    sell_signal = self.detect_selling_signal(current_view)
                    if sell_signal.get('signal', False) and sell_signal.get('score', 0) >= self.sell_threshold:
                        # 计算利润
                        profit = position * (latest_price - position_price)
                        self.current_capital += profit
                        
                        # 记录交易
                        trade = {
                            'time': str(idx),
                            'action': 'sell',
                            'price': latest_price,
                            'amount': position,
                            'profit': profit
                        }
                        result['trades'].append(trade)
                        result['trades_count'] += 1
                        self.work_logger.info(f"卖出信号触发: 时间={idx}, 价格={latest_price}, 数量={position}, 利润={profit}")
                        
                        # 重置持仓
                        position = 0
                        position_price = 0
            
            # 更新结果
            result['end_capital'] = self.current_capital
            result['day_profit'] = result['end_capital'] - result['start_capital']
            result['end_position'] = position
            result['position_price'] = position_price
            result['latest_price'] = latest_price
            
            return result
            
        except Exception as e:
            self.work_logger.error(f"回测交易日出错: {str(e)}")
            traceback.print_exc()
            # 返回默认结果
            return {
                'date': date_str,
                'day': 1,
                'start_capital': self.initial_capital,
                'end_capital': self.initial_capital,
                'day_profit': 0,
                'trades_count': 0,
                'trades': [],
                'start_position': 0,
                'end_position': 0,
                'position_price': 0,
                'latest_price': 0
            }

    def should_buy(self, signal: Dict[str, Any]) -> bool:
        """
        判断是否应该买入
        
        参数:
            signal (Dict[str, Any]): 买入信号
            
        返回:
            bool: 是否应该买入
        """
        return signal.get('signal', False) and signal.get('score', 0) >= self.buy_threshold
    
    def should_sell(self, signal: Dict[str, Any]) -> bool:
        """
        判断是否应该卖出
        
        参数:
            signal (Dict[str, Any]): 卖出信号
            
        返回:
            bool: 是否应该卖出
        """
        return signal.get('signal', False) and signal.get('score', 0) >= self.sell_threshold


# 命令行运行函数
def run_chan_work_command(args=None):
    """
    命令行入口函数
    
    参数:
        args (Namespace, 可选): 解析后的命令行参数
    """
    import argparse
    
    # 如果没有提供参数，则解析命令行
    if args is None:
        parser = argparse.ArgumentParser(description='缠论T+0训练系统')
        parser.add_argument('stock_code', help='股票代码，如000001、600001')
        parser.add_argument('--initial-capital', type=float, default=100000.0, help='初始资金，默认10万')
        parser.add_argument('--days', type=int, default=10, help='训练天数，默认10天')
        parser.add_argument('--focus-level', choices=['1min', '5min', '30min'], default='1min', help='主要关注级别')
        parser.add_argument('--interactive', action='store_true', help='是否进行交互式训练')
        parser.add_argument('--backtest', action='store_true', help='启用回测模式，使用历史数据进行策略回测')
        parser.add_argument('--start-date', help='回测开始日期，格式：YYYY-MM-DD，默认为最近交易日前30天')
        parser.add_argument('--end-date', help='回测结束日期，格式：YYYY-MM-DD，默认为最近交易日')
        
        args = parser.parse_args()
    
    # 股票代码处理
    stock_code = args.stock_code
    if len(stock_code) > 6:
        stock_code = stock_code[-6:]
    
    # 初始化训练器
    trainer = ChanWorkTrainer(
        stock_code=stock_code,
        initial_capital=args.initial_capital,
        day_limit=args.days,
        focus_level=args.focus_level
    )
    
    # 根据参数决定训练模式
    if args.backtest:
        print(f"开始缠论T+0回测: {stock_code} ({trainer.stock_name})")
        print(f"回测区间: {args.start_date or '默认'} 至 {args.end_date or '默认'}")
        result = trainer.run_backtest(start_date=args.start_date, end_date=args.end_date)
    elif args.interactive:
        print(f"开始交互式缠论T+0训练: {stock_code} ({trainer.stock_name})")
        result = trainer.run_interactive_training()
    else:
        print(f"开始自动缠论T+0训练: {stock_code} ({trainer.stock_name})")
        result = trainer.run_training()
    
    # 输出训练结果
    if result.get('success', False):
        summary = result.get('summary', {})
        print("\n========== 训练总结 ==========")
        print(f"股票: {summary.get('stock_code')} ({summary.get('stock_name')})")
        
        # 如果是回测，显示回测区间
        if args.backtest:
            print(f"回测区间: {summary.get('backtest_start')} 至 {summary.get('backtest_end')}")
            
        print(f"训练天数: {summary.get('training_days')}天")
        print(f"初始资金: ¥{summary.get('initial_capital', 0):.2f}, 最终资金: ¥{summary.get('final_capital', 0):.2f}")
        print(f"总盈亏: ¥{summary.get('total_profit_loss', 0):.2f} ({summary.get('profit_rate', 0):.2f}%)")
        print(f"交易次数: {summary.get('total_trades', 0)}次, 胜率: {summary.get('win_rate', 0):.2f}%")
        print(f"总盈利: ¥{summary.get('total_profit', 0):.2f}, 总亏损: ¥{summary.get('total_loss', 0):.2f}")
        
        # 如果有指定输出文件，显示文件名
        if result.get('summary_file'):
            print(f"详细结果已保存至: {result.get('summary_file')}")
        else:
            print("训练完成，结果已保存")
    else:
        print(f"训练/回测失败: {result.get('message', '未知错误')}")


# 如果是直接运行此文件，则执行命令行入口函数
if __name__ == "__main__":
    run_chan_work_command()
