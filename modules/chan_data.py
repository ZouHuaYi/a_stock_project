# -*- coding: utf-8 -*-
"""
缠论T+0训练系统 - 数据处理模块

该模块负责缠论训练系统的数据获取和处理，包括：
1. 获取历史数据
2. 获取实时数据
3. 数据预处理和计算指标
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

# 导入项目模块
from analyzer.chan_making_analyzer import ChanMakingAnalyzer
from utils.indicators import calculate_technical_indicators
from utils.akshare_api import AkshareAPI
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger(__name__)

class ChanDataHandler:
    """
    缠论数据处理类
    
    该类负责获取、处理和管理缠论T+0训练系统所需的数据。
    """
    
    def __init__(self, stock_code: str, stock_name: str = None, levels: List[str] = None):
        """
        初始化数据处理器
        
        参数:
            stock_code (str): 股票代码
            stock_name (str, 可选): 股票名称，不提供则自动获取
            levels (List[str], 可选): 要分析的周期级别，默认为["daily", "30min", "5min", "1min"]
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
        
        # 数据存储
        self.level_data = {}  # 各级别K线数据
        self.real_time_data = pd.DataFrame()  # 实时数据
        self.latest_price = 0.0  # 最新价格
        
        # 分析器
        self.chan_analyzer = None  # 缠论分析器实例
        
        # 内部日志
        self.logger = logger

    def get_historical_data(self, start_date: Union[str, datetime] = None, end_date: Union[str, datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        获取历史数据
        
        参数:
            start_date (str or datetime, 可选): 开始日期
            end_date (str or datetime, 可选): 结束日期
            
        返回:
            Dict[str, pd.DataFrame]: 各级别历史数据
        """
        self.logger.info(f"获取{self.stock_code} ({self.stock_name})的历史数据...")
        
        # 处理日期
        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_date = datetime.now() - timedelta(days=365)  # 默认获取一年数据
            
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
        
        try:
            # 使用缠论分析器获取历史数据
            days = (end_date - start_date).days + 1
            
            self.chan_analyzer = ChanMakingAnalyzer(
                stock_code=self.stock_code,
                stock_name=self.stock_name,
                levels=self.levels,
                end_date=end_date.strftime('%Y-%m-%d'),
                days=days
            )
            
            # 获取多级别数据
            self.level_data = self.chan_analyzer.get_multi_level_data()
            
            if not self.level_data:
                self.logger.error(f"获取{self.stock_code}历史数据失败")
                return {}
            
            for level, data in self.level_data.items():
                self.logger.info(f"成功获取{level}级别数据: {len(data)}条")
            
            return self.level_data
            
        except Exception as e:
            self.logger.error(f"获取历史数据时出错: {str(e)}")
            return {}

    def get_real_time_data(self, level: str) -> pd.DataFrame:
        """
        获取实时数据（模拟或实时）
        
        参数:
            level (str): 数据级别，如'1min', '5min'等
            
        返回:
            pd.DataFrame: 实时数据
        """
        self.logger.info(f"获取{self.stock_code} {level}级别实时数据...")
        
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
                self.logger.warning(f"不支持的级别: {level}")
                return pd.DataFrame()
            
            minute_data = self.akshare_api.get_stock_history_min(
                stock_code=self.stock_code,
                period=period,
                days=days
            )
            
            if minute_data.empty:
                self.logger.warning(f"获取{level}级别实时数据失败")
                return pd.DataFrame()
            
            # 计算技术指标
            minute_data, _ = calculate_technical_indicators(minute_data)
            
            self.real_time_data = minute_data
            self.latest_price = minute_data['close'].iloc[-1] if not minute_data.empty else 0.0
            
            self.logger.info(f"成功获取{level}级别实时数据: {len(minute_data)}条，最新价: {self.latest_price}")
            
            return minute_data
            
        except Exception as e:
            self.logger.error(f"获取实时数据时出错: {str(e)}")
            return pd.DataFrame()
            
    def update_data(self, level: str, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        更新数据，合并新数据到历史数据
        
        参数:
            level (str): 数据级别
            new_data (pd.DataFrame): 新数据
            
        返回:
            pd.DataFrame: 更新后的数据
        """
        if new_data.empty:
            return self.level_data.get(level, pd.DataFrame())
            
        # 如果该级别已有数据，则合并
        if level in self.level_data and not self.level_data[level].empty:
            # 找出新数据中不在历史数据中的部分
            existing_dates = set(self.level_data[level].index)
            unique_new_data = new_data[~new_data.index.isin(existing_dates)]
            
            if not unique_new_data.empty:
                self.level_data[level] = pd.concat([self.level_data[level], unique_new_data])
                self.level_data[level] = self.level_data[level].sort_index()
                self.logger.info(f"添加{len(unique_new_data)}条新数据到{level}级别历史数据")
        else:
            # 如果该级别没有数据，则直接使用新数据
            self.level_data[level] = new_data
            self.logger.info(f"初始化{level}级别历史数据: {len(new_data)}条")
            
        return self.level_data[level]
    
    def get_day_data(self, level: str, date: Union[str, datetime]) -> pd.DataFrame:
        """
        获取特定日期的数据
        
        参数:
            level (str): 数据级别
            date (str or datetime): 日期
            
        返回:
            pd.DataFrame: 该日期的数据
        """
        if level not in self.level_data or self.level_data[level].empty:
            return pd.DataFrame()
            
        # 将日期转为字符串格式
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date
            
        # 筛选该日期的数据
        day_data = self.level_data[level][self.level_data[level].index.astype(str).str.startswith(date_str)]
        
        return day_data
    
    def get_latest_price(self) -> float:
        """
        获取最新价格
        
        返回:
            float: 最新价格
        """
        return self.latest_price 