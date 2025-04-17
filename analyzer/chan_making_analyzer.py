# -*- coding: utf-8 -*-
"""缠论分析器模块

该模块实现了基于缠论理论的多级别股票分析方法，包括：
1. 识别笔和线段
2. 确定中枢位置和级别
3. 进行多级别联立分析
4. 识别买卖点信号
5. 生成决策依据
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# 导入基类和工具
from analyzer.base_analyzer import BaseAnalyzer
from utils.logger import get_logger
from utils.indicators import calculate_technical_indicators
from config import ANALYZER_CONFIG, PATH_CONFIG

# 创建日志记录器
logger = get_logger(__name__)

class ChanMakingAnalyzer(BaseAnalyzer):
    """缠论分析器类，实现多级别缠论理论分析

    该类实现了:
    1. 笔和线段的自动识别
    2. 中枢形成与确认
    3. 买卖点信号识别
    4. 多级别联立分析
    5. 背驰和共振判断
    """
    
    def __init__(self, stock_code: str, stock_name: str = None, end_date: Union[str, datetime] = None, 
                 days: int = None, levels: List[str] = None):
        """
        初始化缠论分析器
        
        参数:
            stock_code (str): 股票代码
            stock_name (str, 可选): 股票名称
            end_date (str 或 datetime, 可选): 结束日期
            days (int, 可选): 回溯天数
            levels (List[str], 可选): 要分析的周期级别，默认为["daily", "30min", "5min", "1min"]
        """
        super().__init__(stock_code, stock_name, end_date, days)
        
        # 设置分析周期级别
        self.levels = levels if levels else ["daily", "30min", "5min", "1min"]
        
        # 各级别数据字典
        self.level_data = {}
        
        # 分析结果字典
        self.bi_data = {}  # 笔数据
        self.xianduan_data = {}  # 线段数据
        self.zhongshu_data = {}  # 中枢数据
        self.signals = {}  # 买卖点信号
        self.trend_matrix = pd.DataFrame()  # 趋势状态矩阵
        
        # 设置图表保存路径
        self.chart_path = os.path.join(PATH_CONFIG.get('output_path', './output'), 'chan')
        os.makedirs(self.chart_path, exist_ok=True)
    
    def get_multi_level_data(self) -> Dict[str, pd.DataFrame]:
        """
        获取多级别K线数据
        
        返回:
            Dict[str, pd.DataFrame]: 各级别数据字典
        """
        logger.info(f"正在获取{self.stock_code}的多级别数据...")
        
        try:
            from utils.akshare_api import AkshareAPI
            
            akshare = AkshareAPI()
            
            # 获取日线数据
            if "daily" in self.levels:
                daily_data = self.get_stock_daily_data(period="daily")
                if not daily_data.empty:
                    self.level_data["daily"] = daily_data
                    logger.info(f"成功获取{self.stock_code}日线数据：{len(daily_data)}条")
                else:
                    logger.warning(f"获取{self.stock_code}日线数据失败")
            
            # 获取不同分钟级别的数据
            minute_levels = [level for level in self.levels if level != "daily"]
            
            if minute_levels:
                logger.info(f"正在获取分钟级别数据: {', '.join(minute_levels)}")
                
                # 这里需要根据实际数据源情况获取分钟级别数据
                # 目前示例方法，实际应用中可能需要修改为实际数据源API
                for level in minute_levels:
                    try:
                        # 示例：使用AkShare获取分钟级别数据
                        # 一天有 8 个 30 分钟
                        # 一天有 48 个 5 分钟
                        # 一天有 240 个 1 分钟
                        # 实际实现可能有所不同，取决于数据源的API
                        # 计算每个级别需要的天数，使k线数量相等
                        # 假设需要240根k线
                        # 30分钟: 240 / (8根/天) = 30天
                        # 5分钟: 240 / (48根/天) = 5天
                        # 1分钟: 240 / (240根/天) = 1天

                        if level == "30min":
                            minute_data = akshare.get_stock_history_min(
                                stock_code=self.stock_code,
                                period="30",  # 期间可能需要调整为实际API支持的参
                                days=30
                            )
                        elif level == "5min":
                            minute_data = akshare.get_stock_history_min(
                                stock_code=self.stock_code,
                                period="5",
                                days=5
                            )
                        elif level == "1min":
                            minute_data = akshare.get_stock_history_min(
                                stock_code=self.stock_code,
                                period="1",
                                days=1
                            )
                        else:
                            minute_data = pd.DataFrame()
                            
                        if not minute_data.empty:
                            self.level_data[level] = minute_data
                            logger.info(f"成功获取{self.stock_code} {level}级别数据：{len(minute_data)}条")
                        else:
                            logger.warning(f"获取{self.stock_code} {level}级别数据失败")
                    except Exception as e:
                        logger.error(f"获取{level}级别数据出错: {str(e)}")
                        # 如果无法获取分钟数据，可以尝试从日线数据重采样（仅作示例）
                        if "daily" in self.level_data:
                            logger.info(f"尝试从日线重采样生成{level}数据...")
                            self.level_data[level] = self._resample_from_daily(self.level_data["daily"], level)
                            logger.info(f"通过重采样生成{self.stock_code} {level}级别数据：{len(self.level_data[level])}条")
            
            return self.level_data
                
        except Exception as e:
            logger.error(f"获取多级别数据时出错: {str(e)}")
            return {}
    
    def _resample_from_daily(self, daily_data: pd.DataFrame, target_level: str) -> pd.DataFrame:
        """
        从日线数据重采样生成分钟级别数据（模拟用，实际分析应使用真实分钟数据）
        
        参数:
            daily_data (pd.DataFrame): 日线数据
            target_level (str): 目标级别
            
        返回:
            pd.DataFrame: 重采样后的数据
        """
        # 注意：这只是一个模拟方法，实际交易中应使用真实分钟数据
        logger.warning("使用从日线重采样的数据仅用于演示，实际分析应使用真实分钟数据")
        
        # 复制日线数据基础
        resampled = daily_data.copy()
        
        # 根据目标级别设置缩放因子
        if target_level == "30min":
            scale = 8  # 一天8个30分钟
        elif target_level == "5min":
            scale = 48  # 一天48个5分钟
        elif target_level == "1min":
            scale = 240  # 一天240个1分钟
        else:
            scale = 1
        
        # 创建更多行，以模拟分钟数据
        result = []
        for idx, row in resampled.iterrows():
            base_open = row['open']
            base_close = row['close']
            base_high = row['high']
            base_low = row['low']
            base_volume = row['volume'] / scale
            
            # 判断总体趋势
            trend = 1 if base_close >= base_open else -1
            
            # 生成当天的分钟级别数据
            for i in range(scale):
                # 模拟日内波动
                random_factor = 0.99 + np.random.random() * 0.02  # 0.99-1.01之间的随机因子
                
                if trend > 0:
                    # 上涨趋势
                    sub_open = base_open * (1 + i * 0.001 * random_factor)
                    sub_close = sub_open * (1 + 0.002 * random_factor * (1 if np.random.random() > 0.3 else -1))
                    sub_high = max(sub_open, sub_close) * (1 + 0.001 * random_factor)
                    sub_low = min(sub_open, sub_close) * (1 - 0.001 * random_factor)
                else:
                    # 下跌趋势
                    sub_open = base_open * (1 - i * 0.001 * random_factor)
                    sub_close = sub_open * (1 - 0.002 * random_factor * (1 if np.random.random() > 0.3 else -1))
                    sub_high = max(sub_open, sub_close) * (1 + 0.001 * random_factor)
                    sub_low = min(sub_open, sub_close) * (1 - 0.001 * random_factor)
                
                # 确保高低点正确
                sub_high = max(sub_high, sub_open, sub_close)
                sub_low = min(sub_low, sub_open, sub_close)
                
                # 最后一个分钟要保证收盘价接近日线收盘价
                if i == scale - 1:
                    sub_close = base_close
                
                # 模拟分钟级别成交量
                sub_volume = base_volume * (0.5 + np.random.random())
                
                # 添加到结果集
                result.append({
                    'trade_date': idx + timedelta(minutes=i*(1440//scale)),
                    'open': sub_open,
                    'high': sub_high,
                    'low': sub_low,
                    'close': sub_close,
                    'volume': sub_volume
                })
        
        # 转换为DataFrame
        result_df = pd.DataFrame(result)
        result_df.set_index('trade_date', inplace=True)
        
        return result_df

    def mark_fractal_points(self, df: pd.DataFrame, min_elements: int = 3) -> pd.DataFrame:
        """
        标记分型（顶分型和底分型）
        
        参数:
            df (pd.DataFrame): K线数据
            min_elements (int): 最小元素数量，默认为3
            
        返回:
            pd.DataFrame: 带有分型标记的DataFrame
        """
        if df.empty or len(df) < min_elements:
            return df
        
        # 复制输入数据
        result_df = df.copy()
        
        # 初始化分型列
        result_df['fractal_top'] = False
        result_df['fractal_bottom'] = False
        
        # 标记顶分型和底分型
        for i in range(1, len(result_df) - 1):
            # 前、中、后三个K线
            prev = result_df.iloc[i-1]
            curr = result_df.iloc[i]
            next = result_df.iloc[i+1]
            
            # 顶分型: 中间K线的高点高于前后两个K线的高点
            if curr['high'] > prev['high'] and curr['high'] > next['high']:
                result_df.iloc[i, result_df.columns.get_loc('fractal_top')] = True
            
            # 底分型: 中间K线的低点低于前后两个K线的低点
            if curr['low'] < prev['low'] and curr['low'] < next['low']:
                result_df.iloc[i, result_df.columns.get_loc('fractal_bottom')] = True
        
        return result_df
    
    def mark_bi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标记缠论笔
        
        参数:
            df (pd.DataFrame): 带有分型标记的K线数据
            
        返回:
            pd.DataFrame: 带有笔标记的DataFrame
        """
        if df.empty or len(df) < 5:  # 至少需要5个K线才能形成笔
            return df
        
        # 复制输入数据
        result_df = df.copy()
        
        # 确保已标记分型
        if 'fractal_top' not in result_df.columns or 'fractal_bottom' not in result_df.columns:
            result_df = self.mark_fractal_points(result_df)
        
        # 初始化笔标记列
        result_df['bi_type'] = None  # 可能的值: 'top', 'bottom', None
        result_df['bi_price'] = np.nan
        result_df['bi_start'] = False
        result_df['bi_end'] = False
        
        # 提取所有有效分型
        tops = result_df[result_df['fractal_top'] == True].index.tolist()
        bottoms = result_df[result_df['fractal_bottom'] == True].index.tolist()
        
        if not tops or not bottoms:
            return result_df
        
        # 合并并排序所有分型
        all_points = [(idx, 'top' if idx in tops else 'bottom') for idx in set(tops + bottoms)]
        all_points.sort()  # 按日期排序
        
        # 严格按照缠论笔的定义标记
        bi_points = []
        for i in range(len(all_points) - 1):
            curr_idx, curr_type = all_points[i]
            next_idx, next_type = all_points[i + 1]
            
            # 相邻分型方向相反
            if curr_type != next_type:
                # 确保分型之间有包含关系（缠论笔的要求）
                curr_k = result_df.loc[curr_idx]
                next_k = result_df.loc[next_idx]
                
                # 顶分型后接底分型，需要下跌
                if curr_type == 'top' and next_type == 'bottom':
                    if next_k['low'] < curr_k['low']:  # 下方笔突破
                        bi_points.append((curr_idx, curr_type))
                        bi_points.append((next_idx, next_type))
                
                # 底分型后接顶分型，需要上涨
                elif curr_type == 'bottom' and next_type == 'top':
                    if next_k['high'] > curr_k['high']:  # 上方笔突破
                        bi_points.append((curr_idx, curr_type))
                        bi_points.append((next_idx, next_type))
        
        # 标记笔
        for i, (idx, bi_type) in enumerate(bi_points):
            price = result_df.loc[idx, 'high'] if bi_type == 'top' else result_df.loc[idx, 'low']
            result_df.loc[idx, 'bi_type'] = bi_type
            result_df.loc[idx, 'bi_price'] = price
            
            # 标记笔的起点和终点
            if i % 2 == 0:  # 每偶数点作为起点
                result_df.loc[idx, 'bi_start'] = True
            else:  # 每奇数点作为终点
                result_df.loc[idx, 'bi_end'] = True
        
        return result_df
    
    def mark_xianduan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标记缠论线段
        
        参数:
            df (pd.DataFrame): 带有笔标记的K线数据
            
        返回:
            pd.DataFrame: 带有线段标记的DataFrame
        """
        if df.empty:
            return df
        
        # 复制输入数据
        result_df = df.copy()
        
        # 确保已标记笔
        if 'bi_type' not in result_df.columns:
            result_df = self.mark_bi(result_df)
        
        # 初始化线段标记列
        result_df['xianduan_type'] = None  # 可能的值: 'top', 'bottom', None
        result_df['xianduan_price'] = np.nan
        result_df['xianduan_start'] = False
        result_df['xianduan_end'] = False
        
        # 提取所有笔端点
        bi_points = result_df[(result_df['bi_start'] == True) | (result_df['bi_end'] == True)]
        
        if len(bi_points) < 3:  # 至少需要3个笔端点才能确定线段
            return result_df
        
        # 线段标记逻辑
        # 采用特征序列法（简化版）
        xianduan_points = []
        
        # 初始状态
        if bi_points.iloc[0]['bi_type'] == 'bottom':
            direction = 1  # 向上线段
            extreme_idx = bi_points.iloc[0].name
            extreme_val = bi_points.iloc[0]['low']
        else:
            direction = -1  # 向下线段
            extreme_idx = bi_points.iloc[0].name
            extreme_val = bi_points.iloc[0]['high']
        
        xianduan_points.append((extreme_idx, 'bottom' if direction == 1 else 'top'))
        
        # 遍历所有笔，检查是否有线段特征序列成立
        i = 1
        while i < len(bi_points):
            curr = bi_points.iloc[i]
            
            if direction == 1:  # 向上线段中
                if curr['bi_type'] == 'top':
                    # 潜在的线段顶点
                    potential_top = curr
                    # 检查是否有效顶分型
                    if potential_top['high'] > extreme_val:
                        # 确认线段顶点
                        xianduan_points.append((potential_top.name, 'top'))
                        # 转向
                        direction = -1
                        extreme_idx = potential_top.name
                        extreme_val = potential_top['high']
                elif curr['bi_type'] == 'bottom' and curr['low'] < extreme_val:
                    # 更低的底，更新当前线段起点
                    extreme_idx = curr.name
                    extreme_val = curr['low']
                    # 修正最后一个线段点
                    if xianduan_points and xianduan_points[-1][1] == 'bottom':
                        xianduan_points[-1] = (extreme_idx, 'bottom')
                    else:
                        xianduan_points.append((extreme_idx, 'bottom'))
            else:  # 向下线段中
                if curr['bi_type'] == 'bottom':
                    # 潜在的线段底点
                    potential_bottom = curr
                    # 检查是否有效底分型
                    if potential_bottom['low'] < extreme_val:
                        # 确认线段底点
                        xianduan_points.append((potential_bottom.name, 'bottom'))
                        # 转向
                        direction = 1
                        extreme_idx = potential_bottom.name
                        extreme_val = potential_bottom['low']
                elif curr['bi_type'] == 'top' and curr['high'] > extreme_val:
                    # 更高的顶，更新当前线段起点
                    extreme_idx = curr.name
                    extreme_val = curr['high']
                    # 修正最后一个线段点
                    if xianduan_points and xianduan_points[-1][1] == 'top':
                        xianduan_points[-1] = (extreme_idx, 'top')
                    else:
                        xianduan_points.append((extreme_idx, 'top'))
            
            i += 1
        
        # 标记线段
        for i, (idx, xianduan_type) in enumerate(xianduan_points):
            price = result_df.loc[idx, 'high'] if xianduan_type == 'top' else result_df.loc[idx, 'low']
            result_df.loc[idx, 'xianduan_type'] = xianduan_type
            result_df.loc[idx, 'xianduan_price'] = price
            
            # 标记线段的起点和终点
            if i % 2 == 0:  # 每偶数点作为起点
                result_df.loc[idx, 'xianduan_start'] = True
            else:  # 每奇数点作为终点
                result_df.loc[idx, 'xianduan_end'] = True
        
        return result_df
    
    def detect_zhongshu(self, df: pd.DataFrame, min_segments: int = 3) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        检测中枢
        
        参数:
            df (pd.DataFrame): 带有线段标记的K线数据
            min_segments (int): 形成中枢的最小线段数，默认为3
            
        返回:
            Tuple[pd.DataFrame, List[Dict]]: 带有中枢标记的DataFrame和中枢数据列表
        """
        if df.empty:
            return df, []
        
        # 复制输入数据
        result_df = df.copy()
        
        # 确保已标记线段
        if 'xianduan_type' not in result_df.columns:
            result_df = self.mark_xianduan(result_df)
        
        # 提取所有线段端点
        xianduan_points = result_df[(result_df['xianduan_start'] == True) | (result_df['xianduan_end'] == True)]
        
        if len(xianduan_points) < min_segments + 1:  # 至少需要min_segments+1个线段端点才能形成中枢
            return result_df, []
        
        # 初始化中枢标记列
        result_df['in_zhongshu'] = False
        result_df['zhongshu_id'] = np.nan
        result_df['zhongshu_start'] = False
        result_df['zhongshu_end'] = False
        
        # 中枢识别逻辑
        zhongshu_list = []
        
        # 至少3个线段重叠区间形成中枢
        for i in range(len(xianduan_points) - min_segments):
            # 获取连续min_segments+1个线段端点
            segment_points = xianduan_points.iloc[i:i+min_segments+1]
            
            # 判断是否有重叠区间
            highs = []
            lows = []
            
            for j in range(min_segments):
                start_point = segment_points.iloc[j]
                end_point = segment_points.iloc[j+1]
                
                if start_point['xianduan_type'] == 'top':
                    highs.append(start_point['high'])
                    lows.append(end_point['low'])
                else:
                    lows.append(start_point['low'])
                    highs.append(end_point['high'])
            
            # 检查是否有共同区间
            overlap_high = min(highs)
            overlap_low = max(lows)
            
            # 有效中枢条件：重叠区间存在（上限高于下限）
            if overlap_high > overlap_low:
                # 找到中枢起止时间
                zhongshu_start = segment_points.iloc[0].name
                zhongshu_end = segment_points.iloc[-1].name
                
                # 添加中枢记录
                zhongshu_id = len(zhongshu_list) + 1
                zhongshu = {
                    'id': zhongshu_id,
                    'start_idx': zhongshu_start,
                    'end_idx': zhongshu_end,
                    'zg': overlap_high,  # 中枢上限
                    'zd': overlap_low,   # 中枢下限
                    'level': min_segments # 中枢级别
                }
                zhongshu_list.append(zhongshu)
                
                # 标记中枢范围内的K线
                mask = (result_df.index >= zhongshu_start) & (result_df.index <= zhongshu_end)
                result_df.loc[mask, 'in_zhongshu'] = True
                result_df.loc[mask, 'zhongshu_id'] = zhongshu_id
                result_df.loc[zhongshu_start, 'zhongshu_start'] = True
                result_df.loc[zhongshu_end, 'zhongshu_end'] = True
        
        # 合并相邻中枢
        i = 0
        while i < len(zhongshu_list) - 1:
            curr = zhongshu_list[i]
            next = zhongshu_list[i+1]
            
            # 检查是否为同方向中枢且有重叠
            if curr['end_idx'] >= next['start_idx']:
                # 合并中枢
                merged = {
                    'id': curr['id'],
                    'start_idx': curr['start_idx'],
                    'end_idx': next['end_idx'],
                    'zg': min(curr['zg'], next['zg']),
                    'zd': max(curr['zd'], next['zd']),
                    'level': max(curr['level'], next['level'])
                }
                
                # 更新中枢列表
                zhongshu_list[i] = merged
                zhongshu_list.pop(i+1)
                
                # 更新DataFrame标记
                mask = (result_df.index >= merged['start_idx']) & (result_df.index <= merged['end_idx'])
                result_df.loc[mask, 'in_zhongshu'] = True
                result_df.loc[mask, 'zhongshu_id'] = merged['id']
                result_df.loc[merged['start_idx'], 'zhongshu_start'] = True
                result_df.loc[merged['end_idx'], 'zhongshu_end'] = True
            else:
                i += 1
        
        return result_df, zhongshu_list

    def find_signals(self, df: pd.DataFrame, zhongshu_list: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        识别缠论买卖点信号
        
        参数:
            df (pd.DataFrame): 带有中枢标记的K线数据
            zhongshu_list (List[Dict]): 中枢数据列表
            
        返回:
            Tuple[pd.DataFrame, List[Dict]]: 带有买卖点信号的DataFrame和信号列表
        """
        if df.empty or not zhongshu_list:
            return df, []
        
        # 复制输入数据
        result_df = df.copy()
        
        # 初始化买卖点信号列
        result_df['signal_type'] = None  # 可能的值: '1buy', '2buy', '3buy', '1sell', '2sell', '3sell', None
        result_df['signal_price'] = np.nan
        
        # 提取所有线段端点
        xianduan_points = result_df[(result_df['xianduan_start'] == True) | (result_df['xianduan_end'] == True)]
        
        # 买卖点识别逻辑
        signals = []
        
        # 遍历中枢
        for zhongshu in zhongshu_list:
            zhongshu_id = zhongshu['id']
            zhongshu_start = zhongshu['start_idx']
            zhongshu_end = zhongshu['end_idx']
            zhongshu_zg = zhongshu['zg']  # 中枢上限
            zhongshu_zd = zhongshu['zd']  # 中枢下限
            
            # 提取中枢内的线段点
            zhongshu_xianduan = xianduan_points[(xianduan_points.index >= zhongshu_start) & 
                                              (xianduan_points.index <= zhongshu_end)]
            
            # 提取中枢后的线段点
            post_zhongshu_xianduan = xianduan_points[xianduan_points.index > zhongshu_end]
            
            if post_zhongshu_xianduan.empty:
                continue
                
            # 获取中枢方向
            if zhongshu_xianduan.iloc[0]['xianduan_type'] == 'bottom' and zhongshu_xianduan.iloc[-1]['xianduan_type'] == 'top':
                zhongshu_direction = 1  # 上涨中枢
            elif zhongshu_xianduan.iloc[0]['xianduan_type'] == 'top' and zhongshu_xianduan.iloc[-1]['xianduan_type'] == 'bottom':
                zhongshu_direction = -1  # 下跌中枢
            else:
                zhongshu_direction = 0  # 盘整中枢
            
            # 第一类买卖点：中枢第一次离开
            first_break_point = post_zhongshu_xianduan.iloc[0]
            if zhongshu_direction >= 0 and first_break_point['xianduan_type'] == 'top' and first_break_point['high'] > zhongshu_zg:
                # 第一类卖点
                signal = {
                    'date': first_break_point.name,
                    'type': '1sell',
                    'price': first_break_point['high'],
                    'zhongshu_id': zhongshu_id,
                    'description': f'第一类卖点：中枢{zhongshu_id}向上突破后回落'
                }
                signals.append(signal)
                result_df.loc[first_break_point.name, 'signal_type'] = '1sell'
                result_df.loc[first_break_point.name, 'signal_price'] = first_break_point['high']
                
            elif zhongshu_direction <= 0 and first_break_point['xianduan_type'] == 'bottom' and first_break_point['low'] < zhongshu_zd:
                # 第一类买点
                signal = {
                    'date': first_break_point.name,
                    'type': '1buy',
                    'price': first_break_point['low'],
                    'zhongshu_id': zhongshu_id,
                    'description': f'第一类买点：中枢{zhongshu_id}向下突破后反弹'
                }
                signals.append(signal)
                result_df.loc[first_break_point.name, 'signal_type'] = '1buy'
                result_df.loc[first_break_point.name, 'signal_price'] = first_break_point['low']
            
            # 如果中枢后有更多线段点，检查第二、三类买卖点
            if len(post_zhongshu_xianduan) > 1:
                # 第二类买卖点：中枢离开，回试中枢，未回中枢
                second_point = post_zhongshu_xianduan.iloc[1]
                if first_break_point['xianduan_type'] == 'top' and second_point['xianduan_type'] == 'bottom':
                    if second_point['low'] >= zhongshu_zd and second_point['low'] <= zhongshu_zg:
                        # 第二类买点
                        signal = {
                            'date': second_point.name,
                            'type': '2buy',
                            'price': second_point['low'],
                            'zhongshu_id': zhongshu_id,
                            'description': f'第二类买点：中枢{zhongshu_id}上方回试不跌破'
                        }
                        signals.append(signal)
                        result_df.loc[second_point.name, 'signal_type'] = '2buy'
                        result_df.loc[second_point.name, 'signal_price'] = second_point['low']
                        
                elif first_break_point['xianduan_type'] == 'bottom' and second_point['xianduan_type'] == 'top':
                    if second_point['high'] >= zhongshu_zd and second_point['high'] <= zhongshu_zg:
                        # 第二类卖点
                        signal = {
                            'date': second_point.name,
                            'type': '2sell',
                            'price': second_point['high'],
                            'zhongshu_id': zhongshu_id,
                            'description': f'第二类卖点：中枢{zhongshu_id}下方回试不突破'
                        }
                        signals.append(signal)
                        result_df.loc[second_point.name, 'signal_type'] = '2sell'
                        result_df.loc[second_point.name, 'signal_price'] = second_point['high']
                
                # 第三类买卖点：中枢完成，离开后无回试或不回中枢
                if len(post_zhongshu_xianduan) > 2:
                    third_point = post_zhongshu_xianduan.iloc[2]
                    if (first_break_point['xianduan_type'] == 'top' and 
                        second_point['xianduan_type'] == 'bottom' and 
                        third_point['xianduan_type'] == 'top' and
                        third_point['high'] > first_break_point['high']):
                        # 第三类卖点
                        signal = {
                            'date': third_point.name,
                            'type': '3sell',
                            'price': third_point['high'],
                            'zhongshu_id': zhongshu_id,
                            'description': f'第三类卖点：中枢{zhongshu_id}上方新高确认上升趋势'
                        }
                        signals.append(signal)
                        result_df.loc[third_point.name, 'signal_type'] = '3sell'
                        result_df.loc[third_point.name, 'signal_price'] = third_point['high']
                        
                    elif (first_break_point['xianduan_type'] == 'bottom' and 
                          second_point['xianduan_type'] == 'top' and 
                          third_point['xianduan_type'] == 'bottom' and
                          third_point['low'] < first_break_point['low']):
                        # 第三类买点
                        signal = {
                            'date': third_point.name,
                            'type': '3buy',
                            'price': third_point['low'],
                            'zhongshu_id': zhongshu_id,
                            'description': f'第三类买点：中枢{zhongshu_id}下方新低确认下降趋势'
                        }
                        signals.append(signal)
                        result_df.loc[third_point.name, 'signal_type'] = '3buy'
                        result_df.loc[third_point.name, 'signal_price'] = third_point['low']
        
        return result_df, signals
    
    def check_beichi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测背驰
        
        参数:
            df (pd.DataFrame): 带有线段和中枢标记的K线数据
            
        返回:
            pd.DataFrame: 带有背驰标记的DataFrame
        """
        if df.empty:
            return df
        
        # 复制输入数据
        result_df = df.copy()
        
        # 初始化背驰标记列
        result_df['beichi'] = False
        result_df['beichi_type'] = None  # 可能的值: 'price', 'vol', 'macd', None
        
        # 提取所有线段端点
        xianduan_points = result_df[(result_df['xianduan_start'] == True) | (result_df['xianduan_end'] == True)]
        
        if len(xianduan_points) < 4:  # 至少需要两个完整线段才能比较
            return result_df
        
        # 分别处理向上线段和向下线段
        for i in range(2, len(xianduan_points) - 1, 2):
            current_point = xianduan_points.iloc[i]
            previous_point = xianduan_points.iloc[i-2]
            
            # 获取当前和前一个线段区间
            if current_point['xianduan_type'] == 'top':
                # 向上线段
                current_segment_start = xianduan_points.iloc[i-1].name
                current_segment_end = current_point.name
                previous_segment_start = xianduan_points.iloc[i-3].name
                previous_segment_end = previous_point.name
                
                # 提取线段区间数据
                current_segment = result_df.loc[current_segment_start:current_segment_end]
                previous_segment = result_df.loc[previous_segment_start:previous_segment_end]
                
                # 比较价格涨幅
                current_price_change = current_point['high'] - result_df.loc[current_segment_start, 'low']
                previous_price_change = previous_point['high'] - result_df.loc[previous_segment_start, 'low']
                
                # 比较成交量
                current_volume = current_segment['volume'].sum()
                previous_volume = previous_segment['volume'].sum()
                
                # 比较MACD柱状和
                if 'MACD_BAR' in result_df.columns:
                    current_macd_sum = current_segment[current_segment['MACD_BAR'] > 0]['MACD_BAR'].sum()
                    previous_macd_sum = previous_segment[previous_segment['MACD_BAR'] > 0]['MACD_BAR'].sum()
                else:
                    current_macd_sum = previous_macd_sum = 0
                
                # 顶背驰判断：价格创新高但动能减弱
                if (current_point['high'] > previous_point['high'] and 
                    (current_volume < previous_volume or current_macd_sum < previous_macd_sum)):
                    result_df.loc[current_point.name, 'beichi'] = True
                    if current_volume < previous_volume:
                        result_df.loc[current_point.name, 'beichi_type'] = 'vol'
                    else:
                        result_df.loc[current_point.name, 'beichi_type'] = 'macd'
                
            elif current_point['xianduan_type'] == 'bottom':
                # 向下线段
                current_segment_start = xianduan_points.iloc[i-1].name
                current_segment_end = current_point.name
                previous_segment_start = xianduan_points.iloc[i-3].name
                previous_segment_end = previous_point.name
                
                # 提取线段区间数据
                current_segment = result_df.loc[current_segment_start:current_segment_end]
                previous_segment = result_df.loc[previous_segment_start:previous_segment_end]
                
                # 比较价格跌幅
                current_price_change = result_df.loc[current_segment_start, 'high'] - current_point['low']
                previous_price_change = result_df.loc[previous_segment_start, 'high'] - previous_point['low']
                
                # 比较成交量
                current_volume = current_segment['volume'].sum()
                previous_volume = previous_segment['volume'].sum()
                
                # 比较MACD柱状和
                if 'MACD_BAR' in result_df.columns:
                    current_macd_sum = current_segment[current_segment['MACD_BAR'] < 0]['MACD_BAR'].abs().sum()
                    previous_macd_sum = previous_segment[previous_segment['MACD_BAR'] < 0]['MACD_BAR'].abs().sum()
                else:
                    current_macd_sum = previous_macd_sum = 0
                
                # 底背驰判断：价格创新低但动能减弱
                if (current_point['low'] < previous_point['low'] and 
                    (current_volume < previous_volume or current_macd_sum < previous_macd_sum)):
                    result_df.loc[current_point.name, 'beichi'] = True
                    if current_volume < previous_volume:
                        result_df.loc[current_point.name, 'beichi_type'] = 'vol'
                    else:
                        result_df.loc[current_point.name, 'beichi_type'] = 'macd'
        
        return result_df
    
    def build_trend_matrix(self) -> pd.DataFrame:
        """
        构建4x4趋势状态矩阵
        
        返回:
            pd.DataFrame: 多级别趋势状态矩阵
        """
        # 初始化趋势矩阵
        matrix_data = {
            'level': self.levels,
            'trend': ['unknown'] * len(self.levels),
            'beichi': [False] * len(self.levels),
            'signal': [None] * len(self.levels),
            'zhongshu': [None] * len(self.levels)
        }
        
        matrix = pd.DataFrame(matrix_data)
        matrix.set_index('level', inplace=True)
        
        # 填充矩阵数据
        for level in self.levels:
            if level in self.level_data and not self.level_data[level].empty:
                df = self.level_data[level]
                
                # 判断趋势
                if 'MA20' in df.columns and 'MA60' in df.columns:
                    last_ma20 = df['MA20'].iloc[-1]
                    last_ma60 = df['MA60'].iloc[-1]
                    ma20_slope = df['MA20'].iloc[-5] - df['MA20'].iloc[-1]
                    
                    if last_ma20 > last_ma60 and ma20_slope > 0:
                        matrix.loc[level, 'trend'] = 'up'
                    elif last_ma20 < last_ma60 and ma20_slope < 0:
                        matrix.loc[level, 'trend'] = 'down'
                    else:
                        matrix.loc[level, 'trend'] = 'sideways'
                
                # 判断背驰
                if 'beichi' in df.columns:
                    matrix.loc[level, 'beichi'] = df['beichi'].iloc[-5:].any()
                
                # 获取最近信号
                if 'signal_type' in df.columns:
                    recent_signals = df[df['signal_type'].notna()].iloc[-3:]
                    if not recent_signals.empty:
                        matrix.loc[level, 'signal'] = recent_signals['signal_type'].iloc[-1]
                
                # 判断是否在中枢中
                if 'in_zhongshu' in df.columns:
                    matrix.loc[level, 'zhongshu'] = df['in_zhongshu'].iloc[-1]
        
        self.trend_matrix = matrix
        return matrix
    
    def trend_confirm(self, higher_level: str, lower_level: str) -> bool:
        """
        验证高低级别趋势是否协调
        
        参数:
            higher_level (str): 高级别标识，如 "daily"
            lower_level (str): 低级别标识，如 "30min"
            
        返回:
            bool: 高低级别趋势是否协调
        """
        if not isinstance(self.trend_matrix, pd.DataFrame) or self.trend_matrix.empty:
            self.build_trend_matrix()
            
        if higher_level not in self.trend_matrix.index or lower_level not in self.trend_matrix.index:
            return False
            
        # 获取高低级别趋势和信号
        higher_trend = self.trend_matrix.loc[higher_level, 'trend']
        lower_signal = self.trend_matrix.loc[lower_level, 'signal']
        
        # 验证规则
        if higher_trend == 'up' and lower_signal in ['1buy', '2buy', '3buy']:
            return True
        elif higher_trend == 'down' and lower_signal in ['1sell', '2sell', '3sell']:
            return True
        else:
            return False
    
    def generate_signal(self) -> Dict:
        """
        生成当前交易信号
        
        返回:
            Dict: 交易信号及其详情
        """
        # 确保已构建趋势矩阵
        if not isinstance(self.trend_matrix, pd.DataFrame) or self.trend_matrix.empty:
            self.build_trend_matrix()
        
        # 初始化信号
        signal = {
            'action': 'hold',  # 默认持仓不动
            'confidence': 0.5,  # 信心度
            'stop_loss': None,  # 止损位
            'reason': '没有明确信号',
            'matrix': self.trend_matrix.to_dict()
        }
        
        # 特殊情况1：日线上升+30分钟无背驰+5分钟三买
        if ("daily" in self.trend_matrix.index and 
            "30min" in self.trend_matrix.index and 
            "5min" in self.trend_matrix.index):
            
            daily_trend = self.trend_matrix.loc["daily", "trend"]
            min30_beichi = self.trend_matrix.loc["30min", "beichi"]
            min5_signal = self.trend_matrix.loc["5min", "signal"]
            
            if daily_trend == 'up' and not min30_beichi and min5_signal == '3buy':
                signal.update({
                    'action': 'buy',
                    'confidence': 0.8,
                    'reason': '日线上升+30分钟无背驰+5分钟三买',
                    'stop_loss': '30分钟中枢下沿'
                })
                
                # 设置止损位
                if "30min" in self.level_data:
                    df = self.level_data["30min"]
                    zhongshu_list = self.zhongshu_data.get("30min", [])
                    if zhongshu_list:
                        last_zhongshu = zhongshu_list[-1]
                        signal['stop_loss_price'] = last_zhongshu['zd']
        
        # 特殊情况2：日线下跌+30分钟背驰+5分钟二买
        if ("daily" in self.trend_matrix.index and 
            "30min" in self.trend_matrix.index and 
            "5min" in self.trend_matrix.index):
            
            daily_trend = self.trend_matrix.loc["daily", "trend"]
            min30_beichi = self.trend_matrix.loc["30min", "beichi"]
            min5_signal = self.trend_matrix.loc["5min", "signal"]
            
            if daily_trend == 'down' and min30_beichi and min5_signal == '2buy':
                signal.update({
                    'action': 'buy',
                    'confidence': 0.6,
                    'reason': '日线下跌+30分钟背驰+5分钟二买',
                    'stop_loss': '最近低点下方'
                })
                
                # 设置止损位
                if "5min" in self.level_data:
                    df = self.level_data["5min"]
                    if 'low' in df.columns:
                        lowest = df.iloc[-20:]['low'].min()
                        signal['stop_loss_price'] = lowest * 0.98  # 设置止损在最低点下方2%
        
        # 特殊情况3：多级别共振下跌
        if all(level in self.trend_matrix.index for level in ["daily", "30min", "5min"]):
            daily_trend = self.trend_matrix.loc["daily", "trend"]
            min30_trend = self.trend_matrix.loc["30min", "trend"]
            min5_trend = self.trend_matrix.loc["5min", "trend"]
            
            if daily_trend == 'down' and min30_trend == 'down' and min5_trend == 'down':
                signal.update({
                    'action': 'sell',
                    'confidence': 0.9,
                    'reason': '多级别共振下跌',
                    'stop_loss': '日线低点形成确认'
                })
        
        return signal

    def plot_chan_chart(self, level: str) -> str:
        """
        绘制缠论分析图表
        
        参数:
            level (str): 要绘制的级别
            
        返回:
            str: 保存的图表路径
        """
        if level not in self.level_data or self.level_data[level].empty:
            logger.warning(f"没有{level}级别的数据，无法绘制图表")
            return ""
        
        df = self.level_data[level]
        
        # 创建图表
        fig = plt.figure(figsize=(16, 12))
        
        # 设置子图
        price_ax = plt.subplot2grid((4, 1), (0, 0), rowspan=2)  # 价格子图
        vol_ax = plt.subplot2grid((4, 1), (2, 0), rowspan=1)    # 成交量子图
        macd_ax = plt.subplot2grid((4, 1), (3, 0), rowspan=1)   # MACD子图
        
        # 设置图表标题
        fig.suptitle(f"{self.stock_code} {self.stock_name} {level}级别缠论分析", fontsize=16)
        
        # 绘制K线图
        dates = df.index
        
        # 上涨K线
        up = df[df['close'] >= df['open']]
        # 下跌K线
        down = df[df['close'] < df['open']]
        
        # 绘制K线图
        width = 0.4
        price_ax.bar(up.index, up['high'] - up['low'], width, bottom=up['low'], color='red', alpha=0.8)
        price_ax.bar(up.index, up['close'] - up['open'], width, bottom=up['open'], color='red')
        price_ax.bar(down.index, down['high'] - down['low'], width, bottom=down['low'], color='green', alpha=0.8)
        price_ax.bar(down.index, down['close'] - down['open'], width, bottom=down['open'], color='green')
        
        # 绘制MA均线
        for ma in [5, 10, 20, 60]:
            if f'MA{ma}' in df.columns:
                price_ax.plot(df.index, df[f'MA{ma}'], label=f'MA{ma}')
        
        # 绘制笔
        if 'bi_start' in df.columns:
            # 连接笔的起点和终点
            bi_start_points = df[df['bi_start'] == True]
            bi_end_points = df[df['bi_end'] == True]
            
            for i in range(min(len(bi_start_points), len(bi_end_points))):
                start_idx = bi_start_points.index[i]
                end_idx = bi_end_points.index[i]
                
                start_type = df.loc[start_idx, 'bi_type']
                end_type = df.loc[end_idx, 'bi_type']
                
                start_price = df.loc[start_idx, 'low'] if start_type == 'bottom' else df.loc[start_idx, 'high']
                end_price = df.loc[end_idx, 'low'] if end_type == 'bottom' else df.loc[end_idx, 'high']
                
                price_ax.plot([start_idx, end_idx], [start_price, end_price], 'b-', linewidth=1)
        
        # 绘制线段
        if 'xianduan_start' in df.columns:
            # 连接线段的起点和终点
            xianduan_start_points = df[df['xianduan_start'] == True]
            xianduan_end_points = df[df['xianduan_end'] == True]
            
            for i in range(min(len(xianduan_start_points), len(xianduan_end_points))):
                start_idx = xianduan_start_points.index[i]
                end_idx = xianduan_end_points.index[i]
                
                start_type = df.loc[start_idx, 'xianduan_type']
                end_type = df.loc[end_idx, 'xianduan_type']
                
                start_price = df.loc[start_idx, 'low'] if start_type == 'bottom' else df.loc[start_idx, 'high']
                end_price = df.loc[end_idx, 'low'] if end_type == 'bottom' else df.loc[end_idx, 'high']
                
                price_ax.plot([start_idx, end_idx], [start_price, end_price], 'r-', linewidth=2)
        
        # 绘制中枢
        if 'in_zhongshu' in df.columns and 'zhongshu_id' in df.columns:
            zhongshu_ids = df[df['in_zhongshu'] == True]['zhongshu_id'].unique()
            
            for zhongshu_id in zhongshu_ids:
                zhongshu_df = df[(df['in_zhongshu'] == True) & (df['zhongshu_id'] == zhongshu_id)]
                start_idx = zhongshu_df.index[0]
                end_idx = zhongshu_df.index[-1]
                
                # 获取中枢上下限
                zhongshu_list = self.zhongshu_data.get(level, [])
                zhongshu = next((z for z in zhongshu_list if z['id'] == zhongshu_id), None)
                
                if zhongshu:
                    zg = zhongshu['zg']  # 中枢上限
                    zd = zhongshu['zd']  # 中枢下限
                    
                    # 绘制中枢区间
                    rect = Rectangle((start_idx, zd), end_idx - start_idx, zg - zd, 
                                     linewidth=1, edgecolor='purple', facecolor='yellow', alpha=0.2)
                    price_ax.add_patch(rect)
                    
                    # 标注中枢编号
                    price_ax.text(start_idx, zg, f"中枢{int(zhongshu_id)}", fontsize=10, color='purple')
        
        # 标记买卖点信号
        if 'signal_type' in df.columns:
            signal_points = df[df['signal_type'].notna()]
            
            for idx, row in signal_points.iterrows():
                signal_type = row['signal_type']
                signal_price = row['signal_price']
                
                if 'buy' in signal_type:
                    marker = '^'
                    color = 'red'
                    y_offset = -df['high'].max() * 0.01  # 标记在K线底部
                else:
                    marker = 'v'
                    color = 'green'
                    y_offset = df['high'].max() * 0.01  # 标记在K线顶部
                
                price_ax.scatter(idx, signal_price, s=100, marker=marker, color=color)
                price_ax.text(idx, signal_price + y_offset, signal_type, fontsize=8, color=color, 
                              horizontalalignment='center')
        
        # 标记背驰点
        if 'beichi' in df.columns:
            beichi_points = df[df['beichi'] == True]
            
            for idx, row in beichi_points.iterrows():
                beichi_type = row['beichi_type']
                price = row['high'] if row['xianduan_type'] == 'top' else row['low']
                
                price_ax.scatter(idx, price, s=120, marker='*', color='orange')
                price_ax.text(idx, price, f"背驰({beichi_type})", fontsize=8, color='orange', 
                              horizontalalignment='center')
        
        # 绘制成交量
        vol_ax.bar(up.index, up['volume'], width, color='red')
        vol_ax.bar(down.index, down['volume'], width, color='green')
        
        # 绘制成交量均线
        if 'VOL_MA5' in df.columns:
            vol_ax.plot(df.index, df['VOL_MA5'], 'b-', label='VOL MA5')
        if 'VOL_MA10' in df.columns:
            vol_ax.plot(df.index, df['VOL_MA10'], 'y-', label='VOL MA10')
            
        # 绘制MACD
        if all(x in df.columns for x in ['MACD_DIF', 'MACD_DEA', 'MACD_BAR']):
            macd_ax.plot(df.index, df['MACD_DIF'], 'b-', label='DIF')
            macd_ax.plot(df.index, df['MACD_DEA'], 'y-', label='DEA')
            
            # 绘制MACD柱状图
            macd_ax.bar(df.index, df['MACD_BAR'], width, 
                       color=df['MACD_BAR'].apply(lambda x: 'red' if x > 0 else 'green'))
            
            # 添加零轴线
            macd_ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        
        # 设置坐标轴
        for ax in [price_ax, vol_ax, macd_ax]:
            ax.grid(True, alpha=0.3)
            ax.set_xlim(df.index[0], df.index[-1])
            
            # 设置x轴标签
            if len(df) > 30:
                # 仅显示部分日期标签
                x_ticks = np.linspace(0, len(df) - 1, 10, dtype=int)
                ax.set_xticks([df.index[i] for i in x_ticks])
                # 确保索引是datetime类型
                if isinstance(df.index[0], (pd.Timestamp, datetime)):
                    ax.set_xticklabels([df.index[i].strftime('%Y-%m-%d') for i in x_ticks], rotation=45)
                else:
                    ax.set_xticklabels([str(df.index[i]) for i in x_ticks], rotation=45)
        
        # 设置标签
        price_ax.set_ylabel('价格')
        vol_ax.set_ylabel('成交量')
        macd_ax.set_ylabel('MACD')
        
        # 添加图例
        price_ax.legend(loc='upper left')
        vol_ax.legend(loc='upper left')
        macd_ax.legend(loc='upper left')
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # 保存图表
        chart_filename = f"{self.stock_code}_{level}_{self.end_date_str}_chan.png"
        chart_path = os.path.join(self.chart_path, chart_filename)
        plt.savefig(chart_path, dpi=300)
        plt.close()
        
        logger.info(f"已保存{level}级别缠论分析图表到 {chart_path}")
        return chart_path
    
    def llm_enhance_analysis(self) -> Dict:
        """
        使用LLM增强分析结果
        
        返回:
            Dict: LLM增强的分析结果
        """
        # 确保已构建趋势矩阵
        if not isinstance(self.trend_matrix, pd.DataFrame) or self.trend_matrix.empty:
            self.build_trend_matrix()
        
        # 构建分析摘要文本
        daily_analysis = "无数据"
        min30_analysis = "无数据"
        min5_analysis = "无数据"
        sup_res = {}
        
        # 获取支撑压力位
        if "daily" in self.level_data and not self.level_data["daily"].empty:
            from utils.indicators import calculate_support_resistance
            
            df = self.level_data["daily"]
            sup_res = calculate_support_resistance(df)
            
            daily_trend = self.trend_matrix.loc["daily", "trend"] if "daily" in self.trend_matrix.index else "unknown"
            daily_signal = self.trend_matrix.loc["daily", "signal"] if "daily" in self.trend_matrix.index else None
            
            daily_analysis = f"趋势:{daily_trend}, 信号:{daily_signal if daily_signal else '无'}"
            
            if "zhongshu_data" in self.__dict__ and "daily" in self.zhongshu_data:
                zhongshu_list = self.zhongshu_data["daily"]
                if zhongshu_list:
                    last_zhongshu = zhongshu_list[-1]
                    daily_analysis += f", 当前中枢:[{last_zhongshu['zd']:.2f},{last_zhongshu['zg']:.2f}]"
        
        if "30min" in self.level_data and not self.level_data["30min"].empty:
            min30_trend = self.trend_matrix.loc["30min", "trend"] if "30min" in self.trend_matrix.index else "unknown"
            min30_beichi = self.trend_matrix.loc["30min", "beichi"] if "30min" in self.trend_matrix.index else False
            
            min30_analysis = f"趋势:{min30_trend}, 背驰:{min30_beichi}"
        
        if "5min" in self.level_data and not self.level_data["5min"].empty:
            min5_trend = self.trend_matrix.loc["5min", "trend"] if "5min" in self.trend_matrix.index else "unknown"
            min5_signal = self.trend_matrix.loc["5min", "signal"] if "5min" in self.trend_matrix.index else None
            
            min5_analysis = f"趋势:{min5_trend}, 信号:{min5_signal if min5_signal else '无'}"
        
        # 构建LLM提示词
        prompt = f"""基于以下缠论分析结果给出建议：
            1. 日线级别：{daily_analysis}
            2. 30分钟级别：{min30_analysis} 
            3. 5分钟级别：{min5_analysis}
            4. 关键位置：支撑{sup_res.get('support', [])} 压力{sup_res.get('resistance', [])}

            请用专业术语回答：
            - 当前多级别联立状态
            - 最优交易策略
            - 风控位设置依据"""
        
        try:
            # 使用LLM API获取分析
            llm_result = self.llm_api.query(prompt)
            
            logger.info(f"已使用LLM增强分析结果")
            
            return {
                'prompt': prompt,
                'result': llm_result,
                'levels': {
                    'daily': daily_analysis,
                    '30min': min30_analysis,
                    '5min': min5_analysis
                },
                'support_resistance': sup_res
            }
            
        except Exception as e:
            logger.error(f"使用LLM增强分析时出错: {str(e)}")
            
            # 返回基本分析结果
            return {
                'prompt': prompt,
                'result': "LLM分析失败，请查看基本分析结果",
                'levels': {
                    'daily': daily_analysis,
                    '30min': min30_analysis,
                    '5min': min5_analysis
                },
                'support_resistance': sup_res
            }
    
    def analyze_level(self, level: str) -> Dict:
        """
        分析指定级别的数据
        
        参数:
            level (str): 要分析的级别
            
        返回:
            Dict: 分析结果
        """
        if level not in self.level_data or self.level_data[level].empty:
            logger.warning(f"没有{level}级别的数据，无法进行分析")
            return {}
        
        logger.info(f"开始分析{self.stock_code} {level}级别数据...")
        
        # 获取对应级别的数据
        df = self.level_data[level]
        
        # 计算技术指标
        df, indicators = calculate_technical_indicators(df)
        self.level_data[level] = df
        
        # 标记分型
        df = self.mark_fractal_points(df)
        
        # 标记笔
        df = self.mark_bi(df)
        self.bi_data[level] = df[df['bi_type'].notna()]
        
        # 标记线段
        df = self.mark_xianduan(df)
        self.xianduan_data[level] = df[df['xianduan_type'].notna()]
        
        # 检测中枢
        df, zhongshu_list = self.detect_zhongshu(df)
        self.zhongshu_data[level] = zhongshu_list
        
        # 检测买卖点信号
        df, signals = self.find_signals(df, zhongshu_list)
        self.signals[level] = signals
        
        # 检测背驰
        df = self.check_beichi(df)
        
        # 更新数据
        self.level_data[level] = df
        
        # 绘制图表
        chart_path = self.plot_chan_chart(level)
        
        # 返回分析结果
        result = {
            'level': level,
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'data_count': len(df),
            'fractal_count': len(df[(df['fractal_top'] == True) | (df['fractal_bottom'] == True)]),
            'bi_count': len(self.bi_data[level]),
            'xianduan_count': len(self.xianduan_data[level]),
            'zhongshu_count': len(zhongshu_list),
            'signal_count': len(signals),
            'signals': signals,
            'chart_path': chart_path
        }
        
        logger.info(f"完成{level}级别分析: 发现{result['bi_count']}笔, {result['xianduan_count']}线段, {result['zhongshu_count']}中枢, {result['signal_count']}信号")
        
        return result
    
    def run_analysis(self) -> Dict:
        """
        执行完整的缠论分析流程
        
        返回:
            Dict: 分析结果
        """
        logger.info(f"开始对{self.stock_code} {self.stock_name}执行缠论分析...")
        
        # 获取多级别数据
        self.get_multi_level_data()
        
        # 分析各个级别
        level_results = {}
        for level in self.levels:
            if level in self.level_data and not self.level_data[level].empty:
                level_results[level] = self.analyze_level(level)
        
        # 构建多级别趋势矩阵
        self.build_trend_matrix()
        
        # 生成交易信号
        signal = self.generate_signal()
        
        # LLM增强分析
        llm_analysis = self.llm_enhance_analysis()
        
        # 汇总分析结果
        self.analysis_result = {
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'analysis_date': self.end_date.strftime('%Y-%m-%d'),
            'levels': self.levels,
            'level_results': level_results,
            'trend_matrix': self.trend_matrix.to_dict(),
            'signal': signal,
            'llm_analysis': llm_analysis,
            'description': f"{self.stock_name}({self.stock_code})多级别缠论分析",
            'chart_path': level_results.get('daily', {}).get('chart_path', '')
        }
        
        # 保存分析结果
        self.save_analysis_result()
        
        logger.info(f"完成{self.stock_code} {self.stock_name}的缠论分析")
        
        return self.analysis_result
