# -*- coding: utf-8 -*-
"""缠论底背驰选股模块，专注于识别底部趋势背驰形态"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from typing import Dict, List, Optional, Tuple
import os

from selector.base_selector import BaseSelector
from utils.logger import get_logger
from data.db_manager import DatabaseManager

# 创建日志记录器
logger = get_logger(__name__)

class ChanBackchSelector(BaseSelector):
    """缠论底背驰选股系统，专注于识别底部趋势背驰买点"""
    
    def __init__(self, days=250, threshold=60, limit=50):
        """
        初始化缠论底背驰选股器
        
        参数:
            days (int, 可选): 回溯数据天数，默认250天
            threshold (int, 可选): 最低背驰分数阈值，默认60分
            limit (int, 可选): 限制结果数量，默认50只
        """
        super().__init__(days, threshold, limit)
        self.db_manager = DatabaseManager()
        
        # 背驰选股专用参数（基于缠论背驰定义设置）
        self.params = {
            'pivot_window': 5,       # 中枢识别窗口
            'macd_fast': 12,         # MACD快线周期
            'macd_slow': 26,         # MACD慢线周期
            'macd_signal': 9,        # MACD信号线周期
            'vol_ratio': 0.85,       # 量能萎缩阈值（小于前一段的比例）- 放宽条件
            'min_vol': 1e6,          # 最小成交量（股）
            'min_trend_len': 15,     # 最小趋势长度（天）- 缩短趋势长度要求
            'diff_threshold': 0.12,  # MACD差值阈值（判断背驰的百分比）
            'recency_days': 30       # 最近30天内的背驰信号才有效
        }

    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取日线数据（包含复权价格）
        
        参数:
            stock_code (str): 股票代码
            start_date (str): 开始日期
            end_date (str): 结束日期
            
        返回:
            pd.DataFrame: 股票日线数据
        """
        sql = f"""
        SELECT trade_date, open, high, low, close, volume 
        FROM stock_daily 
        WHERE stock_code='{stock_code}' AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY trade_date
        """
        try:
            df = self.db_manager.read_sql(sql)
            if df.empty:
                logger.warning(f"未能获取到股票 {stock_code} 的数据")
                return pd.DataFrame()
                
            # 确保数值列为float类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = df[col].astype(float)
                
            # 设置日期索引
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 数据时出错: {str(e)}")
            return pd.DataFrame()

    def evaluate_stock(self, stock_code: str) -> Optional[Dict]:
        """
        评估单只股票，寻找背驰信号
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            Optional[Dict]: 评估结果字典，如果无法评估则返回None
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=self.days)).strftime('%Y-%m-%d')
        
        df = self.get_daily_data(stock_code, start_date, end_date)
        if len(df) < 60:  # 至少需要60个交易日数据才能形成有效趋势
            return None
        
        # 计算技术指标和中枢
        df = self._calculate_features(df)
        
        # 中枢识别（日线级别）
        # 获取波峰波谷点位置（高点和低点）
        df['pivots'] = self._identify_pivots(df)
        highs, lows = self._get_peak_trough_indices(df)
        
        # 判断背驰
        divergence_signals = self._detect_divergence(df, highs, lows)
        
        # 获取股票名称
        stock_name = self._get_stock_name(stock_code)
        
        # 趋势评估（判断当前趋势方向）
        trend_direction, trend_strength = self._evaluate_trend(df)
        
        # 如果没有背驰信号，返回None
        if not divergence_signals:
            return None
        
        # 计算背驰得分
        score = self._calculate_score(divergence_signals, df, trend_direction, trend_strength)
        
        # 如果得分低于阈值，返回None
        if score < self.threshold:
            return None
            
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'last_close': df.iloc[-1]['close'] if not df.empty else 0,
            'signals': divergence_signals,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'score': score,
            'signal_count': len(divergence_signals)
        }
    
    def _get_stock_name(self, stock_code: str) -> str:
        """获取股票名称"""
        try:
            sql = f"SELECT stock_name FROM stock_basic WHERE stock_code = '{stock_code}'"
            result = self.db_manager.read_sql(sql)
            if not result.empty:
                return result.iloc[0]['stock_name']
            return stock_code
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 名称失败: {str(e)}")
            return stock_code

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # MACD指标
        exp12 = df['close'].ewm(span=self.params['macd_fast'], adjust=False).mean()
        exp26 = df['close'].ewm(span=self.params['macd_slow'], adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['signal'] = df['macd'].ewm(span=self.params['macd_signal'], adjust=False).mean()
        df['hist'] = df['macd'] - df['signal']  # MACD柱状图
        
        # 均线系统（5/13/21/34/55日）
        for ma in [5, 13, 21, 34, 55]:
            df[f'ma{ma}'] = df['close'].rolling(ma).mean()
            
        # 量能指标
        df['vol_ma5'] = df['volume'].rolling(5).mean()
        df['vol_ma10'] = df['volume'].rolling(10).mean()
        
        return df

    def _identify_pivots(self, df: pd.DataFrame) -> pd.Series:
        """识别日线中枢"""
        pivots = pd.Series(0, index=df.index)
        n = self.params['pivot_window']
        
        # 找局部高低点
        highs = argrelextrema(df['high'].values, np.greater_equal, order=n)[0]
        lows = argrelextrema(df['low'].values, np.less_equal, order=n)[0]
        
        # 标记中枢区域（简化版三线段重合）
        for i in range(2, len(df)-2):
            if (i in highs and i-1 in lows and i+1 in lows) or \
               (i in lows and i-1 in highs and i+1 in highs):
                pivot_range = df.iloc[i-1:i+2]
                if (pivot_range['high'].max() - pivot_range['low'].min()) < df['close'].mean() * 0.1:  # 振幅小于10%
                    pivots.iloc[i] = 1
        return pivots
    
    def _get_peak_trough_indices(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        获取价格波峰波谷的索引位置
        
        参数:
            df (pd.DataFrame): 股票数据
            
        返回:
            Tuple[List[int], List[int]]: 高点和低点索引列表
        """
        n = self.params['pivot_window']
        
        # 局部高点（价格创新高）
        highs = list(argrelextrema(df['high'].values, np.greater_equal, order=n)[0])
        
        # 局部低点（价格创新低）
        lows = list(argrelextrema(df['low'].values, np.less_equal, order=n)[0])
        
        return highs, lows
    
    def _detect_divergence(self, df: pd.DataFrame, 
                          highs: List[int], lows: List[int]) -> List[Dict]:
        """
        检测底背驰信号
        
        参数:
            df (pd.DataFrame): 股票数据
            highs (List[int]): 高点索引
            lows (List[int]): 低点索引
            
        返回:
            List[Dict]: 底背驰信号列表
        """
        signals = []
        
        # 底背驰检测 (价格创新低但MACD未创新低)
        for i in range(1, len(lows)):
            if lows[i] - lows[i-1] < self.params['min_trend_len']:
                continue
                
            current_low = lows[i]
            prev_low = lows[i-1]
            
            # 只关注最近指定天数内的背驰信号
            if len(df) - current_low > self.params['recency_days']:
                continue
                
            # 价格创新低
            if df['low'].iloc[current_low] < df['low'].iloc[prev_low]:
                # 但MACD指标未创新低（绿柱状图面积减小）
                current_macd = df['macd'].iloc[current_low]
                prev_macd = df['macd'].iloc[prev_low]
                
                # MACD底背驰判断（新低但MACD绿柱减少）- 放宽条件
                if current_macd < 0:  # 只要MACD为负值
                    # 计算两段MACD柱状图面积之比
                    segment1 = df['hist'].iloc[prev_low-8:prev_low+1]  # 前一段低点附近
                    segment2 = df['hist'].iloc[current_low-8:current_low+1]  # 当前低点附近
                    
                    # 如果当前段绿柱面积小于前一段，考虑为底背驰
                    area1 = abs(segment1[segment1 < 0].sum())
                    area2 = abs(segment2[segment2 < 0].sum())
                    
                    # 量能是否萎缩（放宽条件）
                    vol_decreased = False
                    if area1 > 0 and area2 > 0:
                        vol1 = df['volume'].iloc[prev_low-5:prev_low+1].mean()
                        vol2 = df['volume'].iloc[current_low-5:current_low+1].mean()
                        vol_decreased = vol2 < vol1 * self.params['vol_ratio']
                    
                    # 柱状图面积比较和量能萎缩 - 修正条件必须同时满足
                    if area1 > 0 and area2 > 0 and area2 <= area1 * 1.1 and vol_decreased:
                        signals.append({
                            'date': df.index[current_low].strftime('%Y-%m-%d'),
                            'type': 'bottom_divergence',
                            'price': df['close'].iloc[current_low],
                            'reason': '底部背驰：价格创新低但MACD指标未创新低并且量能萎缩',
                            'confidence': min(85, int(100 * (1 - area2/area1 if area2 < area1 else 0.7)))  # 置信度调整
                        })
        
        return signals[-3:] if signals else []  # 返回最近3个信号

    def _evaluate_trend(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        评估趋势方向和强度（基于均线系统）
        
        参数:
            df (pd.DataFrame): 股票数据
            
        返回:
            Tuple[str, str]: 趋势方向和强度
        """
        last = df.iloc[-1]
        
        # 均线多空排列评估
        ma_rank = sum(last['close'] > last[[f'ma{ma}' for ma in [5,13,21,34,55]]])
        ma5_above_ma13 = last['ma5'] > last['ma13']
        ma13_above_ma21 = last['ma13'] > last['ma21']
        ma21_above_ma55 = last['ma21'] > last['ma55']
        
        # 趋势方向判断
        if ma_rank >= 4 and ma5_above_ma13 and ma13_above_ma21 and ma21_above_ma55:
            direction = '上升'
            strength = '强势' if ma_rank == 5 else '偏强'
        elif ma_rank <= 1 and last['ma5'] < last['ma13'] < last['ma21'] < last['ma55']:
            direction = '下降'
            strength = '强势' if ma_rank == 0 else '偏强'
        elif ma_rank >= 3:
            direction = '上升'
            strength = '弱势'
        elif ma_rank <= 2:
            direction = '下降'
            strength = '弱势'
        else:
            direction = '震荡'
            strength = '中性'
            
        return direction, strength
    
    def _calculate_score(self, signals: List[Dict], df: pd.DataFrame,
                         trend_direction: str, trend_strength: str) -> float:
        """
        计算底背驰信号总得分
        
        参数:
            signals (List[Dict]): 底背驰信号列表
            df (pd.DataFrame): 股票数据
            trend_direction (str): 趋势方向
            trend_strength (str): 趋势强度
            
        返回:
            float: 底背驰得分(0-100)
        """
        if not signals:
            return 0
            
        # 基础信号分数
        base_score = 50  # 降低基础分数
        
        # 最新背驰信号的置信度加分
        latest_signal = signals[-1]
        confidence_score = latest_signal.get('confidence', 0) * 0.3  # 最高加30分
        
        # 底背驰额外加分
        signal_type_score = 15
        # 底背驰+上升趋势加分（若已处于上升趋势说明背驰已经确认）
        if trend_direction == '上升':
            signal_type_score += 15
        
        # 时效性分数（最近发生的背驰信号分数更高）
        recency_score = 0
        days_ago = (df.index[-1] - datetime.strptime(latest_signal['date'], '%Y-%m-%d')).days
        if days_ago <= 3:  # 3天内
            recency_score = 30
        elif days_ago <= 7:  # 7天内
            recency_score = 25
        elif days_ago <= 14:  # 14天内
            recency_score = 20
        elif days_ago <= 21:  # 21天内
            recency_score = 15
        elif days_ago <= 30:  # 30天内
            recency_score = 10
            
        # 多重背驰信号加分
        multiple_signals_score = min(len(signals) * 5, 10)  # 最多10分
        
        # 计算总分
        total_score = base_score + confidence_score + signal_type_score + recency_score + multiple_signals_score
        
        # 限制最高分为100
        return min(round(total_score), 100)

    def run_screening(self, filename: str = None) -> str:
        """
        执行底背驰选股流程
        
        参数:
            filename (str, 可选): 输出文件名
            
        返回:
           str: 保存的文件路径
        """
        logger.info("开始执行缠论底背驰日线选股分析...")
        
        # 获取股票列表
        stocks = self.get_stock_list()
        if stocks.empty:
            logger.error("未能获取股票列表，选股终止")
            return ""
            
        logger.info(f"数据库中共有 {len(stocks)} 只股票")
        
        results = []
        processed_count = 0
        
        # 计算30天前的日期
        today = datetime.now()
        thirty_days_ago = today - timedelta(days=30)
        thirty_days_ago_str = thirty_days_ago.strftime('%Y-%m-%d')
        
        for _, row in stocks.iterrows():
            code = row['stock_code']
            
            try:
                analysis = self.evaluate_stock(code)
                if analysis and analysis['score'] >= self.threshold:
                    # 过滤，只保留底背驰信号的股票
                    recent_bottom_divergence = False
                    for signal in analysis['signals']:
                        if signal['type'] == 'bottom_divergence':
                            # 检查信号日期是否在最近30天内
                            signal_date = datetime.strptime(signal['date'], '%Y-%m-%d')
                            if signal_date >= thirty_days_ago:
                                recent_bottom_divergence = True
                                break
                    
                    if recent_bottom_divergence:
                        results.append(analysis)
                        
                        # 打印底背驰信号
                        latest_signal = [s for s in analysis['signals'] if s['type'] == 'bottom_divergence'][-1]
                        logger.info(f"{code} {analysis['stock_name']}: 底背驰 | 信号日期:{latest_signal['date']} | 价格:{analysis['last_close']:.2f} | 分数:{analysis['score']} | 趋势:{analysis['trend_direction']}{analysis['trend_strength']}")
            except Exception as e:
                logger.error(f"分析 {code} 时出错: {str(e)}")
            
            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(f"已处理 {processed_count} / {len(stocks)} 只股票...")
                
        # 关闭数据库连接
        self.db_manager.close()
        
        # 结果转为DataFrame
        if results:
            # 提取需要的字段到DataFrame
            df_results = pd.DataFrame([
                {
                    'stock_code': r['stock_code'],
                    'stock_name': r['stock_name'],
                    'current_price': r['last_close'],
                    'score': r['score'],
                    'trend_direction': r['trend_direction'] + r['trend_strength'],
                    'signal_count': r['signal_count'],
                    'signal_date': [s for s in r['signals'] if s['type'] == 'bottom_divergence'][-1]['date'],
                    'signal_price': [s for s in r['signals'] if s['type'] == 'bottom_divergence'][-1]['price'],
                    'days_ago': (today - datetime.strptime([s for s in r['signals'] if s['type'] == 'bottom_divergence'][-1]['date'], '%Y-%m-%d')).days,
                    'confidence': [s for s in r['signals'] if s['type'] == 'bottom_divergence'][-1].get('confidence', 0),
                    'signal_reason': [s for s in r['signals'] if s['type'] == 'bottom_divergence'][-1]['reason']
                } for r in results
            ])
            
            # 按信号日期倒序排序（最近的信号排在前面）
            df_results = df_results.sort_values('days_ago').reset_index(drop=True)
            
            # 限制返回数量
            if self.limit and len(df_results) > self.limit:
                df_results = df_results.head(self.limit)
                
            # 保存结果
            self.results = df_results
            filepath = self.save_results(df_results, filename)
            
            # 打印结果概要
            self.print_results(df_results)
            
            return filepath
        else:
            logger.warning("未找到符合条件的最近30天内底背驰的股票")
            return ''

    def process_stocks(self, stock_list, limit=None):
        """
        处理股票列表并寻找底背驰信号
        
        参数:
            stock_list (List): 股票列表
            limit (int, 可选): 限制返回结果数量
            
        返回:
            List[Dict]: 符合条件的股票列表
        """
        # 初始化数据库连接
        db_manager = DatabaseManager()
        
        # 初始化结果列表
        self.results = []
        
        # 获取当前日期，用于筛选近期信号
        current_date = datetime.now()
        days_ago_30 = current_date - timedelta(days=30)
        days_ago_30_str = days_ago_30.strftime('%Y-%m-%d')
        
        processed_count = 0
        total_count = len(stock_list)
        
        logger.info(f"开始筛选底背驰股票，共 {total_count} 只...")
        
        for stock in stock_list:
            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(f"已处理 {processed_count} / {total_count} 只股票...")
            
            stock_code = stock['stock_code']
            
            # 获取每日数据
            daily_data = self.get_daily_data(stock_code, days_ago_30_str, current_date.strftime('%Y-%m-%d'))
            
            if daily_data is None or len(daily_data) < 60:
                continue
                
            # 计算技术指标
            daily_data = self._calculate_features(daily_data)
            
            # 获取波峰波谷点位置
            highs, lows = self._get_peak_trough_indices(daily_data)
            
            # 检测底背驰
            signals = self._detect_divergence(daily_data, highs, lows)
            
            # 过滤最近的信号
            recent_signals = []
            for s in signals:
                signal_date = datetime.strptime(s['date'], '%Y-%m-%d')
                if signal_date >= days_ago_30:
                    recent_signals.append(s)
            
            if recent_signals:
                # 评估趋势方向和强度
                trend_direction, trend_strength = self._evaluate_trend(daily_data)
                
                # 计算背驰得分
                score = self._calculate_score(recent_signals, daily_data, trend_direction, trend_strength)
                
                # 如果得分低于阈值，跳过
                if score < self.threshold:
                    continue
                
                # 获取最新的信号
                latest_signal = recent_signals[-1]
                
                # 计算信号距今天数
                signal_date = datetime.strptime(latest_signal['date'], '%Y-%m-%d')
                days_ago = (current_date - signal_date).days
                
                # 获取股票名称
                stock_name = self._get_stock_name(stock_code)
                
                # 创建结果记录
                result = {
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'current_price': daily_data.iloc[-1]['close'],
                    'score': score,
                    'trend_direction': f"{trend_direction}{trend_strength}",
                    'signal_count': len(recent_signals),
                    'signal_date': latest_signal['date'],
                    'signal_price': latest_signal['price'],
                    'days_ago': days_ago,
                    'confidence': latest_signal.get('confidence', 60),
                    'signal_reason': latest_signal['reason']
                }
                
                self.results.append(result)
                logger.info(
                    f"{result['stock_code']} {result['stock_name']}: 底背驰 | "
                    f"信号日期:{result['signal_date']} | 价格:{result['current_price']:.2f} | "
                    f"分数:{result['score']} | 趋势:{result['trend_direction']}"
                )
        
        # 关闭数据库连接
        db_manager.close()
        
        if not self.results:
            logger.warning(f"没有找到在最近30天内出现底背驰信号的股票。")
            return []
        
        # 按分数和信号日期排序（先分数高的，同等分数下取信号最近的）
        self.results.sort(key=lambda x: (-x['score'], x['days_ago']))
        
        # 按趋势方向分组
        uptrend_stocks = [s for s in self.results if '上升' in s['trend_direction']]
        downtrend_stocks = [s for s in self.results if '下降' in s['trend_direction']]
        
        # 统计分组信息
        logger.info(f"\n--- 底背驰股票分布 ---")
        logger.info(f"上升趋势: {len(uptrend_stocks)}只")
        logger.info(f"下降趋势: {len(downtrend_stocks)}只")
        
        # 限制结果数量
        if limit and limit > 0:
            # 平衡两种趋势方向的股票
            limit_per_group = max(limit // 2, 1)
            balanced_results = []
            
            # 先添加上升趋势的股票（优先高分数）
            balanced_results.extend(uptrend_stocks[:limit_per_group])
            
            # 再添加下降趋势的股票（优先高分数）
            balanced_results.extend(downtrend_stocks[:limit_per_group])
            
            # 如果两组加起来不足limit，用另一组补充
            remaining_slots = limit - len(balanced_results)
            if remaining_slots > 0:
                if len(uptrend_stocks) > limit_per_group:
                    balanced_results.extend(uptrend_stocks[limit_per_group:limit_per_group+remaining_slots])
                elif len(downtrend_stocks) > limit_per_group:
                    balanced_results.extend(downtrend_stocks[limit_per_group:limit_per_group+remaining_slots])
            
            self.results = balanced_results[:limit]
        
        # 打印统计信息
        logger.info(f"\n--- 选股结果 ---")
        logger.info(f"共选出 {len(self.results)} 只符合条件的股票:")
        
        # 创建一个DataFrame来更好地展示结果
        df = pd.DataFrame(self.results)
        if not df.empty:
            # 只显示关键列
            display_df = df[['stock_code', 'stock_name', 'current_price', 'score', 'trend_direction', 'days_ago']]
            # 设置浮点数格式
            pd.set_option('display.float_format', '{:.2f}'.format)
            logger.info(f"\n{display_df.to_string(index=False)}")
        
        return self.results

    def generate_selection(self):
        """
        生成选股结果，处理股票列表并保存结果
        
        返回:
            List[Dict]: 符合条件的股票列表
        """
        # 获取股票列表
        stocks = self.get_stock_list()
        logger.info(f"数据库中共有 {len(stocks)} 只股票")
        
        # 处理股票并获取结果
        results = self.process_stocks(stocks, self.limit)
        
        if not results:
            logger.warning("未找到符合条件的底背驰股票")
            return []
        
        # 进一步分析结果
        
        # 按趋势方向分组统计
        uptrend_count = sum(1 for r in results if '上升' in r['trend_direction'])
        downtrend_count = sum(1 for r in results if '下降' in r['trend_direction'])
        
        # 按强弱势分组统计
        strong_count = sum(1 for r in results if '强势' in r['trend_direction'])
        weak_count = sum(1 for r in results if '弱势' in r['trend_direction'])
        
        # 按信号日期分组统计
        recent_3d = sum(1 for r in results if r['days_ago'] <= 3)
        recent_7d = sum(1 for r in results if 3 < r['days_ago'] <= 7)
        recent_14d = sum(1 for r in results if 7 < r['days_ago'] <= 14)
        recent_30d = sum(1 for r in results if 14 < r['days_ago'] <= 30)
        
        # 打印详细分组统计信息
        logger.info(f"\n--- 趋势统计 ---")
        logger.info(f"上升趋势: {uptrend_count}只, 下降趋势: {downtrend_count}只")
        logger.info(f"强势: {strong_count}只, 弱势: {weak_count}只")
        logger.info(f"\n--- 时效性统计 ---")
        logger.info(f"3天内: {recent_3d}只, 7天内: {recent_7d}只, 14天内: {recent_14d}只, 30天内: {recent_30d}只")
        
        # 保存结果到CSV文件
        self._save_results_to_csv(results)
        
        return results

    def _save_results_to_csv(self, results):
        """
        将选股结果保存为CSV文件，并按趋势方向分类输出
        
        参数:
            results (List[Dict]): 选股结果列表
        
        返回:
            str: 保存的主文件路径
        """
        if not results:
            return None
            
        # 创建输出目录
        today = datetime.now().strftime('%Y%m%d')
        output_dir = f"output/selector/{today}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 主结果文件路径
        main_file = f"{output_dir}/chanbc_selection_{today}.csv"
        
        # 将结果转为DataFrame
        df = pd.DataFrame(results)
        
        # 保存主结果文件
        df.to_csv(main_file, index=False, encoding='utf-8-sig')
        logger.info(f"选股结果已保存到: {main_file}")
        
        # 按趋势方向分类保存
        uptrend_df = df[df['trend_direction'].str.contains('上升')]
        downtrend_df = df[df['trend_direction'].str.contains('下降')]
        
        # 按强弱势分类保存
        strong_df = df[df['trend_direction'].str.contains('强势')]
        weak_df = df[df['trend_direction'].str.contains('弱势')]
        
        # 按时效性分类保存
        recent_3d_df = df[df['days_ago'] <= 3]
        recent_7d_df = df[df['days_ago'] <= 7]
        
        # 保存分类文件（如果有对应的数据）
        if not uptrend_df.empty:
            uptrend_file = f"{output_dir}/chanbc_uptrend_{today}.csv"
            uptrend_df.to_csv(uptrend_file, index=False, encoding='utf-8-sig')
            logger.info(f"上升趋势股票已保存到: {uptrend_file}")
            
        if not downtrend_df.empty:
            downtrend_file = f"{output_dir}/chanbc_downtrend_{today}.csv"
            downtrend_df.to_csv(downtrend_file, index=False, encoding='utf-8-sig')
            logger.info(f"下降趋势股票已保存到: {downtrend_file}")
            
        if not strong_df.empty:
            strong_file = f"{output_dir}/chanbc_strong_{today}.csv"
            strong_df.to_csv(strong_file, index=False, encoding='utf-8-sig')
            logger.info(f"强势股票已保存到: {strong_file}")
            
        if not weak_df.empty:
            weak_file = f"{output_dir}/chanbc_weak_{today}.csv"
            weak_df.to_csv(weak_file, index=False, encoding='utf-8-sig')
            logger.info(f"弱势股票已保存到: {weak_file}")
            
        if not recent_3d_df.empty:
            recent_3d_file = f"{output_dir}/chanbc_recent3d_{today}.csv"
            recent_3d_df.to_csv(recent_3d_file, index=False, encoding='utf-8-sig')
            logger.info(f"3天内信号股票已保存到: {recent_3d_file}")
            
        if not recent_7d_df.empty:
            recent_7d_file = f"{output_dir}/chanbc_recent7d_{today}.csv"
            recent_7d_df.to_csv(recent_7d_file, index=False, encoding='utf-8-sig')
            logger.info(f"7天内信号股票已保存到: {recent_7d_file}")
        
        return main_file
