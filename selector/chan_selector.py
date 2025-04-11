# -*- coding: utf-8 -*-
"""缠论选股模块"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from typing import Dict, List, Optional

from selector.base_selector import BaseSelector
from utils.logger import get_logger
from utils.db_manager import DatabaseManager

# 创建日志记录器
logger = get_logger(__name__)

class ChanSelector(BaseSelector):
    """缠论日线选股系统"""
    
    def __init__(self, days=250, threshold=1, limit=50):
        """
        初始化缠论选股器
        
        参数:
            days (int, 可选): 回溯数据天数，默认250天
            threshold (int, 可选): 最低信号数量阈值，默认1个
            limit (int, 可选): 限制结果数量，默认50只
        """
        super().__init__(days, threshold, limit)
        self.db_manager = DatabaseManager()
        
        # 日线专用参数（根据缠论日线操作标准设置）
        self.params = {
            'pivot_window': 5,       # 中枢识别窗口
            'macd_fast': 12,         # MACD快线周期
            'macd_slow': 26,         # MACD慢线周期
            'vol_ratio': 1.5,        # 量能放大阈值
            'min_vol': 1e6           # 最小成交量（股）
        }

    def get_daily_data(self, stock_code, start_date, end_date):
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
        SELECT trade_date as date, open, high, low, close, volume 
        FROM stock_daily 
        WHERE stock_code='{stock_code}' AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
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
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 数据时出错: {str(e)}")
            return pd.DataFrame()

    def evaluate_stock(self, stock_code: str) -> Optional[Dict]:
        """
        评估单只股票
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            Optional[Dict]: 评估结果字典，如果无法评估则返回None
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=self.days)).strftime('%Y-%m-%d')
        
        df = self.get_daily_data(stock_code, start_date, end_date)
        if len(df) < 30:  # 至少需要30个交易日数据
            return None
        
        # 特征计算
        df = self._calculate_features(df)
        
        # 中枢识别（日线级别）
        df['pivots'] = self._identify_pivots(df)
        
        # 买卖信号检测
        signals = self._generate_signals(df)
        
        # 获取股票名称
        stock_name = self._get_stock_name(stock_code)
        
        # 趋势评估
        trend_strength = self._evaluate_trend(df)
        
        # 计算得分
        score = self._calculate_score(signals, trend_strength)
        
        if len(signals) < self.threshold:
            return None
            
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'last_close': df.iloc[-1]['close'] if not df.empty else 0,
            'signals': signals,
            'trend_strength': trend_strength,
            'score': score,
            'signal_count': len(signals)
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

    def _calculate_features(self, df):
        """计算技术指标"""
        # MACD指标
        exp12 = df['close'].ewm(span=self.params['macd_fast'], adjust=False).mean()
        exp26 = df['close'].ewm(span=self.params['macd_slow'], adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 均线系统（5/13/21/34/55日）
        for ma in [5, 13, 21, 34, 55]:
            df[f'ma{ma}'] = df['close'].rolling(ma).mean()
            
        # 量能指标
        df['vol_ma5'] = df['volume'].rolling(5).mean()
        return df

    def _identify_pivots(self, df):
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

    def _generate_signals(self, df):
        """生成买卖信号"""
        signals = []
        
        # 第一类买点：趋势背驰
        for i in range(2, len(df)-1):
            if df['pivots'].iloc[i] and \
               df['macd'].iloc[i] < 0 and df['signal'].iloc[i] < 0 and \
               df['macd'].iloc[i] > df['macd'].iloc[i-1] and \
               df['volume'].iloc[i] > df['vol_ma5'].iloc[i] * self.params['vol_ratio']:
                signals.append({
                    'date': df.index[i].strftime('%Y-%m-%d'),
                    'type': 'buy1',
                    'price': df['close'].iloc[i],
                    'reason': '日线底背驰+量能放大'
                })
        
        # 第二类买点：回调不破前低
        for i in range(3, len(df)):
            if df['close'].iloc[i] > df['ma21'].iloc[i] and \
               df['close'].iloc[i-1] < df['ma21'].iloc[i-1] and \
               df['low'].iloc[i] > df['low'].iloc[i-2]:
                signals.append({
                    'date': df.index[i].strftime('%Y-%m-%d'),
                    'type': 'buy2',
                    'price': df['close'].iloc[i],
                    'reason': '21日均线回踩不破前低'
                })
        
        return signals[-3:] if signals else []  # 返回最近3个信号

    def _evaluate_trend(self, df):
        """评估趋势强度（基于均线系统）"""
        last = df.iloc[-1]
        ma_rank = sum(last['close'] > last[[f'ma{ma}' for ma in [5,13,21,34,55]]])
        
        if last['close'] > last['ma55']:
            return '强势' if ma_rank >=4 else '震荡偏强'
        elif last['close'] > last['ma21']:
            return '弱势反弹' if ma_rank >=3 else '下跌中继'
        else:
            return '极弱'
    
    def _calculate_score(self, signals, trend_strength):
        """计算总得分"""
        # 信号得分
        signal_score = len(signals) * 10
        
        # 趋势得分
        trend_scores = {
            '强势': 50,
            '震荡偏强': 30,
            '弱势反弹': 20,
            '下跌中继': 10,
            '极弱': 0
        }
        trend_score = trend_scores.get(trend_strength, 0)
        
        # 信号类型得分
        type_scores = 0
        for signal in signals:
            if signal['type'] == 'buy1':
                type_scores += 15
            elif signal['type'] == 'buy2':
                type_scores += 10
        
        return signal_score + trend_score + type_scores

    def run_screening(self) -> pd.DataFrame:
        """
        执行选股流程
        
        返回:
            pd.DataFrame: 选股结果数据框
        """
        logger.info("开始执行缠论日线选股分析...")
        
        # 获取股票列表
        stocks = self.get_stock_list()
        if stocks.empty:
            logger.error("未能获取股票列表，选股终止")
            return pd.DataFrame()
            
        logger.info(f"数据库中共有 {len(stocks)} 只股票")
        
        results = []
        processed_count = 0
        
        for _, row in stocks.iterrows():
            code = row['stock_code']
            
            try:
                analysis = self.evaluate_stock(code)
                if analysis and analysis['signal_count'] >= self.threshold:
                    results.append(analysis)
                    
                    # 打印选股信号
                    latest_signal = analysis['signals'][-1] if analysis['signals'] else None
                    if latest_signal:
                        logger.info(f"{code} {analysis['stock_name']}: {latest_signal['type']} | 价格:{analysis['last_close']:.2f} | 趋势:{analysis['trend_strength']}")
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
                    'trend_strength': r['trend_strength'],
                    'signal_count': r['signal_count'],
                    'latest_signal': r['signals'][-1]['type'] if r['signals'] else '',
                    'signal_date': r['signals'][-1]['date'] if r['signals'] else '',
                    'signal_price': r['signals'][-1]['price'] if r['signals'] else 0,
                    'signal_reason': r['signals'][-1]['reason'] if r['signals'] else ''
                } for r in results
            ])
            
            # 按分数排序
            df_results = df_results.sort_values('score', ascending=False).reset_index(drop=True)
            
            # 限制返回数量
            if self.limit and len(df_results) > self.limit:
                df_results = df_results.head(self.limit)
                
            # 保存结果
            self.results = df_results
            self.save_results(df_results)
            
            return df_results
        else:
            logger.warning("未找到符合条件的股票")
            return pd.DataFrame() 