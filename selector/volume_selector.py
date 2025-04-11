# -*- coding: utf-8 -*-
"""量能选股模块"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from selector.base_selector import BaseSelector
from utils.logger import get_logger
from utils.db_manager import DatabaseManager

# 创建日志记录器
logger = get_logger(__name__)

class VolumeSelector(BaseSelector):
    """量能选股器，基于成交量和主力行为特征选股"""
    
    def __init__(self, days=120, threshold=2, limit=30):
        """
        初始化量能选股器
        
        参数:
            days (int, 可选): 回溯数据天数，默认120天
            threshold (float, 可选): 选股分数阈值，默认2分
            limit (int, 可选): 限制结果数量，默认30只
        """
        super().__init__(days, threshold, limit)
        self.db_manager = DatabaseManager()
        
    def fetch_daily_data(self, stock_code: str) -> pd.DataFrame:
        """
        获取股票日线数据
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            pd.DataFrame: 股票日线数据
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=self.days)).strftime('%Y-%m-%d')
        
        sql = f"""
            SELECT trade_date, open, high, low, close, volume, amount 
            FROM stock_daily 
            WHERE stock_code = '{stock_code}' AND trade_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY trade_date
        """
        try:
            df = self.db_manager.read_sql(sql)
            if df.empty:
                logger.warning(f"未能获取到股票 {stock_code} 的数据")
                return pd.DataFrame()
                
            # 确保数值列为float类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_columns:
                df[col] = df[col].astype(float)
                
            # 设置日期索引
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 数据时出错: {str(e)}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        参数:
            df (pd.DataFrame): 股票日线数据
            
        返回:
            pd.DataFrame: 添加了技术指标的数据框
        """
        # 均线系统
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma21'] = df['close'].rolling(window=21).mean()
        df['ma65'] = df['close'].rolling(window=65).mean()
        
        # 量能指标
        df['vol_ma5'] = df['volume'].rolling(window=5).mean()
        df['vol_ma21'] = df['volume'].rolling(window=21).mean()
        
        # 黄金分割目标位计算
        low_idx = df['close'].idxmin()
        if low_idx is not None and len(df.loc[low_idx:]) > 1:
            high_idx = df.loc[low_idx:, 'close'].idxmax()
            base_low = df.at[low_idx, 'close']
            base_high = df.at[high_idx, 'close']
            
            df['target_1.618'] = (base_high - base_low) * 1.618 + base_low
            df['target_2.618'] = (base_high - base_low) * 2.618 + base_low
        else:
            df['target_1.618'] = np.nan
            df['target_2.618'] = np.nan
        
        # OBV能量潮
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # VR指标
        up_volume = df[df['close'] > df['close'].shift(1)]['volume']
        down_volume = df[df['close'] < df['close'].shift(1)]['volume']
        up_sum = up_volume.rolling(window=21).sum()
        down_sum = down_volume.rolling(window=21).sum()
        
        # 防止除零错误
        denominator = (down_sum + 0.5 * (df['volume'].rolling(window=21).sum() - up_sum - down_sum))
        df['vr'] = np.where(
            denominator != 0,
            (up_sum + 0.5 * (df['volume'].rolling(window=21).sum() - up_sum - down_sum)) / denominator * 100,
            np.nan
        )
        
        return df

    def analyze_volume_pattern(self, df: pd.DataFrame) -> List[str]:
        """
        分析量价形态
        
        参数:
            df (pd.DataFrame): 带技术指标的股票数据
            
        返回:
            List[str]: 量价形态信号列表
        """
        signals = []
        
        if len(df) < 5:
            return signals
            
        last_5 = df.iloc[-5:]
        
        # 顺势量判断
        up_days = last_5[last_5['close'] > last_5['close'].shift(1)]
        down_days = last_5[last_5['close'] < last_5['close'].shift(1)]
        
        if len(up_days) > 0 and len(down_days) > 0:
            up_vol_avg = up_days['volume'].mean()
            down_vol_avg = down_days['volume'].mean()
            
            if up_vol_avg > 1.8 * down_vol_avg:
                signals.append("健康顺势量")
            elif down_vol_avg > up_vol_avg:
                signals.append("警惕逆势量")
        
        # 量能潮分析
        if len(df) >= 21 and 'obv' in df.columns and df['obv'].iloc[-1] > df['obv'].iloc[-21] and df['close'].iloc[-1] > df['close'].iloc[-21]:
            signals.append("OBV与价格同步上升")
        
        # 凹洞量识别
        if len(df) >= 2 and 'vol_ma21' in df.columns and df['volume'].iloc[-1] < df['vol_ma21'].iloc[-1] * 0.6 and df['close'].iloc[-1] > df['close'].iloc[-2]:
            signals.append("凹洞量止跌信号")
            
        return signals

    def detect_main_force_action(self, df: pd.DataFrame) -> List[str]:
        """
        识别主力行为
        
        参数:
            df (pd.DataFrame): 带技术指标的股票数据
            
        返回:
            List[str]: 主力行为信号列表
        """
        signals = []
        
        if len(df) < 6:
            return signals
        
        # 破底型进货识别
        if len(df) >= 5 and 'vol_ma21' in df.columns and df['close'].iloc[-1] > df['close'].iloc[-5] and df['volume'].iloc[-5] > df['vol_ma21'].iloc[-5] * 2:
            signals.append("破底型进货嫌疑")
        
        # 盘跌型出货识别
        if len(df) >= 6 and 'vol_ma21' in df.columns and df['close'].iloc[-3] < df['close'].iloc[-6] and df['close'].iloc[-1] < df['close'].iloc[-3] and df['volume'].iloc[-1] > df['vol_ma21'].iloc[-1]:
            signals.append("盘跌型出货嫌疑")
        
        # 头部量识别
        if 'target_2.618' in df.columns and pd.notna(df['target_2.618'].iloc[-1]) and 'vol_ma21' in df.columns and df['close'].iloc[-1] > df['target_2.618'].iloc[-1] * 0.95 and df['volume'].iloc[-1] > df['vol_ma21'].iloc[-1] * 2:
            signals.append("目标位满足+异常大量")
            
        return signals

    def generate_trading_signals(self, df: pd.DataFrame) -> Dict:
        """
        生成交易信号
        
        参数:
            df (pd.DataFrame): 带技术指标的股票数据
            
        返回:
            Dict: 交易信号字典
        """
        signals = {
            'volume_pattern': self.analyze_volume_pattern(df),
            'main_force': self.detect_main_force_action(df),
            'technical': []
        }
        
        # 技术指标信号
        if len(df) >= 2 and 'ma5' in df.columns and 'ma21' in df.columns and df['ma5'].iloc[-1] > df['ma21'].iloc[-1] and df['ma5'].iloc[-2] <= df['ma21'].iloc[-2]:
            signals['technical'].append("5日均线上穿21日均线")
        
        if 'vr' in df.columns and pd.notna(df['vr'].iloc[-1]) and df['vr'].iloc[-1] < 40:
            signals['technical'].append(f"VR指标低位({df['vr'].iloc[-1]:.1f})")
            
        if 'target_1.618' in df.columns and pd.notna(df['target_1.618'].iloc[-1]) and df['close'].iloc[-1] > df['target_1.618'].iloc[-1]:
            signals['technical'].append(f"突破1.618目标位({df['target_1.618'].iloc[-1]:.2f})")
        
        return signals

    def evaluate_stock(self, stock_code: str) -> Optional[Dict]:
        """
        评估单只股票
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            Optional[Dict]: 评估结果字典，如果无法评估则返回None
        """
        try:
            df = self.fetch_daily_data(stock_code)
            if len(df) < 65:  # 数据不足
                return None
                
            df = self.calculate_technical_indicators(df)
            signals = self.generate_trading_signals(df)
            
            # 获取股票名称
            stock_name = self._get_stock_name(stock_code)
            
            # 综合评估
            score = 0
            positive_signals = []
            warning_signals = []
            
            for signal in signals['volume_pattern']:
                if signal in ["健康顺势量", "OBV与价格同步上升", "凹洞量止跌信号"]:
                    score += 1
                    positive_signals.append(signal)
                else:
                    score -= 1
                    warning_signals.append(signal)
                    
            for signal in signals['main_force']:
                if "进货" in signal:
                    score += 2
                    positive_signals.append(signal)
                else:
                    score -= 2
                    warning_signals.append(signal)
                    
            for signal in signals['technical']:
                if "上穿" in signal or "低位" in signal:
                    score += 1
                    positive_signals.append(signal)
                elif "目标位" in signal:
                    score -= 1
                    warning_signals.append(signal)
            
            if score < self.threshold:
                return None
                
            ma21_value = df['ma21'].iloc[-1] if 'ma21' in df.columns and pd.notna(df['ma21'].iloc[-1]) else None
            target_value = df['target_1.618'].iloc[-1] if 'target_1.618' in df.columns and pd.notna(df['target_1.618'].iloc[-1]) else None
            vr_value = df['vr'].iloc[-1] if 'vr' in df.columns and pd.notna(df['vr'].iloc[-1]) else None
                
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'current_price': df['close'].iloc[-1],
                'ma21': ma21_value,
                'target_1.618': target_value,
                'vr': vr_value,
                'score': score,
                'positive_signals': positive_signals,
                'warnings': warning_signals
            }
            
        except Exception as e:
            logger.error(f"分析 {stock_code} 时出错: {str(e)}")
            return None
    
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

    def run_screening(self) -> pd.DataFrame:
        """
        执行选股流程
        
        返回:
            pd.DataFrame: 选股结果数据框
        """
        logger.info("开始执行量能选股分析...")
        
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
                if analysis and analysis['score'] >= self.threshold:
                    # 转换列表为字符串用于展示
                    analysis['positive_signals_str'] = " | ".join(analysis['positive_signals'])
                    analysis['warnings_str'] = " | ".join(analysis['warnings']) if analysis['warnings'] else "无"
                    results.append(analysis)
                    
                    # 记录选股结果
                    logger.info(f"{code} {analysis['stock_name']}: 得分:{analysis['score']} | 价格:{analysis['current_price']:.2f}")
                    
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
                    'current_price': r['current_price'],
                    'ma21': r['ma21'],
                    'target_1.618': r['target_1.618'],
                    'vr': r['vr'],
                    'score': r['score'],
                    'positive_signals': r['positive_signals_str'],
                    'warnings': r['warnings_str']
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


if __name__ == '__main__':
    # 直接运行测试
    selector = VolumeSelector()
    results = selector.run_screening()
    
    # 保存结果
    if not results.empty:
        selector.save_results(
            results, 
            f"volume_selection_{datetime.now().strftime('%Y%m%d')}.csv"
        ) 