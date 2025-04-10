# -*- coding: utf-8 -*-
"""量能选股器模块"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# 导入选股基类和日志
from selector.base_selector import BaseSelector
from data.db_manager import DatabaseManager
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger(__name__)

class VolumeSelector(BaseSelector):
    """量能选股器，基于量能和价格关系进行选股"""
    
    def __init__(self, days=None, threshold=None, limit=None):
        """初始化量能选股器"""
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
            SELECT d.trade_date, d.open, d.high, d.low, d.close, d.volume, d.amount,
                  i.ma5, i.ma10, i.ma20, i.vol_ma5, i.vol_ma10
            FROM stock_daily d
            LEFT JOIN stock_daily_indicator i ON d.stock_code = i.stock_code AND d.trade_date = i.trade_date
            WHERE d.stock_code = '{stock_code}' AND d.trade_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY d.trade_date
        """
        
        df = self.db_manager.read_sql(sql)
        
        if df.empty:
            logger.warning(f"未找到股票 {stock_code} 的历史数据")
            return pd.DataFrame()
            
        # 转换日期列
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 确保数值列为float类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                          'ma5', 'ma10', 'ma20', 'vol_ma5', 'vol_ma10']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        参数:
            df (pd.DataFrame): 股票日线数据
            
        返回:
            pd.DataFrame: 添加技术指标后的数据框
        """
        if df.empty or len(df) < 21:  # 至少需要21天数据
            return df
            
        # 计算均线(如果数据库中没有)
        if 'ma5' not in df.columns or df['ma5'].isna().all():
            df['ma5'] = df['close'].rolling(window=5).mean()
            
        if 'ma21' not in df.columns:
            df['ma21'] = df['close'].rolling(window=21).mean()
            
        if 'ma65' not in df.columns:
            df['ma65'] = df['close'].rolling(window=65).mean()
            
        # 计算成交量均线(如果数据库中没有)
        if 'vol_ma5' not in df.columns or df['vol_ma5'].isna().all():
            df['vol_ma5'] = df['volume'].rolling(window=5).mean()
            
        if 'vol_ma21' not in df.columns:
            df['vol_ma21'] = df['volume'].rolling(window=21).mean()
        
        # 计算黄金分割目标位
        low_price = df['close'].min()
        high_price = df['close'].max()
        df['target_1618'] = (high_price - low_price) * 1.618 + low_price
        df['target_2618'] = (high_price - low_price) * 2.618 + low_price
        
        # 计算OBV能量潮
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # 计算VR指标
        up_volume = df[df['close'] > df['close'].shift(1)]['volume']
        down_volume = df[df['close'] < df['close'].shift(1)]['volume']
        up_sum = up_volume.rolling(window=21).sum()
        down_sum = down_volume.rolling(window=21).sum()
        
        # 避免除零错误
        divisor = (down_sum + 0.5 * (df['volume'].rolling(window=21).sum() - up_sum - down_sum))
        divisor = divisor.replace(0, np.nan)  # 将0替换为NaN
        
        df['vr'] = (up_sum + 0.5 * (df['volume'].rolling(window=21).sum() - up_sum - down_sum)) / divisor * 100
        df['vr'] = df['vr'].fillna(100)  # 将NaN替换为中性值100
        
        return df
        
    def analyze_volume_pattern(self, df: pd.DataFrame) -> List[str]:
        """
        分析量价形态
        
        参数:
            df (pd.DataFrame): 技术指标数据框
            
        返回:
            List[str]: 量价形态分析结果
        """
        if df.empty or len(df) < 5:
            return []
            
        signals = []
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
        if len(df) > 21 and 'obv' in df.columns:
            if df['obv'].iloc[-1] > df['obv'].iloc[-21] and df['close'].iloc[-1] > df['close'].iloc[-21]:
                signals.append("OBV与价格同步上升")
        
        # 凹洞量识别
        if 'vol_ma21' in df.columns and len(df) > 1:
            if (df['volume'].iloc[-1] < df['vol_ma21'].iloc[-1] * 0.6 and 
                df['close'].iloc[-1] > df['close'].iloc[-2]):
                signals.append("凹洞量止跌信号")
                
        return signals
        
    def detect_main_force_action(self, df: pd.DataFrame) -> List[str]:
        """
        识别主力行为
        
        参数:
            df (pd.DataFrame): 技术指标数据框
            
        返回:
            List[str]: 主力行为分析结果
        """
        if df.empty or len(df) < 6:
            return []
            
        signals = []
        
        # 破底型进货识别
        if len(df) >= 5 and 'vol_ma21' in df.columns:
            if (df['close'].iloc[-1] > df['close'].iloc[-5] and 
                df['volume'].iloc[-5] > df['vol_ma21'].iloc[-5] * 2):
                signals.append("破底型进货嫌疑")
        
        # 盘跌型出货识别
        if len(df) >= 6 and 'vol_ma21' in df.columns:
            if (df['close'].iloc[-3] < df['close'].iloc[-6] and 
                df['close'].iloc[-1] < df['close'].iloc[-3] and
                df['volume'].iloc[-1] > df['vol_ma21'].iloc[-1]):
                signals.append("盘跌型出货嫌疑")
        
        # 头部量识别
        if 'target_2618' in df.columns and 'vol_ma21' in df.columns:
            if (df['close'].iloc[-1] > df['target_2618'].iloc[-1] * 0.95 and
                df['volume'].iloc[-1] > df['vol_ma21'].iloc[-1] * 2):
                signals.append("目标位满足+异常大量")
                
        return signals
        
    def generate_trading_signals(self, df: pd.DataFrame) -> Dict:
        """
        生成交易信号
        
        参数:
            df (pd.DataFrame): 技术指标数据框
            
        返回:
            Dict: 交易信号字典
        """
        if df.empty:
            return {
                'volume_pattern': [],
                'main_force': [],
                'technical': []
            }
        
        signals = {
            'volume_pattern': self.analyze_volume_pattern(df),
            'main_force': self.detect_main_force_action(df),
            'technical': []
        }
        
        # 技术指标信号
        if 'ma5' in df.columns and 'ma21' in df.columns and len(df) >= 2:
            if df['ma5'].iloc[-1] > df['ma21'].iloc[-1] and df['ma5'].iloc[-2] <= df['ma21'].iloc[-2]:
                signals['technical'].append("5日均线上穿21日均线")
        
        if 'vr' in df.columns:
            if df['vr'].iloc[-1] < 40:
                signals['technical'].append(f"VR指标低位({df['vr'].iloc[-1]:.1f})")
                
        if 'target_1618' in df.columns:
            if df['close'].iloc[-1] > df['target_1618'].iloc[-1]:
                signals['technical'].append(f"突破1.618目标位({df['target_1618'].iloc[-1]:.2f})")
        
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
            if df.empty or len(df) < 65:  # 数据不足
                return None
                
            df = self.calculate_technical_indicators(df)
            signals = self.generate_trading_signals(df)
            
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
            
            return {
                'stock_code': stock_code,
                'current_price': df['close'].iloc[-1],
                'ma21': df['ma21'].iloc[-1] if 'ma21' in df.columns else None,
                'target_1618': df['target_1618'].iloc[-1] if 'target_1618' in df.columns else None,
                'vr': df['vr'].iloc[-1] if 'vr' in df.columns else None,
                'score': score,
                'positive_signals': " | ".join(positive_signals) if positive_signals else "无",
                'warnings': " | ".join(warning_signals) if warning_signals else "无"
            }
            
        except Exception as e:
            logger.error(f"分析{stock_code}时出错: {str(e)}")
            return None
            
    def run_screening(self) -> pd.DataFrame:
        """
        执行选股流程
        
        返回:
            pd.DataFrame: 选股结果数据框
        """
        logger.info("开始执行量能选股分析...")
        
        try:
            # 获取股票列表
            stocks = self.get_stock_list()
            if stocks.empty:
                logger.error("未能获取股票列表，程序退出")
                return pd.DataFrame()
                
            logger.info(f"数据库中共有 {len(stocks)} 只股票")
            results = []
            
            # 批量处理
            batch_size = 20  # 批处理大小
            sleep_time = 2   # 批次间隔(秒)
            
            processed_count = 0
            for i in range(0, len(stocks), batch_size):
                batch = stocks.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    stock_code = row['stock_code']
                    stock_name = row['stock_name']
                    
                    # 评估股票
                    result = self.evaluate_stock(stock_code)
                    if result and result['score'] > self.threshold:  # 超过阈值的股票
                        result['stock_name'] = stock_name
                        results.append(result)
                        
                        # 打印选股信号
                        logger.info(f"发现符合条件股票: {stock_code} {stock_name} 评分:{result['score']} | {result['positive_signals']}")
                    
                    processed_count += 1
                    
                # 每处理一批打印进度
                logger.info(f"已处理 {processed_count} / {len(stocks)} 只股票...")
                
                # 避免频繁查询数据库
                if i + batch_size < len(stocks):
                    time.sleep(sleep_time)
            
            # 处理结果
            if results:
                # 按评分排序
                results.sort(key=lambda x: x['score'], reverse=True)
                
                # 限制结果数量
                if self.limit > 0 and len(results) > self.limit:
                    results = results[:self.limit]
                    
                # 转换为DataFrame
                self.results = pd.DataFrame(results)
                
                # 打印结果
                self.print_results()
                
                logger.info(f"选股完成，共选出 {len(self.results)} 只符合条件的股票")
            else:
                logger.warning("没有找到符合条件的股票")
                self.results = pd.DataFrame()
                
            return self.results
            
        except Exception as e:
            logger.error(f"执行选股过程中出错: {str(e)}")
            return pd.DataFrame()
            
        finally:
            # 关闭数据库连接
            self.db_manager.close()


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