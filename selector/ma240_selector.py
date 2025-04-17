from datetime import datetime, timedelta
from typing import Optional, Dict
from venv import logger

import pandas as pd
from data.db_manager import DatabaseManager
from selector.base_selector import BaseSelector

class Ma240Selector(BaseSelector):
    """
    240日均线选股系统, 这个数据起码是两年以上才够用
    股价穿过240日均线，然后回踩 240 日均线
    参数:
        days (int, 可选): 回溯数据天数，默认240天
        threshold (int, 可选): 最低信号数量阈值，默认1个
        limit (int, 可选): 限制结果数量，默认50只
    """
    
    def __init__(self, days=365 * 2, threshold=None, limit=None):
        super().__init__(days, threshold, limit)
        self.db_manager = DatabaseManager()

    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取股票日线数据
        
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
            df['ma240'] = df['close'].rolling(window=240).mean()
            return df
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 日线数据失败: {e}")
            return pd.DataFrame()
        
    def evaluate_stock(self, stock_code: str) -> True:
        """
        评估单只股票
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            bool: 评估结果 True 或 False
        """
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=self.days)).strftime('%Y-%m-%d')
            # 获取日线数据
            df = self.get_daily_data(stock_code, start_date, end_date)

            # 判断是否存在240日均线
            if 'ma240' not in df.columns:
                logger.warning(f"股票 {stock_code} 缺少240日均线数据")
                return False
            
            # 检查股价是否穿过240日均线
            latest_close = df['close'].iloc[-1]
            latest_ma240 = df['ma240'].iloc[-1]
            prev_close = df['close'].iloc[-2]
            prev_ma240 = df['ma240'].iloc[-2]
            
            # 条件1：当前收盘价低于 MA240
            condition1 = latest_close < latest_ma240

            # 条件2：之前曾突破 MA240（至少有一天收盘价 > MA240）
            condition2 = (df['close'] > df['ma240']).any()
            
            # 条件3：最近是从上方回踩（前一天收盘价 > 前一天 MA240）
            condition3 = prev_close > prev_ma240
            
            return condition1 and condition2 and condition3
        
        except Exception as e:
            logger.error(f"评估股票 {stock_code} 失败: {e}")
            return False
        

    def run_screening(self, filename: str = None) -> str:
        """
        执行选股流程
      
        返回:
            path: 保存的文件路径
        """ 
      
        logger.info("开始执行240日均线选股分析...")

        # 获取股票列表
        stocks = self.get_stock_list()
        if stocks.empty:
            logger.error("未能获取股票列表，选股终止")
            return ''

        # 筛选符合条件的股票
        results = []
        for index, row in stocks.iterrows():
            code = row['stock_code']
            stock_name = row['stock_name']
            
            if index % 100 == 0 and index > 0:
                logger.info(f"已处理 {index} / {len(stocks)} 只股票...")
            try:
                if self.evaluate_stock(code):
                    results.append({
                        'stock_code': code,
                        'stock_name': stock_name
                    })
            except Exception as e:
                logger.error(f"处理股票 {code} 失败: {e}")

            

        # 保存结果
        results = pd.DataFrame(results)
        filepath = self.save_results(results, filename)
        return filepath

