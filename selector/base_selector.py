# -*- coding: utf-8 -*-
"""选股基类模块"""

from abc import ABC, abstractmethod
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Optional

# 导入配置和日志
from config import SELECTOR_CONFIG, PATH_CONFIG
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger(__name__)

class BaseSelector(ABC):
    """选股基类，定义选股接口"""
    
    def __init__(self, days=None, threshold=None, limit=None):
        """
        初始化选股器
        
        参数:
            days (int, 可选): 回溯数据天数
            threshold (float, 可选): 选股分数阈值
            limit (int, 可选): 限制结果数量
        """
        self.days = days or SELECTOR_CONFIG['default_days']
        self.threshold = threshold or SELECTOR_CONFIG['default_threshold']
        self.limit = limit or SELECTOR_CONFIG['default_limit']
        self.results = pd.DataFrame()
        
    @abstractmethod
    def evaluate_stock(self, stock_code: str) -> Optional[Dict]:
        """
        评估单只股票
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            Optional[Dict]: 评估结果字典，如果无法评估则返回None
        """
        pass
        
    @abstractmethod
    def run_screening(self) -> pd.DataFrame:
        """
        执行选股流程
        
        返回:
            pd.DataFrame: 选股结果数据框
        """
        pass
    
    def save_results(self, results: pd.DataFrame, filename=None) -> str:
        """
        保存选股结果
        
        参数:
            results (pd.DataFrame): 选股结果数据框
            filename (str, 可选): 文件名，默认使用日期
            
        返回:
            str: 保存的文件路径
        """
        # 确保目录存在
        if not os.path.exists(PATH_CONFIG['selector_path']):
            os.makedirs(PATH_CONFIG['selector_path'])
            
        # 生成文件名
        if not filename:
            filename = f"stock_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        # 完整路径
        filepath = os.path.join(PATH_CONFIG['selector_path'], filename)
        
        # 保存结果
        results.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"选股结果已保存至: {filepath}")
        
        return filepath
        
    def print_results(self, results: pd.DataFrame = None) -> None:
        """
        打印选股结果
        
        参数:
            results (pd.DataFrame, 可选): 选股结果数据框，默认使用self.results
        """  
        if results.empty:
            logger.warning("没有找到符合条件的股票")
            return
            
        # 格式化打印结果
        print("\n--- 选股结果 ---")
        print(f"共选出 {len(results)} 只符合条件的股票:")
        
        # 选择要显示的列
        display_columns = []
        for col in ['stock_code', 'stock_name', 'current_price', 'score', 'positive_signals', 'warnings']:
            if col in results.columns:
                display_columns.append(col)
                
        # 打印结果
        print(results[display_columns].to_string(index=False))
        
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        返回:
            pd.DataFrame: 股票列表数据框
        """
        from data.db_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        
        try:
            sql = "SELECT stock_code, stock_name FROM stock_basic"
            stocks = db_manager.read_sql(sql)
            logger.info(f"从数据库获取到 {len(stocks)} 只股票")
            return stocks
        except Exception as e:
            logger.error(f"获取股票列表失败: {str(e)}")
            return pd.DataFrame()
        finally:
            db_manager.close() 