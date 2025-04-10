# -*- coding: utf-8 -*-
"""基础分析器模块"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# 导入配置和工具
from config import ANALYZER_CONFIG, PATH_CONFIG
from utils.logger import get_logger
from utils.indicators import calculate_basic_indicators

# 创建日志记录器
logger = get_logger(__name__)

class BaseAnalyzer:
    """分析器基类，提供基本的数据获取和处理功能"""
    
    def __init__(self, stock_code: str, stock_name: str = None, end_date: Union[str, datetime] = None, days: int = None):
        """
        初始化基础分析器
        
        参数:
            stock_code (str): 股票代码
            stock_name (str, 可选): 股票名称，如不提供则使用股票代码
            end_date (str 或 datetime, 可选): 结束日期，默认为当前日期
            days (int, 可选): 回溯天数，默认使用配置中的默认值
        """
        self.stock_code = stock_code
        self.stock_name = stock_name if stock_name else stock_code
        
        # 处理end_date参数
        if end_date:
            if isinstance(end_date, str):
                try:
                    self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
                except ValueError:
                    logger.warning(f"日期格式错误 '{end_date}'，使用当前日期代替")
                    self.end_date = datetime.now()
            else:
                self.end_date = end_date
        else:
            self.end_date = datetime.now()
            
        # 设置回溯天数
        self.days = days if days is not None else ANALYZER_CONFIG.get('default_days', 365)
        
        # 计算开始日期
        self.start_date = self.end_date - timedelta(days=self.days)
        
        # 日期字符串格式
        self.end_date_str = self.end_date.strftime('%Y%m%d')
        self.start_date_str = self.start_date.strftime('%Y%m%d')
        
        # 初始化数据和结果属性
        self.data = pd.DataFrame()
        self.analysis_result = {}
    
    def get_stock_name(self) -> str:
        """
        获取股票名称
        
        返回:
            str: 股票名称
        """
        if self.stock_name and self.stock_name != self.stock_code:
            return self.stock_name
            
        try:
            from utils.akshare_api import AkshareAPI
            
            akshare = AkshareAPI()
            name = akshare.get_stock_name(self.stock_code)
            
            if name and name != self.stock_code:
                self.stock_name = name
                return name
            else:
                logger.warning(f"未能获取股票 {self.stock_code} 的名称")
                return self.stock_code
                
        except Exception as e:
            logger.error(f"获取股票名称时出错: {str(e)}")
            return self.stock_code
    
    def get_stock_daily_data(self) -> pd.DataFrame:
        """
        从数据库获取股票日线数据，并计算技术指标
        
        返回:
            pd.DataFrame: 股票日线数据（含技术指标）
        """
        try:
            from data.db_manager import DatabaseManager
            
            db = DatabaseManager()
            
            # 构建SQL查询
            sql = f"""
                SELECT * FROM stock_daily
                WHERE stock_code = '{self.stock_code}'
                AND trade_date BETWEEN '{self.start_date.strftime('%Y-%m-%d')}' AND '{self.end_date.strftime('%Y-%m-%d')}'
                ORDER BY trade_date
            """
            
            # 执行查询
            df = db.read_sql(sql)
            
            if not df.empty:
                logger.info(f"从数据库成功获取 {len(df)} 条 {self.stock_code} 数据")
                
                # 使用工具函数计算技术指标
                df = calculate_basic_indicators(df)
                
                return df
            else:
                logger.warning(f"数据库中未找到股票 {self.stock_code} 的数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"从数据库获取股票数据时出错: {str(e)}")
            return pd.DataFrame()
    
    def save_analysis_result(self) -> bool:
        """
        保存分析结果到数据库
        
        返回:
            bool: 是否成功保存
        """
        if not self.analysis_result:
            logger.warning("没有分析结果可保存")
            return False
            
        try:
            from data.db_manager import DatabaseManager
            
            db = DatabaseManager()
            
            # 提取分析结果中的必要字段
            analyzer_type = self.__class__.__name__
            analysis_date = datetime.now().strftime('%Y-%m-%d')
            analysis_content = self.analysis_result.get('description', '')
            chart_path = self.analysis_result.get('chart_path', '')
            
            # 准备要保存的数据
            result_df = pd.DataFrame({
                'analysis_date': [analysis_date],
                'analyzer_type': [analyzer_type],
                'stock_code': [self.stock_code],
                'stock_name': [self.stock_name],
                'analysis_content': [analysis_content],
                'chart_path': [chart_path],
                'additional_info': [str(self.analysis_result)]
            })
            
            # 保存到数据库
            success = db.to_sql(result_df, 'stock_analysis_report', 'append')
            
            if success:
                logger.info(f"分析结果已保存到数据库: {self.stock_code} ({analyzer_type})")
            else:
                logger.error("保存分析结果到数据库失败")
                
            return success
                
        except Exception as e:
            logger.error(f"保存分析结果时出错: {str(e)}")
            return False
    
    def fetch_data(self) -> bool:
        """
        获取股票数据
        
        返回:
            bool: 是否成功获取数据
        """
        # 在子类中实现
        raise NotImplementedError("fetch_data方法需要在子类中实现")
    
    def prepare_data(self) -> bool:
        """
        准备分析数据，计算指标
        
        返回:
            bool: 是否成功准备数据
        """
        # 在子类中实现
        raise NotImplementedError("prepare_data方法需要在子类中实现")
    
    def run_analysis(self) -> Dict:
        """
        执行分析流程
        
        返回:
            Dict: 分析结果
        """
        # 在子类中实现
        raise NotImplementedError("run_analysis方法需要在子类中实现") 