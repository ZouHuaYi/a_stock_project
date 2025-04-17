# -*- coding: utf-8 -*-
"""基础分析器模块"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# 导入配置和工具
from config import ANALYZER_CONFIG, PATH_CONFIG
from utils.llm_api import LLMAPI
from utils.logger import get_logger
from utils.indicators import calculate_basic_indicators, calculate_technical_indicators
from utils.tavily_api import TavilyAPI

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
            end_date (str 或 datetime, 可选): 结束日期，默认为当前日期 格式是 YYYY-MM-DD
            days (int, 可选): 回溯天数，默认使用配置中的默认值
        """
        self.stock_code = stock_code
        self.stock_name = stock_name if stock_name else stock_code
        self.daily_data = pd.DataFrame() # 股票日线数据
        self.indicators = {} # 技术指标
        self.tavily_api = TavilyAPI() # 新闻搜索
        self.llm_api = LLMAPI() # 大模型
        
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
        self.days = days if days is not None else ANALYZER_CONFIG.get('default_days')
        
        # 计算开始日期
        self.start_date = self.end_date - timedelta(days=self.days)
        
        # 日期字符串格式
        self.end_date_str = self.end_date.strftime('%Y%m%d')
        self.start_date_str = self.start_date.strftime('%Y%m%d')
        
        # 初始化数据和结果属性
        self.data = pd.DataFrame()
        self.analysis_result = {}
        
        # 设置统一的保存路径
        self.save_path = PATH_CONFIG.get('analyzer_path')
        os.makedirs(self.save_path, exist_ok=True)
    
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
    
    def get_stock_daily_data(self, period: str = "daily") -> pd.DataFrame:
        """
        从AkShare获取股票日线数据，并计算技术指标
        
        返回:
            pd.DataFrame: 股票日线数据（含技术指标）
        """
        try:
            from utils.akshare_api import AkshareAPI
            
            akshare = AkshareAPI()
            
            # 获取历史数据
            years = self.days // 365 + 1 # 向上取整，确保获取足够的数据
            
            df = akshare.get_stock_history(
                stock_code=self.stock_code,
                period=period,
                years=years,
                adjust="qfq"  # 前复权
            )
            name = akshare.get_stock_name(self.stock_code)
            # 获取的数据有可能比自己需要的时间长
            if isinstance(df, pd.DataFrame) and not df.empty:
                # 按照回溯天数筛选
                if 'trade_date' in df.columns:
                    df = df[(df['trade_date'] >= self.start_date.strftime('%Y-%m-%d')) & 
                           (df['trade_date'] <= self.end_date.strftime('%Y-%m-%d'))]
                    df.set_index('trade_date', inplace=True)
                df['stock_name'] = name
                logger.info(f"成功获取 {len(df)} 条 {self.stock_code} 数据")
                return df
            else:
                logger.warning(f"未找到股票 {self.stock_code} 的数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"从AkShare获取股票数据时出错: {str(e)}")
            return pd.DataFrame()
    
    def prepare_data(self) -> bool:
        """
        准备分析数据，计算指标
        
        返回:
            bool: 是否成功准备数据
        """
        if self.daily_data is None or self.daily_data.empty:
            logger.warning(f"股票{self.stock_code}没有日线数据，请先获取数据")
            return False
        
        try:
            # 计算各种技术指标
            self.daily_data, self.indicators = calculate_technical_indicators(self.daily_data)
            logger.info(f"已为{self.stock_code}计算技术指标")
            return True
        except Exception as e:
            logger.error(f"准备数据时出错: {str(e)}")
            return False

    def save_analysis_result(self, analysis_result: Dict = None) -> bool:
        """
        保存分析结果到数据库
        
        参数:
            analysis_result (Dict, 可选): 分析结果字典，如不提供则使用self.analysis_result
            
        返回:
            bool: 是否成功保存
        """
        # 如果提供了分析结果参数，则更新self.analysis_result
        if analysis_result is not None:
            self.analysis_result = analysis_result
            
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
    
    def run_analysis(self) -> Dict:
        """
        执行分析流程
        
        返回:
            Dict: 分析结果
        """
        # 在子类中实现
        raise NotImplementedError("run_analysis方法需要在子类中实现") 