#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
选股CSV文件业务分析模块

本模块用于处理选股后的CSV文件，分析股票的行业划分和业务交叉关系
1. 获取output/selector/YMD 当天的目录下所有的 csv 文件
2. 解析获取文件内容后，这些股票丢给llm_api 的 AI 模型处理
3. AI 模型整理这些股票的行业划分，以及业务交叉关系
4. 整理好的内容输出到到 PATH_CONFIG 中的配置 collect_path，txt 格式的文件

作者: AI助手
创建日期: 2025-04-19
"""

import os
import sys
import glob
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATH_CONFIG
from utils.llm_api import LLMAPI
from utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)

class CSVBusinessAnalyzer:
    """选股CSV文件业务分析器"""
    
    def __init__(self):
        """初始化CSV业务分析器"""
        self.llm_api = LLMAPI()
        self.selector_path = PATH_CONFIG['selector_path']
        self.collect_path = PATH_CONFIG['collect_path']
        self.duplicate_stocks = []
        
        # 确保输出目录存在
        os.makedirs(self.collect_path, exist_ok=True)
        
        logger.info(f"CSV业务分析器初始化完成，选股目录: {self.selector_path}, 输出目录: {self.collect_path}")
    
    def get_today_selector_path(self) -> str:
        """
        获取今天的选股器目录路径
        
        返回:
            str: 今天的选股器目录路径
        """
        today = datetime.now().strftime('%Y%m%d')
        path = os.path.join(self.selector_path, today)
        logger.debug(f"今日选股器目录路径: {path}")
        return path
    
    def get_csv_files(self, directory: Optional[str] = None) -> List[str]:
        """
        获取指定目录下的所有CSV文件
        
        参数:
            directory (str, optional): 要扫描的目录，默认为今天的选股器目录
            
        返回:
            List[str]: CSV文件路径列表
        """
        if directory is None:
            directory = self.get_today_selector_path()
            
        if not os.path.exists(directory):
            logger.warning(f"目录不存在: {directory}")
            return []
            
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        logger.info(f"找到 {len(csv_files)} 个CSV文件")
        return csv_files
    
    def parse_csv_files(self, csv_files: List[str]) -> Dict[str, Any]:
        """
        解析CSV文件，提取股票信息
        
        参数:
            csv_files (List[str]): CSV文件路径列表
            
        返回:
            Dict[str, Any]: 解析后的股票信息
        """
        all_stocks = {}
        total_stocks = 0
        
        for csv_file in csv_files:
            try:
                # 从文件名中提取选股器类型
                selector_type = os.path.basename(csv_file).replace('.csv', '')
                
                # 读取CSV文件
                df = pd.read_csv(csv_file)
                
                # 提取股票代码、名称、行业等信息
                stocks_data = []
                for _, row in df.iterrows():
                    stock_info = {
                        'stock_code': row.get('stock_code', ''),
                        'stock_name': row.get('stock_name', ''),
                        'industry': row.get('industry', ''),
                        'score': row.get('score', 0)
                    }
                    stocks_data.append(stock_info)
                
                all_stocks[selector_type] = stocks_data
                total_stocks += len(stocks_data)
                logger.info(f"已解析 {selector_type} 选股器的 {len(stocks_data)} 只股票")
                
            except Exception as e:
                logger.error(f"解析CSV文件 {csv_file} 失败: {str(e)}")
        
        logger.info(f"共解析 {len(all_stocks)} 个选股器文件，包含 {total_stocks} 只股票")
        return all_stocks
    
    def analyze_business_relationships(self, stocks_data: Dict[str, Any]) -> str:
        """
        使用LLM分析股票的行业划分和业务交叉关系
        
        参数:
            stocks_data (Dict[str, Any]): 解析后的股票信息
            
        返回:
            str: 分析结果
        """
        # 扁平化所有股票数据
        all_stocks = []
        for selector_type, stocks in stocks_data.items():
            for stock in stocks:
                stock['selector_type'] = selector_type
                all_stocks.append(stock)
        
        # 对股票数据进行去重
        unique_stocks = {}
        # 重复股票
        self.duplicate_stocks = []
        for stock in all_stocks:
            stock_code = stock['stock_code']
            if stock_code not in unique_stocks:
                unique_stocks[stock_code] = stock
            else:
                self.duplicate_stocks.append(stock)
        
        logger.info(f"开始分析 {len(unique_stocks)} 只去重后的股票")
        
        # 构建提示词
        prompt = """
        你是一位资深的股票分析师和行业专家，请分析以下股票的行业划分和业务交叉关系：
        
        股票列表：
        """

        # 重复股票
        if self.duplicate_stocks:
            prompt += "\n重复股票："
            for stock in self.duplicate_stocks:
                prompt += f"\n{stock['stock_code']} | {stock['stock_name']} | {stock['industry']} | 来自选股器: {stock.get('selector_type', '')}"
        
        # 去重股票
        prompt += "\n去重股票："
        for stock_code, stock in unique_stocks.items():
            prompt += f"\n{stock['stock_code']} | {stock['stock_name']} | {stock['industry']} | 来自选股器: {stock.get('selector_type', '')}"
        
        prompt += """
        
        请完成以下分析任务：
        1. 将这些股票按行业进行归类和统计
        2. 分析不同行业之间可能存在的业务协同或上下游关系
        3. 找出这些股票中业务模式相似或有竞争关系的公司
        4. 分析这些股票所代表的行业未来发展趋势
        5. 重点分析重复股票，重复选中股票比较重要，需要分析原因
        6. 总结这些被选中股票可能反映的市场热点和投资机会
        
        
        请以清晰的结构化格式输出分析结果，重点突出行业关系和业务交叉。
        """
        
        logger.info("开始使用LLM分析业务关系")
        try:
            # 优先使用OpenAI模型
            logger.debug("尝试使用OpenAI模型进行分析")
            analysis_result = self.llm_api.generate_openai_response(prompt)
            
            if not analysis_result:
                # 如果OpenAI调用失败，尝试使用Gemini
                logger.debug("OpenAI模型分析失败，尝试使用Gemini模型")
                analysis_result = self.llm_api.generate_gemini_response(prompt)
            
            logger.info("业务关系分析完成")
            return analysis_result
            
        except Exception as e:
            logger.error(f"业务关系分析失败: {str(e)}")
            return "分析失败：" + str(e)
    
    def save_analysis_result(self, analysis_result: str) -> str:
        """
        保存分析结果到文件
        
        参数:
            analysis_result (str): 分析结果
            
        返回:
            str: 保存的文件路径
        """
        today = datetime.now().strftime('%Y%m%d')
        result_file = os.path.join(self.collect_path, f"{self.__class__.__name__}_{today}.txt")
        
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(analysis_result)
            
            logger.info(f"分析结果已保存至: {result_file}")
            return result_file
            
        except Exception as e:
            logger.error(f"保存分析结果失败: {str(e)}")
            return ""
    
    def run(self) -> None:
        """运行CSV业务分析器的完整流程"""
        try:
            # 1. 获取CSV文件
            csv_files = self.get_csv_files()
            if not csv_files:
                logger.warning("没有找到CSV文件，分析终止")
                return
            
            # 2. 解析CSV文件
            stocks_data = self.parse_csv_files(csv_files)
            if not stocks_data:
                logger.warning("没有解析到有效的股票数据，分析终止")
                return
            
            # 3. 分析业务关系
            analysis_result = self.analyze_business_relationships(stocks_data)
            if not analysis_result:
                logger.warning("生成业务分析结果失败，分析终止")
                return
            
            # 4. 保存分析结果
            result_file = self.save_analysis_result(analysis_result)
            if result_file:
                logger.info(f"CSV业务分析完成，结果保存在: {result_file}")
            else:
                logger.warning("保存分析结果失败")
                
        except Exception as e:
            logger.error(f"CSV业务分析过程中发生错误: {str(e)}")


if __name__ == "__main__":
    analyzer = CSVBusinessAnalyzer()
    analyzer.run() 