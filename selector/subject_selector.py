# -*- coding: utf-8 -*-
"""题材选股模块"""

import os
import pandas as pd
import akshare as ak
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import time

# 导入基类和工具
from selector.base_selector import BaseSelector
from utils.llm_api import LLMAPI
from utils.akshare_api import AkshareAPI
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger(__name__)

class SubjectSelector(BaseSelector):
    """题材选股器，基于akshare和LLM实现题材选股功能"""
    
    def __init__(self, days=None, threshold=None, limit=None, text=None, ai_type="openai"):
        """
        初始化题材选股器
        
        参数:
            days (int, 可选): 回溯数据天数
            threshold (float, 可选): 选股分数阈值
            limit (int, 可选): 限制结果数量
            text (str, 可选): 用于分析的文本内容
            ai_type (str, 可选): 使用的AI类型，默认为openai
        """
        super().__init__(days, threshold, limit)
        self.text = text
        self.ai_type = ai_type
        self.akshare_api = AkshareAPI()
        self.llm_api = LLMAPI()
        self.concept_stocks = {}
        
    def get_concept_stocks(self) -> Dict[str, List[str]]:
        """
        获取所有概念板块及对应股票
        
        返回:
            Dict[str, List[str]]: 概念板块及其成份股代码字典
        """
        logger.info("正在获取概念板块数据...")
        concept_stocks = {}
        
        try:
            # 获取概念板块列表
            concept_board = ak.stock_board_concept_name_em()
            
            # 获取每个概念板块的成份股
            total_concepts = len(concept_board)
            for idx, concept in enumerate(concept_board['板块名称']):
                try:
                    logger.info(f"获取概念板块 [{concept}] 成份股 ({idx+1}/{total_concepts})...")
                    df = ak.stock_board_concept_cons_em(symbol=concept)
                    # 提取股票代码
                    concept_stocks[concept] = df['代码'].tolist()
                    # 避免请求过快
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"获取概念 {concept} 成份股失败: {str(e)}")
            
            logger.info(f"成功获取 {len(concept_stocks)} 个概念板块数据")
            # 缓存概念股数据
            self.concept_stocks = concept_stocks
            return concept_stocks
        
        except Exception as e:
            logger.error(f"获取概念板块数据失败: {str(e)}")
            return {}
    
    def analyze_text_with_llm(self, text: str) -> List[str]:
        """
        使用LLM提取文本中的题材关键词
        
        参数:
            text (str): 需要分析的文本内容
            
        返回:
            List[str]: 识别出的题材关键词列表
        """
        logger.info("使用LLM分析文本识别题材关键词...")
        
        # 确保概念股数据已加载
        if not self.concept_stocks:
            self.get_concept_stocks()
            
        # 构建概念列表用于提示
        concept_names = list(self.concept_stocks.keys())
        concepts_text = "、".join(concept_names[:30])  # 取前30个作为参考示例
        
        # 构建提示词
        prompt = f"""
作为股票市场分析专家，请从以下文本中识别出与A股市场相关的投资题材或概念。
请参考以下概念板块名称：{concepts_text}等。

分析文本: {text}

请分析出最相关的3-5个投资题材概念，直接以逗号分隔的方式输出概念名称，无需其他解释。如果提取不到相关题材，请返回'无相关题材'。
"""
        
        # 调用不同的AI接口
        try:
            if self.ai_type.lower() == "gemini":
                response = self.llm_api.generate_gemini_response(prompt)
            else:
                response = self.llm_api.generate_openai_response(prompt)
                
            # 处理响应结果
            if response:
                concepts = [concept.strip() for concept in response.split(',')]
                logger.info(f"成功识别题材关键词: {concepts}")
                return concepts
            else:
                logger.warning("LLM分析未返回有效结果")
                return []
                
        except Exception as e:
            logger.error(f"LLM分析文本失败: {str(e)}")
            return []
    
    def evaluate_stock(self, stock_code: str) -> Optional[Dict]:
        """
        评估单只股票
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            Optional[Dict]: 评估结果字典，如果无法评估则返回None
        """
        # 题材选股器不使用单只股票评估方法
        # 而是在run_screening中直接根据题材关键词筛选
        return None
    
    def run_screening(self, output_file=None) -> str:
        """
        执行选股流程
        
        参数:
            output_file (str, 可选): 输出文件名
            
        返回:
            str: 保存的文件路径
        """
        # 如果没有提供文本，则无法进行选股
        if not self.text:
            logger.error("未提供用于分析的文本内容，无法进行题材选股")
            return ""
        
        logger.info("开始执行题材选股...")
        
        # 1. 加载概念股数据
        if not self.concept_stocks:
            self.get_concept_stocks()
        
        # 如果无法获取概念股数据，则无法进行选股
        if not self.concept_stocks:
            logger.error("无法获取概念股数据，选股失败")
            return ""
        
        # 2. 使用LLM分析文本，提取题材关键词
        target_concepts = self.analyze_text_with_llm(self.text)
        
        if not target_concepts:
            logger.warning("未能从文本中识别出相关题材，选股失败")
            return ""
        
        # 3. 根据题材关键词筛选股票
        logger.info(f"基于题材关键词 {target_concepts} 筛选股票...")
        selected_stocks = []
        
        for concept in target_concepts:
            # 可能存在部分匹配的情况，如输入"新能源"可能匹配"新能源汽车"
            matched_concepts = [k for k in self.concept_stocks.keys() if concept in k]
            
            for matched_concept in matched_concepts:
                if matched_concept in self.concept_stocks:
                    stocks = self.concept_stocks[matched_concept]
                    logger.info(f"概念 [{matched_concept}] 包含 {len(stocks)} 只股票")
                    for stock in stocks:
                        selected_stocks.append({
                            'stock_code': stock,
                            'concept': matched_concept,
                            'matched_keyword': concept
                        })
        
        # 统计每只股票出现在几个题材中，并作为权重
        stock_weights = {}
        for item in selected_stocks:
            code = item['stock_code']
            if code in stock_weights:
                stock_weights[code]['count'] += 1
                if item['concept'] not in stock_weights[code]['concepts']:
                    stock_weights[code]['concepts'].append(item['concept'])
            else:
                stock_weights[code] = {
                    'count': 1,
                    'concepts': [item['concept']]
                }
        
        # 按权重排序并转换为DataFrame
        sorted_stocks = sorted(stock_weights.items(), key=lambda x: x[1]['count'], reverse=True)
        
        # 限制结果数量
        if self.limit and len(sorted_stocks) > self.limit:
            sorted_stocks = sorted_stocks[:self.limit]
        
        # 准备结果数据
        results_data = []
        for code, info in sorted_stocks:
            # 获取股票名称和当前价格
            stock_name = self.akshare_api.get_stock_name(code)
            
            try:
                # 获取实时行情数据
                stock_info = ak.stock_individual_info_em(symbol=code)
                current_price = stock_info.iloc[3, 1]  # 当前价格位置
                
                # 计算得分(简单使用题材匹配数作为得分)
                score = info['count']
                
                results_data.append({
                    'stock_code': code,
                    'stock_name': stock_name,
                    'current_price': current_price,
                    'score': score,
                    'concepts': ','.join(info['concepts']),
                    'matched_keywords': ','.join(set([item['matched_keyword'] for item in selected_stocks if item['stock_code'] == code]))
                })
                
                # 避免请求过快
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"获取股票 {code} 信息失败: {str(e)}")
        
        # 转换为DataFrame
        self.results = pd.DataFrame(results_data)
        
        # 确保结果不为空
        if self.results.empty:
            logger.warning("题材选股未找到符合条件的股票")
            return ""
        
        # 排序
        self.results.sort_values(by='score', ascending=False, inplace=True)
        
        # 打印结果
        self.print_results(self.results)
        
        # 保存结果
        if output_file:
            return self.save_results(self.results, output_file)
        else:
            return self.save_results(self.results, f"subject_selection_{datetime.now().strftime('%Y%m%d')}.csv")
