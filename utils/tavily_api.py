# -*- coding: utf-8 -*-
"""Tavily API工具模块，用于获取和处理舆情数据"""

from datetime import datetime
import pandas as pd
import requests
from utils.logger import get_logger
from config import API_CONFIG

# 创建日志记录器
logger = get_logger(__name__)

class TavilyAPI:
    """Tavily搜索API工具类，用于获取舆情数据"""
    
    def __init__(self):
        """初始化Tavily API工具"""
        self.api_key = API_CONFIG.get('tavily', {}).get('api_key', '')
        self.url = "https://api.tavily.com/search"

    def search_base_news(self, payload: dict) -> requests.Response:
        """
        通过Tavily API搜索新闻
        
        参数:
            payload (dict): 搜索参数
            
        返回:
            requests.Response: API响应对象
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            response = requests.post(
                url=self.url, 
                headers=headers, 
                json=payload, 
                timeout=15
            )

            if response.status_code == 200:
                logger.info(f"成功从Tavily获取数据")
                return response
            else:
                logger.error(f"Tavily API认证失败 (HTTP {response.status_code})")
                return None
        except Exception as e:
            logger.error(f"Tavily API请求失败: {str(e)}")
            return None

    def process_tavily_response(self, response: requests.Response) -> pd.DataFrame:
        """
        处理Tavily舆论数据评分
        
        参数:
            response (requests.Response): Tavily API响应
            
        返回:
            pd.DataFrame: 处理后的舆情数据框
        """
        try:
            if response is None:
                return pd.DataFrame()
                
            results = response.json()
            # 提取新闻数据
            news_samples = []
            for result in results.get("results", []):
                title = result.get("title", "")
                sentiment = 0.5  # 默认中性

                # 简单规则：基于关键词的情感判断
                pos_words = ["利好", "增长", "提升", "突破", "上涨", "盈利", "利润", "收购", "合作", "成功"]
                neg_words = ["下跌", "亏损", "减持", "风险", "警示", "下滑", "退市", "违规", "处罚", "失败"]

                for word in pos_words:
                    if word in title:
                        sentiment += 0.1
                
                for word in neg_words:
                    if word in title:
                        sentiment -= 0.1
                
                # 将情感值限制在[-1, 1]范围内
                sentiment = max(-1, min(1, sentiment))

                publish_date = result.get("published_date", datetime.now().strftime("%Y-%m-%d"))
                
                news_samples.append({
                    "title": title,
                    "date": publish_date,
                    "sentiment": sentiment,
                    "url": result.get("url", ""),
                    "content": result.get("content", "")
                })
        
            if not news_samples:
                logger.warning("未找到相关新闻")
                return pd.DataFrame()
            
            logger.info(f"获得新闻舆情数据 ({len(news_samples)}条)")
            news_data = pd.DataFrame(news_samples)
            news_data['date'] = pd.to_datetime(news_data['date'])
            news_data.sort_values(by='date', ascending=False, inplace=True)
            news_data.reset_index(drop=True, inplace=True)
            return news_data
            
        except Exception as e:
            logger.error(f"处理Tavily API响应失败: {str(e)}")
            return pd.DataFrame()
    
    def search_stock_news(self, stock_code: str, stock_name: str = None, days: int = 7) -> pd.DataFrame:
        """
        搜索指定股票的相关新闻
        
        参数:
            stock_code (str): 股票代码
            stock_name (str): 股票名称
            days (int): 获取最近几天的新闻，默认7天
            
        返回:
            pd.DataFrame: 股票相关的舆情数据
        """
        search_term = f"{stock_code}"
        if stock_name:
            search_term += f" {stock_name}"
            
        payload = {
            "query": f"{search_term} 股票 新闻 财经",
            "search_depth": "advanced",
            "include_domains": ["eastmoney.com", "sina.com.cn", "10jqka.com.cn", "cnstock.com", "finance.qq.com"],
            "include_answer": True,
            "max_results": 15,
            "topic": "news",
            "days": days
        }
        
        response = self.search_base_news(payload)
        return self.process_tavily_response(response) 