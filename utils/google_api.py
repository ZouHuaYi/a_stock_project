#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google搜索API接口模块

该模块提供了对Google Custom Search API的访问功能，用于获取股票相关的搜索结果。
使用前需要先在Google Cloud Console创建项目并启用Custom Search API，
然后创建API密钥和自定义搜索引擎ID。

文档: https://developers.google.com/custom-search/v1/overview
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlencode
from config import API_CONFIG
import requests
from requests.exceptions import RequestException

# 配置日志
logger = logging.getLogger(__name__)

class GoogleSearchAPI:
    """Google自定义搜索API客户端"""
    
    BASE_URL = "https://customsearch.googleapis.com/customsearch/v1"
    
    def __init__(
        self, 
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        初始化Google搜索API客户端
        
        参数:
            api_key: Google API密钥，如果为None则从环境变量GOOGLE_API_KEY获取
            cx: 自定义搜索引擎ID，如果为None则从环境变量GOOGLE_SEARCH_CX获取
            max_retries: 请求失败时的最大重试次数
            retry_delay: 重试间隔时间(秒)
        """
        self.api_key = API_CONFIG.get("gemini", {}).get("api_key")
        self.cx = API_CONFIG.get("gemini", {}).get("cx_api_key")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            logger.warning("Google API密钥未设置，请设置GOOGLE_API_KEY环境变量或在初始化时提供")
        
        if not self.cx:
            logger.warning("自定义搜索引擎ID未设置，请设置GOOGLE_SEARCH_CX环境变量或在初始化时提供")
    
    def search(
        self, 
        query: str, 
        num: int = 10, 
        start: int = 1,
        safe: str = "off",
        date_restrict: Optional[str] = None,
        site_restrict: Optional[str] = None,
        file_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行Google搜索
        
        参数:
            query: 搜索查询字符串
            num: 返回结果数量，范围1-10
            start: 结果的起始索引
            safe: 安全搜索设置 ("off", "medium", "high")
            date_restrict: 日期限制，格式为"dN"表示最近N天，"wN"表示最近N周等
            site_restrict: 限制在特定网站内搜索，例如"finance.sina.com.cn"
            file_type: 文件类型限制，例如"pdf"、"doc"等
            **kwargs: 其他搜索参数
            
        返回:
            搜索结果字典
            
        异常:
            ValueError: 参数错误
            RequestException: 请求失败
        """
        if not self.api_key or not self.cx:
            raise ValueError("API密钥和搜索引擎ID必须设置才能执行搜索")
        
        if num < 1 or num > 10:
            raise ValueError("num参数必须在1-10之间")
        
        # 构建查询参数
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": num,
            "start": start,
            "safe": safe,
        }
        
        # 添加可选参数
        if date_restrict:
            params["dateRestrict"] = date_restrict
            
        if site_restrict:
            # 修改查询添加site:限制
            params["q"] = f"{query} site:{site_restrict}"
            
        if file_type:
            params["fileType"] = file_type
            
        # 添加其他自定义参数
        params.update(kwargs)
        
        # 执行请求，带重试
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.BASE_URL, params=params)
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                logger.warning(f"搜索请求失败 (尝试 {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
    
    def get_search_results(
        self, 
        query: str, 
        max_results: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        获取格式化的搜索结果列表
        
        参数:
            query: 搜索查询字符串
            max_results: 最大结果数，如果大于10将进行多次API调用
            **kwargs: 传递给search方法的其他参数
            
        返回:
            搜索结果列表，每个结果包含title, link, snippet等字段
        """
        results = []
        num_per_request = min(10, max_results)
        
        # 计算需要发起的请求数
        num_requests = (max_results + num_per_request - 1) // num_per_request
        
        for i in range(num_requests):
            start = i * num_per_request + 1
            num = min(num_per_request, max_results - len(results))
            
            if num <= 0:
                break
                
            try:
                response = self.search(query, num=num, start=start, **kwargs)
                
                # 检查是否有搜索结果
                if "items" not in response:
                    logger.info(f"没有更多搜索结果返回，已获取 {len(results)} 条结果")
                    break
                    
                # 提取结果项
                for item in response.get("items", []):
                    result = {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "display_link": item.get("displayLink"),
                        "snippet": item.get("snippet"),
                        "source": "google"
                    }
                    results.append(result)
                    
                    if len(results) >= max_results:
                        break
                        
            except Exception as e:
                logger.error(f"获取搜索结果时出错: {str(e)}")
                break
                
            # 如果结果不足num，说明已经没有更多结果了
            if len(response.get("items", [])) < num:
                break
                
        return results
    
    def search_stock_info(
        self, 
        stock_code: str, 
        stock_name: Optional[str] = None,
        max_results: int = 10,
        days: Optional[int] = 7,
        site_list: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索股票相关信息
        
        参数:
            stock_code: 股票代码
            stock_name: 股票名称，如果提供会与代码一起使用
            max_results: 最大结果数
            days: 限制为最近几天的内容，None表示不限制
            site_list: 限制搜索的网站列表，如["finance.sina.com.cn", "finance.qq.com"]
            
        返回:
            搜索结果列表
        """
        # 构建查询
        query = f"{stock_code}"
        if stock_name:
            query = f"{stock_name} {query}"
            
        # 日期限制
        date_restrict = f"d{days}" if days else None
        
        # 如果提供了网站列表，对每个网站搜索并合并结果
        results = []
        if site_list:
            results_per_site = max(1, max_results // len(site_list))
            for site in site_list:
                site_results = self.get_search_results(
                    query=query,
                    max_results=results_per_site,
                    date_restrict=date_restrict,
                    site_restrict=site
                )
                results.extend(site_results)
                
            # 如果结果不足，尝试无站点限制搜索补充
            if len(results) < max_results:
                additional_results = self.get_search_results(
                    query=query,
                    max_results=max_results - len(results),
                    date_restrict=date_restrict
                )
                results.extend(additional_results)
        else:
            # 无站点限制搜索
            results = self.get_search_results(
                query=query,
                max_results=max_results,
                date_restrict=date_restrict
            )
            
        return results[:max_results]


def get_google_search_api() -> GoogleSearchAPI:
    """
    获取Google搜索API实例（工厂函数）
    
    返回:
        GoogleSearchAPI实例
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_SEARCH_CX")
    
    return GoogleSearchAPI(api_key=api_key, cx=cx)


# 简单使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 获取API实例
    api = get_google_search_api()
    
    # 搜索示例
    try:
        results = api.search_stock_info(
            stock_code="600519", 
            stock_name="贵州茅台",
            max_results=5,
            days=7,
            site_list=["finance.sina.com.cn", "finance.eastmoney.com"]
        )
        
        print(f"找到 {len(results)} 条结果:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   链接: {result['link']}")
            print(f"   摘要: {result['snippet']}")
            print()
            
    except Exception as e:
        print(f"搜索失败: {str(e)}")
