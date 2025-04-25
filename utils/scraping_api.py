#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ScrapingBee API 爬取模块

该模块提供了对ScrapingBee API的访问功能，用于对网页进行深度爬取和内容提取。
使用前需要注册ScrapingBee并获取API密钥。
免费版每月提供1000次API调用。

文档: https://www.scrapingbee.com/documentation/
"""

import os
import json
import time
import logging
import base64
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlencode, quote

import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup

from config import API_CONFIG

# 配置日志
logger = logging.getLogger(__name__)

class ScrapingBeeAPI:
    """ScrapingBee API客户端"""
    
    BASE_URL = "https://app.scrapingbee.com/api/v1"
    
    def __init__(
        self, 
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        初始化ScrapingBee API客户端
        
        参数:
            max_retries: 请求失败时的最大重试次数
            retry_delay: 重试间隔时间(秒)
        """
        # 优先使用传入的API密钥，其次从环境变量获取，最后从配置文件获取
        self.api_key = API_CONFIG.get("scrapingbee", {}).get("api_key")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            logger.warning("ScrapingBee API密钥未设置，请在配置文件中设置或提供SCRAPINGBEE_API_KEY环境变量")
        else:
            # 确保API密钥是字符串，且移除可能的空格
            self.api_key = str(self.api_key).strip()
    
    def scrape_page(
        self, 
        url: str,
        render_js: bool = True,
        premium_proxy: bool = False,
        country_code: Optional[str] = None,
        extract_rules: Optional[Dict] = None,
        wait: Optional[int] = None,
        wait_for: Optional[str] = None,
        timeout: int = 30000
    ) -> Dict[str, Any]:
        """
        使用ScrapingBee API爬取网页内容
        
        参数:
            url: 要爬取的网页URL
            render_js: 是否渲染JavaScript
            premium_proxy: 是否使用高级代理
            country_code: 设置代理的国家/地区代码 (如: 'us', 'cn')
            extract_rules: 提取规则，用于从网页提取特定内容
            wait: 等待加载的毫秒数
            wait_for: 等待特定的CSS选择器加载完成
            timeout: 请求超时时间(毫秒)
            
        返回:
            爬取结果字典，包含状态码、内容等
            
        异常:
            ValueError: 参数错误
            RequestException: 请求失败
        """
        if not self.api_key:
            raise ValueError("API密钥必须设置才能使用ScrapingBee")
        
        if not url:
            raise ValueError("URL不能为空")
        
        # 构建查询参数
        params = {
            "api_key": self.api_key,
            "url": url,
            "render_js": "true" if render_js else "false",
            "premium_proxy": "true" if premium_proxy else "false",
        }
        
        if country_code:
            params["country_code"] = country_code
            
        if wait:
            params["wait"] = wait
            
        if wait_for:
            params["wait_for"] = wait_for
            
        if timeout:
            params["timeout"] = timeout
            
        if extract_rules:
            params["extract_rules"] = json.dumps(extract_rules)
        
        # 执行请求，带重试
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=timeout/1000 + 5)
                
                # 检查是否为JSON响应
                content_type = response.headers.get('Content-Type', '')
                
                result = {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "url": url,
                }
                
                if 'application/json' in content_type:
                    # 提取规则响应
                    result["data"] = response.json()
                else:
                    # 普通HTML响应
                    result["content"] = response.text
                    
                return result
                
            except RequestException as e:
                logger.warning(f"爬取请求失败 (尝试 {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"爬取请求最终失败: {str(e)}")
                    raise
    
    def extract_article_content(self, url: str) -> Dict[str, Any]:
        """
        提取文章正文内容
        
        参数:
            url: 文章URL
            
        返回:
            包含提取内容的字典
        """
        extract_rules = {
            "title": {
                "selector": "h1",
                "type": "text"
            },
            "content": {
                "selector": "article, .article, .post, .content, .main-content, #content, .entry-content",
                "type": "text"
            },
            "publish_date": {
                "selector": "time, .time, .date, .publish-date, .article-date, meta[property='article:published_time']",
                "type": "text",
                "output": "first"
            },
            "author": {
                "selector": ".author, .byline, .article-author, meta[name='author']",
                "type": "text",
                "output": "first"
            }
        }
        
        return self.scrape_page(
            url=url,
            render_js=True,
            extract_rules=extract_rules,
            wait=3000
        )
    
    def extract_financial_news(self, url: str) -> Dict[str, Any]:
        """
        提取财经新闻内容，针对财经网站优化
        
        参数:
            url: 财经新闻URL
            
        返回:
            包含提取内容的字典
        """
        # 根据不同的财经网站设置不同的提取规则
        extract_rules = {
            "title": {
                "selector": "h1, .article-title, .title",
                "type": "text"
            },
            "content": {
                "selector": "article, .article-content, .article, #article_content, .main-content",
                "type": "text"
            },
            "publish_date": {
                "selector": ".time, .date, .publish-time, time, .article-info span:contains('发布时间')",
                "type": "text",
                "output": "first"
            },
            "author": {
                "selector": ".source, .author, .article-source, .editor",
                "type": "text",
                "output": "first"
            },
            "keywords": {
                "selector": ".keywords, .tags, meta[name='keywords']",
                "type": "text",
                "output": "first"
            }
        }
        
        return self.scrape_page(
            url=url,
            render_js=True,
            extract_rules=extract_rules,
            wait=3000
        )
    
    def process_search_results(self, search_results: List[Dict[str, Any]], max_pages: int = 3) -> List[Dict[str, Any]]:
        """
        处理搜索结果，爬取每个结果的详细内容
        
        参数:
            search_results: 搜索结果列表
            max_pages: 最大处理页数，限制API使用
            
        返回:
            增强了内容的搜索结果列表
        """
        if not search_results:
            return []
            
        enhanced_results = []
        
        # 限制处理的数量，避免API调用过多
        for i, result in enumerate(search_results[:max_pages]):
            url = result.get('link')
            if not url:
                continue
                
            try:
                logger.info(f"爬取URL内容: {url}")
                
                # 判断是否为财经网站
                is_financial = any(domain in url for domain in [
                    'finance.sina.com.cn', 
                    'finance.eastmoney.com',
                    'finance.qq.com',
                    'business.sohu.com',
                    'money.163.com',
                    'stock.jrj.com.cn'
                ])
                
                if is_financial:
                    content_data = self.extract_financial_news(url)
                else:
                    content_data = self.extract_article_content(url)
                
                # 检查提取结果
                if 'data' in content_data:
                    # 使用提取规则的情况
                    extracted_data = content_data['data']
                    result['extracted_title'] = extracted_data.get('title', '')
                    result['extracted_content'] = extracted_data.get('content', '')
                    result['extracted_date'] = extracted_data.get('publish_date', '')
                    result['extracted_author'] = extracted_data.get('author', '')
                    if 'keywords' in extracted_data:
                        result['extracted_keywords'] = extracted_data.get('keywords', '')
                else:
                    # 使用BeautifulSoup解析HTML内容
                    html_content = content_data.get('content', '')
                    if html_content:
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # 提取标题
                        title_tag = soup.find('h1')
                        if title_tag:
                            result['extracted_title'] = title_tag.get_text(strip=True)
                            
                        # 提取正文内容
                        content_selectors = ['article', '.article', '.content', '.article-content', '#article_content', '.main-content']
                        for selector in content_selectors:
                            content_tag = soup.select_one(selector)
                            if content_tag:
                                result['extracted_content'] = content_tag.get_text(strip=True)
                                break
                                
                # 计算内容长度
                content = result.get('extracted_content', '')
                result['content_length'] = len(content) if content else 0
                
                # 添加提取状态
                result['extraction_success'] = bool(result.get('extracted_content'))
                
                enhanced_results.append(result)
                
                # 控制API调用频率
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"处理搜索结果时出错 ({url}): {str(e)}")
                # 保留原始结果
                result['extraction_success'] = False
                result['extraction_error'] = str(e)
                enhanced_results.append(result)
        
        return enhanced_results
    
    def get_usage_info(self) -> Dict[str, Any]:
        """
        获取API使用情况
        
        返回:
            API使用情况字典
        """
        if not self.api_key:
            return {"error": "API密钥未设置"}
            
        try:
            params = {"api_key": self.api_key}
            response = requests.get(f"{self.BASE_URL}/usage", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取API使用情况时出错: {str(e)}")
            return {"error": str(e)}


def get_scraping_api() -> ScrapingBeeAPI:
    """
    获取ScrapingBee API实例（工厂函数）
    
    返回:
        ScrapingBeeAPI实例
    """
    return ScrapingBeeAPI()


# 简单使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 获取API实例
    api = get_scraping_api()
    
    # 爬取示例
    try:
        url = "https://finance.sina.com.cn/stock/marketresearch/2023-08-07/doc-imfzpmir1531371.shtml"
        result = api.extract_financial_news(url)
        
        print("爬取结果:")
        if 'data' in result:
            data = result['data']
            print(f"标题: {data.get('title', '')}")
            print(f"发布日期: {data.get('publish_date', '')}")
            print(f"作者/来源: {data.get('author', '')}")
            print(f"关键词: {data.get('keywords', '')}")
            print(f"内容长度: {len(data.get('content', ''))}")
            print("\n内容预览:")
            content = data.get('content', '')
            print(content[:300] + "..." if len(content) > 300 else content)
        else:
            print("未能提取结构化数据")
            
    except Exception as e:
        print(f"示例测试失败: {str(e)}") 