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
    
    BASE_URL = "http://api.scrape.do"
    
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
            logger.warning("API密钥未设置，请在配置文件中设置或提供API_KEY环境变量")
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
        使用API爬取网页内容
        
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
            raise ValueError("API密钥必须设置才能使用API")
        
        if not url:
            raise ValueError("URL不能为空")
        
        # 构建查询参数 - 仅使用api.scrape.do支持的参数
        params = {
            "token": self.api_key,
            "url": url
        }
        
        # 针对中文网站的特殊处理
        is_chinese_site = any(domain in url.lower() for domain in [
            '.cn', 
            'sina.com', 
            'sohu.com',
            '163.com',
            'eastmoney.com',
            'qq.com',
            'ifeng.com',
            'jrj.com',
            'xueqiu.com',
            '10jqka.com'
        ])
        
        # 添加常见的请求头
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive"
        }
            
        # 执行请求，带重试
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"请求URL: {self.BASE_URL} 参数: {params}")
                response = requests.get(
                    self.BASE_URL, 
                    params=params,
                    headers=headers,
                    timeout=(timeout/1000 + 5)
                )
                
                result = {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "url": url,
                }
                
                # 检查响应状态码
                if response.status_code == 200:
                    # 检查内容类型
                    content_type = response.headers.get('Content-Type', '')
                    
                    if 'application/json' in content_type:
                        # JSON响应
                        try:
                            result["data"] = response.json()
                        except:
                            # 可能是JSON格式的纯文本
                            result["content"] = response.text
                    else:
                        # HTML响应
                        content = response.text
                        
                        # 可能需要特别处理中文编码
                        if is_chinese_site and ('gb' in content_type.lower() or 'gbk' in content_type.lower()):
                            try:
                                # 尝试检测和修复中文编码问题
                                content = self._fix_chinese_encoding(content)
                            except:
                                pass
                        
                        result["content"] = content
                        
                        # 立即尝试提取内容，如果提供了提取规则
                        if extract_rules and result["content"]:
                            result["data"] = self._extract_content_with_rules(
                                result["content"], 
                                extract_rules
                            )
                            
                    return result
                else:
                    # 请求失败但有响应，重试前记录错误
                    logger.warning(f"爬取失败，状态码: {response.status_code}, 响应: {response.text[:200]}")
                    
                    # 如果已经尝试最大次数，返回失败结果
                    if attempt == self.max_retries - 1:
                        result["error"] = response.text
                        return result
                    
                    # 否则重试
                    time.sleep(self.retry_delay)
                    
            except RequestException as e:
                logger.warning(f"爬取请求失败 (尝试 {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"爬取请求最终失败: {str(e)}")
                    # 返回错误结果而不是引发异常
                    return {
                        "status_code": 0,
                        "error": str(e),
                        "url": url
                    }
    
    def _fix_chinese_encoding(self, content: str) -> str:
        """
        尝试修复中文编码问题
        
        参数:
            content: 原始内容
            
        返回:
            修复后的内容
        """
        # 如果内容为空，直接返回
        if not content:
            return content
            
        # 查找编码声明
        import re
        
        # 从HTML meta标签中查找编码声明
        charset_match = re.search(r'<meta[^>]*charset=["\']*([^"\'>]+)', content, re.IGNORECASE)
        charset = None
        
        if charset_match:
            charset = charset_match.group(1).lower()
            logger.debug(f"从meta标签检测到网页编码: {charset}")
        
        # 针对特定的中文网站进行硬编码处理
        if "sina.com" in content or "sina.com.cn" in content:
            # 新浪网站通常使用utf-8编码
            charset = "utf-8"
            logger.debug("检测到新浪网站，使用utf-8编码")
        elif "163.com" in content:
            # 网易通常使用utf-8编码
            charset = "utf-8"
            logger.debug("检测到网易网站，使用utf-8编码")
        elif "sohu.com" in content:
            # 搜狐通常使用utf-8编码
            charset = "utf-8"
            logger.debug("检测到搜狐网站，使用utf-8编码")
        elif "eastmoney.com" in content:
            # 东方财富通常使用utf-8编码
            charset = "utf-8"
            logger.debug("检测到东方财富网站，使用utf-8编码")
        elif "baidu.com" in content:
            # 百度通常使用utf-8编码
            charset = "utf-8"
            logger.debug("检测到百度网站，使用utf-8编码")
             
        # 直接尝试修复常见的中文乱码问题
        try:
            # 检查内容是否包含常见的中文乱码特征
            if 'æ' in content or 'é' in content or 'è' in content or 'â' in content:
                # 这是最常见的中文乱码修复方法 - latin1误解码的utf-8
                content = content.encode('latin1').decode('utf-8', errors='replace')
                logger.debug("检测到疑似中文乱码，尝试用latin1->utf8修复")
        except Exception as e:
            logger.debug(f"尝试修复中文乱码失败: {str(e)}")
             
        # 如果确定了编码，尝试重新解码内容
        if charset:
            try:
                # 先用latin1编码回字节序列，然后使用指定编码解码
                if isinstance(content, str):
                    # 简单直接的方法，假定内容为utf-8编码
                    if charset.lower() == 'utf-8':
                        # 先检查内容是否已经是有效的utf-8
                        if not all(ord(c) < 128 for c in content[:200]):
                            try:
                                # 可能是latin1误解码的utf-8内容
                                content = content.encode('latin1').decode('utf-8', errors='replace')
                            except:
                                pass
                    # 对于GB系列编码
                    elif 'gb' in charset or 'gbk' in charset or 'gb2312' in charset:
                        try:
                            # 尝试使用对应编码解码
                            content = content.encode('latin1').decode(charset, errors='replace')
                        except:
                            pass
            except Exception as e:
                logger.debug(f"编码转换失败 ({charset}): {str(e)}")
                
        # 处理Unicode转义序列
        if '\\u' in content or '\\x' in content:
            try:
                # 尝试解码Unicode转义序列
                unescaped = content.encode().decode('unicode_escape', errors='replace')
                # 只有当转义后结果看起来合理时才使用
                if len(unescaped) > 0 and not unescaped.isspace():
                    content = unescaped
            except:
                pass
                
        # 移除无意义字符和格式化
        content = content.replace('\ufffd', '') # 替换Unicode替换字符
        
        return content
    
    def _extract_content_with_rules(self, html_content: str, extract_rules: Dict) -> Dict[str, Any]:
        """
        使用BeautifulSoup根据提取规则解析HTML内容
        
        参数:
            html_content: HTML内容
            extract_rules: 提取规则
            
        返回:
            提取的内容字典
        """
        # 处理可能的中文编码问题
        try:
            import re
            # 首先尝试检测页面编码
            charset_match = re.search(r'<meta[^>]*charset=["\']*([^"\'>]+)', html_content)
            if charset_match:
                charset = charset_match.group(1).lower()
                logger.debug(f"检测到网页使用编码: {charset}")
                
                # 针对GB系列编码(常见于中文网站)
                if ('gb' in charset or 'gbk' in charset or 'gb2312' in charset) and isinstance(html_content, str):
                    try:
                        # 尝试处理GB系列编码，采用更安全的方式
                        encoded = html_content.encode('latin1', errors='ignore')
                        html_content = encoded.decode(charset, errors='replace')
                    except Exception as e:
                        logger.debug(f"使用{charset}重新解码失败: {str(e)}")
        except Exception as e:
            # 即使编码处理失败也继续执行
            logger.debug(f"编码预处理失败: {str(e)}")
            
        # 对于已经是Unicode但包含转义编码的情况（常见于爬虫返回结果）
        if isinstance(html_content, str) and ('\\u' in html_content or '\\x' in html_content):
            try:
                # 处理Unicode转义序列
                html_content = html_content.encode().decode('unicode_escape')
            except:
                pass
            
        # 使用BeautifulSoup解析内容
        soup = BeautifulSoup(html_content, 'html.parser')
        result = {}
        
        # 应用提取规则
        for key, rule in extract_rules.items():
            selector = rule.get('selector', '')
            output_type = rule.get('type', 'text')
            output_mode = rule.get('output', 'all')
            
            if not selector:
                continue
                
            try:
                elements = soup.select(selector)
                
                if not elements:
                    continue
                    
                if output_mode == 'first':
                    element = elements[0]
                    if output_type == 'text':
                        # 去除多余空白
                        result[key] = self._normalize_text(element.get_text(strip=True))
                    elif output_type == 'html':
                        result[key] = str(element)
                    elif output_type == 'attr' and 'attr' in rule:
                        result[key] = element.get(rule['attr'], '')
                else:
                    if output_type == 'text':
                        result[key] = [self._normalize_text(el.get_text(strip=True)) for el in elements]
                    elif output_type == 'html':
                        result[key] = [str(el) for el in elements]
                    elif output_type == 'attr' and 'attr' in rule:
                        result[key] = [el.get(rule['attr'], '') for el in elements]
            
            except Exception as e:
                logger.error(f"提取规则'{key}'应用失败: {str(e)}")
        
        # 如果没有提取到内容，尝试使用基本提取方法
        if not result.get('content') and not result.get('title'):
            self._extract_basic_content(soup, result)
                
        return result
        
    def _normalize_text(self, text: str) -> str:
        """
        规范化文本内容，处理常见的编码和格式问题
        
        参数:
            text: 原始文本
            
        返回:
            处理后的文本
        """
        if not text:
            return ""
            
        # 先尝试修复中文编码问题
        text = self._fix_chinese_text(text)
            
        # 去除多余空白
        text = ' '.join(text.split())
        
        # 处理常见的编码问题
        if '\\u' in repr(text) or '\\x' in repr(text):
            try:
                # 尝试修复Unicode和十六进制转义序列
                text = text.encode('latin1').decode('utf-8', errors='replace')
            except:
                pass
                
        return text
        
    def _extract_basic_content(self, soup: BeautifulSoup, result: Dict[str, Any]) -> None:
        """
        基本内容提取，当规则提取失败时使用
        
        参数:
            soup: BeautifulSoup对象
            result: 结果字典，将被修改
        """
        # 提取标题
        if not result.get('title'):
            title_tag = soup.find('title')
            if title_tag:
                result['title'] = self._normalize_text(title_tag.get_text(strip=True))
                
            # 如果没有title标签，尝试h1
            if not result.get('title'):
                h1_tag = soup.find('h1')
                if h1_tag:
                    result['title'] = self._normalize_text(h1_tag.get_text(strip=True))
        
        # 提取正文内容
        if not result.get('content'):
            # 常见的内容容器选择器
            content_selectors = [
                'article', '.article', '.content', '.article-content', 
                '#article_content', '.main-content', '.post-content',
                '.news-content', '.entry-content', '#content'
            ]
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    # 使用最长的内容块
                    contents = [self._normalize_text(el.get_text(strip=True)) for el in elements]
                    if contents:
                        result['content'] = max(contents, key=len)
                        break
            
            # 如果仍未提取到内容，尝试提取所有p标签
            if not result.get('content'):
                p_tags = soup.find_all('p')
                if p_tags:
                    paragraphs = [self._normalize_text(p.get_text(strip=True)) for p in p_tags 
                                  if len(p.get_text(strip=True)) > 20]  # 只保留长度合理的段落
                    if paragraphs:
                        result['content'] = '\n'.join(paragraphs)

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
                "selector": "h1, .article-title, .title, .post-title, .headline, title",
                "type": "text",
                "output": "first"
            },
            "content": {
                "selector": "article, .article, .post, .content, .main-content, #content, .entry-content, .article-content, .news-content",
                "type": "text"
            },
            "publish_date": {
                "selector": "time, .time, .date, .publish-date, .article-date, meta[property='article:published_time'], .article-info span:contains('时间'), .article-info span:contains('日期')",
                "type": "text",
                "output": "first"
            },
            "author": {
                "selector": ".author, .byline, .article-author, meta[name='author'], .source, .article-source, .article-info span:contains('作者'), .article-info span:contains('来源')",
                "type": "text",
                "output": "first"
            }
        }
        
        result = self.scrape_page(
            url=url,
            render_js=True,
            wait=3000
        )
        
        # 检查是否有内容
        if result.get("status_code") == 200 and "content" in result:
            # 使用提取规则解析内容
            result["data"] = self._extract_content_with_rules(result["content"], extract_rules)
            
        return result
    
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
                "selector": "h1, .article-title, .title, .main-title, .content-title, title",
                "type": "text",
                "output": "first"
            },
            "content": {
                "selector": "article, .article-content, .article, #article_content, .main-content, .content, #content, .news-txt, .news-text",
                "type": "text"
            },
            "publish_date": {
                "selector": ".time, .date, .publish-time, time, .article-info span:contains('发布时间'), .article-info span:contains('时间'), .article-info span:contains('日期'), .info span:contains('日期')",
                "type": "text",
                "output": "first"
            },
            "author": {
                "selector": ".source, .author, .article-source, .editor, .article-info span:contains('来源'), .article-info span:contains('编辑'), .article-info span:contains('作者'), .info span:contains('来源')",
                "type": "text",
                "output": "first"
            },
            "keywords": {
                "selector": ".keywords, .tags, meta[name='keywords'], .article-tags",
                "type": "text",
                "output": "first"
            }
        }
        
        result = self.scrape_page(
            url=url,
            render_js=True,
            wait=3000
        )
        
        # 检查是否有内容
        if result.get("status_code") == 200 and "content" in result:
            # 使用提取规则解析内容
            result["data"] = self._extract_content_with_rules(result["content"], extract_rules)
            
        return result
    
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
        successful_crawls = 0
        
        # 限制处理的数量，避免API调用过多
        for i, result in enumerate(search_results[:max_pages]):
            url = result.get('link')
            if not url:
                continue
                
            try:
                logger.info(f"爬取URL内容: {url}")
                
                # 判断网站类型并使用对应的提取方法
                content_data = None
                
                # 判断是否为财经网站
                is_financial = any(domain in url.lower() for domain in [
                    'finance.sina.com.cn', 
                    'finance.eastmoney.com',
                    'eastmoney.com',
                    'finance.qq.com',
                    'business.sohu.com',
                    'money.163.com',
                    'stock.jrj.com.cn',
                    'xueqiu.com',
                    '10jqka.com.cn',
                    'cnstock.com'
                ])
                
                # 尝试爬取，如果失败最多重试2次
                retry_count = 0
                max_retries = 2
                while retry_count <= max_retries:
                    try:
                        if is_financial:
                            content_data = self.extract_financial_news(url)
                        else:
                            content_data = self.extract_article_content(url)
                            
                        # 爬取成功就跳出循环
                        if content_data and content_data.get('status_code') == 200:
                            break
                    except Exception as e:
                        logger.warning(f"爬取尝试 {retry_count+1} 失败: {str(e)}")
                    
                    retry_count += 1
                    if retry_count <= max_retries:
                        time.sleep(2)  # 等待一段时间后重试
                
                # 检查爬取是否成功
                if not content_data or content_data.get('status_code') != 200:
                    # 处理请求失败的情况
                    result['extraction_success'] = False
                    if content_data:
                        error_msg = content_data.get('error', f"状态码: {content_data.get('status_code')}")
                        result['extraction_error'] = error_msg
                    else:
                        result['extraction_error'] = "爬取失败，未返回数据"
                    enhanced_results.append(result)
                    continue
                
                # 检查提取结果
                if 'data' in content_data and content_data['data']:
                    # 使用提取规则的情况
                    extracted_data = content_data['data']
                    
                    # 提取标题并处理可能的编码问题
                    if 'title' in extracted_data:
                        title = extracted_data.get('title', '')
                        if isinstance(title, list) and title:
                            # 如果标题是列表，选择最可能的标题（通常是第一个）
                            clean_title = self._clean_text(title[0]) if title else ''
                        else:
                            clean_title = self._clean_text(title)
                        result['extracted_title'] = clean_title
                    
                    # 提取内容并处理编码问题
                    if 'content' in extracted_data:
                        content = extracted_data.get('content', '')
                        if isinstance(content, list) and content:
                            # 如果内容是列表，选择最长的内容
                            clean_content = self._clean_text(max(content, key=lambda x: len(str(x)) if x else 0))
                        else:
                            clean_content = self._clean_text(content)
                        result['extracted_content'] = clean_content
                    
                    # 提取发布日期
                    if 'publish_date' in extracted_data:
                        date = extracted_data.get('publish_date', '')
                        result['extracted_date'] = self._clean_text(date)
                    
                    # 提取作者/来源
                    if 'author' in extracted_data:
                        author = extracted_data.get('author', '')
                        result['extracted_author'] = self._clean_text(author)
                    
                    # 提取关键词
                    if 'keywords' in extracted_data:
                        keywords = extracted_data.get('keywords', '')
                        result['extracted_keywords'] = self._clean_text(keywords)
                    
                elif 'content' in content_data:
                    # 手动解析HTML内容
                    html_content = content_data.get('content', '')
                    if html_content:
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # 提取标题
                        title_selectors = ['h1', '.article-title', '.title', '.post-title', '.headline', 'title']
                        for selector in title_selectors:
                            title_elements = soup.select(selector)
                            if title_elements:
                                result['extracted_title'] = self._clean_text(title_elements[0].get_text(strip=True))
                                break
                        
                        # 如果没有提取到标题，使用网页标题
                        if not result.get('extracted_title'):
                            title_tag = soup.find('title')
                            if title_tag:
                                result['extracted_title'] = self._clean_text(title_tag.get_text(strip=True))
                            
                        # 提取正文内容
                        content_selectors = [
                            'article', '.article', '.post', '.content', '.main-content', 
                            '#content', '.entry-content', '.article-content', '.news-content',
                            '.news-txt', '.news-text'
                        ]
                        
                        for selector in content_selectors:
                            content_elements = soup.select(selector)
                            if content_elements:
                                # 选择最长的内容块
                                contents = [el.get_text(strip=True) for el in content_elements]
                                if contents:
                                    result['extracted_content'] = self._clean_text(max(contents, key=len))
                                    break
                        
                        # 如果仍未提取到内容，尝试提取所有p标签
                        if not result.get('extracted_content'):
                            p_tags = soup.find_all('p')
                            if p_tags:
                                paragraphs = [p.get_text(strip=True) for p in p_tags 
                                              if len(p.get_text(strip=True)) > 20]  # 只保留长度合理的段落
                                if paragraphs:
                                    result['extracted_content'] = self._clean_text('\n'.join(paragraphs))
                                
                # 计算内容长度
                content = result.get('extracted_content', '')
                result['content_length'] = len(content) if content else 0
                
                # 添加提取状态
                result['extraction_success'] = bool(result.get('extracted_content'))
                
                if result['extraction_success']:
                    successful_crawls += 1
                
                enhanced_results.append(result)
                
                # 控制API调用频率
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"处理搜索结果时出错 ({url}): {str(e)}")
                # 保留原始结果
                result['extraction_success'] = False
                result['extraction_error'] = str(e)
                enhanced_results.append(result)
        
        logger.info(f"深度爬取完成，成功处理 {successful_crawls}/{min(len(search_results), max_pages)} 个URL")
        return enhanced_results
        
    def _clean_text(self, text) -> str:
        """
        清理文本内容，处理编码问题并移除多余空白
        
        参数:
            text: 需要清理的文本或文本列表
            
        返回:
            清理后的文本
        """
        # 检查输入类型
        if not text:
            return ""
            
        # 如果是列表，处理列表内容
        if isinstance(text, list):
            if not text:
                return ""
                
            # 如果只有一个元素，直接处理它
            if len(text) == 1:
                return self._clean_text(str(text[0]))
                
            # 否则连接所有元素
            return "\n".join(self._clean_text(str(item)) for item in text)
        
        # 确保是字符串
        text = str(text)
        
        # 去除多余空白，如多个连续空格、换行等
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        
        # 处理编码问题
        if '\\u' in text or '\\x' in text or re.search(r'[\xc0-\xff][\x80-\xbf]', text):
            # 尝试解决Unicode转义序列
            try:
                fixed = text.encode().decode('unicode_escape', errors='replace')
                if len(fixed) > len(text)/2:  # 只有当结果合理时才使用
                    text = fixed
            except:
                pass
                
        # 针对中文特别处理
        if any(ord(c) > 127 for c in text):
            # 使用直接编码替换处理，优先考虑utf-8
            text = self._fix_chinese_text(text)
                
        return text
        
    def _fix_chinese_text(self, text: str) -> str:
        """
        专门用于修复中文文本编码问题
        
        参数:
            text: 原始文本
            
        返回:
            修复后的文本
        """
        # 如果内容为空，直接返回
        if not text:
            return text
            
        # 对于明显包含中文编码问题的文本
        if 'æ' in text or 'â' in text or 'é' in text or 'è' in text or 'ç' in text:
            try:
                # 这是最常见的中文乱码修复方法
                text = text.encode('latin1').decode('utf-8', errors='replace')
                logger.debug("使用latin1->utf8修复中文乱码")
            except Exception as e:
                logger.debug(f"修复中文乱码失败: {str(e)}")
                
        # 尝试UTF-8字节序列重新解码
        if '\\u00' in text or '\\x' in text:
            try:
                # 常见于将utf-8字节序列误解码为latin1的情况
                text = text.encode('latin1').decode('utf-8', errors='replace')
            except:
                pass
                
        # 处理常见的乱码字符
        replacements = {
            'â€™': "'",  # 撇号
            'â€œ': '"',  # 左双引号
            'â€': '"',   # 右双引号
            'â€¦': '…',  # 省略号
            'â€"': '—',  # 破折号
            '\xa0': ' ',  # 不间断空格
            '\u3000': ' ',  # 全角空格
            # 新增处理常见的中文乱码替换
            'ç»': '经',
            'è´¢': '财',
            'æ°': '新',
            'æµª': '浪',
            'ä¸': '中',
            'å½': '国',
            'é¦': '首',
            'é¡µ': '页',
            'ä¸»': '主',
            'è¦': '要',
            'æ–°': '新',
            'é—»': '闻'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
    
    def get_usage_info(self) -> Dict[str, Any]:
        """
        获取API使用情况
        
        返回:
            API使用情况字典
        """
        if not self.api_key:
            return {"error": "API密钥未设置"}
            
        try:
            params = {"token": self.api_key}
            response = requests.get(f"{self.BASE_URL}/info", params=params)
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