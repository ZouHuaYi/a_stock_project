#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票新闻分析器

基于Google搜索API获取股票相关新闻，并进行分析
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple

from analyzer.base_analyzer import BaseAnalyzer
from utils.google_api import GoogleSearchAPI, get_google_search_api

# 配置日志
logger = logging.getLogger(__name__)

class NewsAnalyzer(BaseAnalyzer):
    """
    股票新闻分析器
    
    使用Google搜索API获取股票相关新闻，并进行分析
    """
    
    def __init__(self, **kwargs):
        """
        初始化股票新闻分析器
        
        参数:
            **kwargs: 其他参数传递给父类
        """
        super().__init__(**kwargs)
        
        # 初始化Google搜索API
        self.search_api = get_google_search_api()
        
        # 检查API是否可用
        if not self.search_api.api_key or not self.search_api.cx:
            logger.warning("Google搜索API未配置，请设置GOOGLE_API_KEY和GOOGLE_SEARCH_CX环境变量")
            
        # 搜索配置
        self.default_max_results = kwargs.get('max_news_results', 10)
        self.default_days = kwargs.get('news_days', 7)
        self.default_sites = kwargs.get('news_sites', [
            'finance.sina.com.cn',
            'finance.eastmoney.com',
            'finance.qq.com',
            'business.sohu.com',
            'money.163.com'
        ])
        
        # 深度爬取配置
        self.enable_deep_crawl = kwargs.get('enable_deep_crawl', True)
        self.deep_crawl_limit = kwargs.get('deep_crawl_limit', 3)

    def analyze(self, stock_code: str, stock_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        分析股票相关新闻
        
        参数:
            stock_code: 股票代码
            stock_name: 股票名称，如果为None则只搜索股票代码
            **kwargs: 其他参数
        
        返回:
            分析结果字典，包含新闻列表、情感分析等
        """
        logger.info(f"分析股票 {stock_code} {stock_name or ''} 的相关新闻")
        
        # 检查API是否可用
        if not self.search_api.api_key or not self.search_api.cx:
            return {
                'status': 'error',
                'message': 'Google搜索API未配置',
                'news': []
            }
        
        # 获取搜索参数
        max_results = kwargs.get('max_results', self.default_max_results)
        days = kwargs.get('days', self.default_days)
        sites = kwargs.get('sites', self.default_sites)
        
        # 获取深度爬取参数
        deep_crawl = kwargs.get('deep_crawl', self.enable_deep_crawl)
        deep_crawl_limit = kwargs.get('deep_crawl_limit', self.deep_crawl_limit)
        
        try:
            # 获取股票相关新闻
            news_results = self.search_api.search_stock_info(
                stock_code=stock_code,
                stock_name=stock_name,
                max_results=max_results,
                days=days,
                site_list=sites,
                deep_crawl=deep_crawl,
                deep_crawl_limit=deep_crawl_limit
            )
            
            # 对新闻进行简单分析
            sentiment, keyword_stats, content_summary = self._analyze_news_content(news_results)
            
            return {
                'status': 'success',
                'message': f'找到 {len(news_results)} 条相关新闻',
                'news': news_results,
                'sentiment': sentiment,
                'keyword_stats': keyword_stats,
                'content_summary': content_summary,
                'has_deep_content': any('extracted_content' in item for item in news_results)
            }
            
        except Exception as e:
            logger.error(f"获取股票新闻时出错: {str(e)}")
            return {
                'status': 'error',
                'message': f'获取新闻失败: {str(e)}',
                'news': []
            }
    
    def _analyze_news_content(self, news_results: List[Dict[str, Any]]) -> Tuple[str, Dict[str, int], str]:
        """
        简单分析新闻内容，提取关键词和情感
        
        参数:
            news_results: 新闻结果列表
            
        返回:
            (情感评估, 关键词统计, 内容总结)
        """
        if not news_results:
            return "neutral", {}, ""
            
        # 正面词汇
        positive_words = [
            '上涨', '增长', '盈利', '利好', '突破', '看好', '强势', 
            '机会', '牛市', '反弹', '回升', '获利', '增持', '推荐'
        ]
        
        # 负面词汇
        negative_words = [
            '下跌', '亏损', '利空', '跌破', '看空', '弱势',
            '风险', '熊市', '下滑', '回落', '亏损', '减持', '谨慎'
        ]
        
        # 统计词频
        keyword_stats = {}
        positive_count = 0
        negative_count = 0
        
        # 内容总结
        content_summary = ""
        has_deep_content = False
        
        # 分析所有新闻
        for news in news_results:
            # 首先使用标题和摘要（基本搜索信息）
            text = (news.get('title', '') + ' ' + news.get('snippet', '')).lower()
            
            # 如果有深度爬取的内容，添加到分析文本中
            if 'extracted_content' in news and news.get('extraction_success', False):
                has_deep_content = True
                extracted_content = news.get('extracted_content', '')
                if extracted_content:
                    text += ' ' + extracted_content.lower()
            
            # 统计正面词汇
            for word in positive_words:
                if word in text:
                    positive_count += 1
                    keyword_stats[word] = keyword_stats.get(word, 0) + 1
                    
            # 统计负面词汇
            for word in negative_words:
                if word in text:
                    negative_count += 1
                    keyword_stats[word] = keyword_stats.get(word, 0) + 1
        
        # 确定整体情感
        if positive_count > negative_count * 1.5:
            sentiment = "positive"
        elif negative_count > positive_count * 1.5:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # 创建内容总结
        if has_deep_content:
            # 整理一个简单的内容总结
            word_count = sum(len(news.get('extracted_content', '').split()) for news in news_results if 'extracted_content' in news)
            content_summary = f"已深度爬取{sum(1 for n in news_results if 'extracted_content' in n)}篇新闻，共{word_count}个词。"
            
            # 添加最长文章的标题
            longest_article = max(
                [n for n in news_results if 'extracted_content' in n], 
                key=lambda x: len(x.get('extracted_content', '')),
                default=None
            )
            if longest_article:
                content_summary += f"\n最详细的文章：{longest_article.get('title', '')}"
            
        return sentiment, keyword_stats, content_summary
    
    def get_related_keywords(self, stock_code: str, stock_name: Optional[str] = None) -> List[str]:
        """
        获取股票相关关键词
        
        参数:
            stock_code: 股票代码
            stock_name: 股票名称
            
        返回:
            关键词列表
        """
        default_keywords = ["财报", "业绩", "利润", "营收", "投资者", "分析师"]
        
        if not stock_name:
            return default_keywords
            
        # 使用股票名称构建关键词
        industry_keywords = []
        
        # 根据股票名称猜测行业
        if "银行" in stock_name:
            industry_keywords = ["存款", "贷款", "不良资产", "净息差", "金融"]
        elif "保险" in stock_name:
            industry_keywords = ["保费", "赔付率", "投资收益", "准备金", "金融"]
        elif "科技" in stock_name or "电子" in stock_name:
            industry_keywords = ["研发", "创新", "专利", "技术", "数字化"]
        elif "医药" in stock_name or "生物" in stock_name:
            industry_keywords = ["新药", "研发", "批准", "临床", "医保"]
        elif "地产" in stock_name or "房" in stock_name:
            industry_keywords = ["销售额", "土地", "楼市", "政策", "调控"]
        
        return default_keywords + industry_keywords
    
    def format_output(self, result: Dict[str, Any]) -> str:
        """
        格式化输出分析结果
        
        参数:
            result: 分析结果字典
            
        返回:
            格式化的输出字符串
        """
        if result['status'] == 'error':
            return f"新闻分析失败: {result['message']}"
            
        output = f"新闻分析结果: 找到{len(result['news'])}条相关新闻\n"
        
        # 添加内容总结（如果有深度爬取）
        if result.get('content_summary'):
            output += f"内容分析: {result['content_summary']}\n"
            
        output += f"整体情感: "
        
        sentiment = result.get('sentiment', 'neutral')
        if sentiment == 'positive':
            output += "正面 📈\n"
        elif sentiment == 'negative':
            output += "负面 📉\n"
        else:
            output += "中性 ➡️\n"
            
        # 添加关键词统计
        keyword_stats = result.get('keyword_stats', {})
        if keyword_stats:
            output += "\n关键词出现频率:\n"
            for word, count in sorted(keyword_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
                output += f"- {word}: {count}次\n"
                
        # 添加新闻列表
        output += "\n最新相关新闻:\n"
        for i, news in enumerate(result['news'][:5], 1):
            output += f"{i}. {news['title']}\n"
            output += f"   来源: {news['display_link']}\n"
            
            # 如果有深度爬取的内容，添加内容预览
            if 'extracted_content' in news and news.get('extraction_success', False):
                content = news.get('extracted_content', '')
                if content:
                    # 截取前150个字符作为预览
                    preview = content[:150] + ('...' if len(content) > 150 else '')
                    output += f"   内容预览: {preview}\n"
                    
            output += f"   链接: {news['link']}\n\n"
            
        return output
    
    def extract_single_news(self, url: str) -> Dict[str, Any]:
        """
        提取单个新闻页面的详细内容
        
        参数:
            url: 新闻URL
            
        返回:
            提取的内容字典
        """
        return self.search_api.extract_financial_news_content(url) 