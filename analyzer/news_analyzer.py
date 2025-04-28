#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票新闻分析器

基于Google搜索API获取股票相关新闻，并进行分析
"""

import os
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple

from analyzer.base_analyzer import BaseAnalyzer

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入jieba分词，如果不存在则提供兼容实现
try:
    import jieba
    logger.debug("成功导入jieba")
except ImportError:
    logger.warning("jieba模块未安装，使用简单的分词方法代替")
    
    class DummyJieba:
        """模拟jieba的简单分词实现"""
        @staticmethod
        def cut(text, cut_all=False):
            """简单的按空格分词"""
            return text.split()
            
    jieba = DummyJieba()

# 尝试导入wordcloud，如果不存在则忽略词云生成
try:
    import wordcloud
    HAS_WORDCLOUD = True
    logger.debug("成功导入wordcloud")
except ImportError:
    HAS_WORDCLOUD = False
    logger.warning("wordcloud模块未安装，词云生成功能将被禁用")

# 尝试导入GoogleSearchAPI，如果失败提供一个空的替代类
try:
    from utils.google_api import GoogleSearchAPI, get_google_search_api
    logger.debug("成功导入GoogleSearchAPI")
except Exception as e:
    logger.error(f"导入GoogleSearchAPI失败: {str(e)}")
    logger.error(traceback.format_exc())
    
    # 创建一个模拟的API类
    class DummyGoogleSearchAPI:
        """GoogleSearchAPI的替代类，用于处理导入失败情况"""
        def __init__(self, *args, **kwargs):
            self.api_key = None
            self.cx = None
            logger.warning("使用DummyGoogleSearchAPI替代，所有API调用将返回空结果")
            
        def search_stock_info(self, *args, **kwargs):
            """模拟搜索方法"""
            return []
            
        def extract_financial_news_content(self, *args, **kwargs):
            """模拟内容提取方法"""
            return {"status": "error", "error": "GoogleSearchAPI不可用"}
    
    def get_google_search_api():
        """获取模拟API实例"""
        return DummyGoogleSearchAPI()

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
        # 从kwargs中提取BaseAnalyzer需要的参数
        stock_code = kwargs.get('stock_code')
        stock_name = kwargs.get('stock_name')
        end_date = kwargs.get('end_date')
        days = kwargs.get('days')
        start_date = kwargs.get('start_date')
        
        logger.info(f"NewsAnalyzer初始化参数: stock_code={stock_code}, days={days}")
        
        # 调用父类初始化方法，只传递父类需要的参数
        try:
            super().__init__(
                stock_code=stock_code,
                stock_name=stock_name,
                end_date=end_date,
                days=days,
                start_date=start_date
            )
            logger.info("BaseAnalyzer初始化完成")
        except Exception as e:
            logger.error(f"BaseAnalyzer初始化失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # 初始化Google搜索API
        try:
            self.search_api = get_google_search_api()
            logger.info("Google搜索API初始化完成")
            
            # 检查API是否可用
            if not self.search_api.api_key or not self.search_api.cx:
                logger.warning("Google搜索API未配置，请设置GOOGLE_API_KEY和GOOGLE_SEARCH_CX环境变量")
        except Exception as e:
            logger.error(f"Google搜索API初始化失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.search_api = DummyGoogleSearchAPI()
            
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
        
        logger.info(f"NewsAnalyzer初始化完成: max_results={self.default_max_results}, deep_crawl={self.enable_deep_crawl}")

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
    
    def fetch_data(self) -> bool:
        """
        获取数据，这里不需要额外的数据获取，直接返回True
        
        返回:
            bool: 是否成功获取数据
        """
        logger.info(f"NewsAnalyzer 不需要额外的数据获取，直接返回True")
        return True
    
    def process_single_url(self, url: str, save_path=None) -> Dict[str, Any]:
        """
        处理单个URL提取命令并格式化输出
        
        参数:
            url: 需要提取的URL
            save_path: 保存结果的路径，如果为None则不保存
            
        返回:
            处理结果字典
        """
        logger.info(f"提取单个新闻URL: {url}")
        
        # 提取新闻内容
        content_data = self.extract_single_news(url)
        
        result = {'status': 'error', 'message': '提取失败'}
        
        if 'data' in content_data:
            # 输出提取结果
            data = content_data['data']
            output = f"新闻提取结果:\n"
            output += f"标题: {data.get('title', '未找到标题')}\n"
            output += f"发布日期: {data.get('publish_date', '未找到日期')}\n"
            output += f"作者/来源: {data.get('author', '未找到作者')}\n"
            output += f"关键词: {data.get('keywords', '未找到关键词')}\n\n"
            output += f"内容:\n{data.get('content', '未找到内容')}\n"
            
            # 添加到结果
            result = {
                'status': 'success',
                'message': '提取成功',
                'data': data,
                'formatted_output': output
            }
            
            # 保存结果
            if save_path:
                output_file = save_path
                if not output_file.endswith('.txt'):
                    output_file += '.txt'
                
                # 确保output目录存在
                os.makedirs('output', exist_ok=True)
                output_path = os.path.join('output', output_file)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output)
                    
                logger.info(f"提取结果已保存到: {output_path}")
                result['output_path'] = output_path
        else:
            # 提取失败
            error_msg = f"从URL提取内容失败: {url}"
            if 'error' in content_data:
                error_msg += f" - {content_data['error']}"
                
            logger.error(error_msg)
            result = {
                'status': 'error',
                'message': error_msg,
                'error': content_data.get('error', '未知错误')
            }
            
        return result
    
    def run_analysis(self, save_path=None, additional_context=None) -> dict:
        """
        运行完整的分析流程
        
        参数:
            save_path: 保存路径，如果为None则不保存结果
            additional_context: 额外的上下文信息，例如其他分析结果
            
        返回:
            分析结果字典
        """
        try:
            logger.info(f"NewsAnalyzer开始运行分析: {self.stock_code}")
            
            # 获取数据
            logger.info("调用fetch_data方法")
            if not self.fetch_data():
                logger.error("数据获取失败")
                return {'status': 'error', 'message': '数据获取失败'}
            
            logger.info("尝试获取股票名称")
            # 获取股票名称，用于搜索
            try:
                from data.stock_data import StockData
                stock_data = StockData()
                logger.info(f"成功创建StockData实例")
                
                stock_info = stock_data.get_stock_info(self.stock_code)
                logger.info(f"获取到股票信息: {stock_info}")
                
                stock_name = stock_info.get('name') if stock_info else None
                logger.info(f"获取到股票名称: {stock_name}")
            except Exception as e:
                logger.error(f"获取股票名称时出错: {str(e)}")
                logger.error(traceback.format_exc())
                stock_name = None
            
            # 执行新闻分析
            logger.info(f"开始执行新闻分析: {self.stock_code} {stock_name}")
            try:
                result = self.analyze(
                    stock_code=self.stock_code, 
                    stock_name=stock_name,
                    max_results=self.default_max_results,
                    days=self.days, 
                    sites=self.default_sites,
                    deep_crawl=self.enable_deep_crawl,
                    deep_crawl_limit=self.deep_crawl_limit
                )
                logger.info(f"新闻分析完成: 状态 {result.get('status')}")
            except Exception as e:
                logger.error(f"执行新闻分析时出错: {str(e)}")
                logger.error(traceback.format_exc())
                return {'status': 'error', 'message': f'新闻分析失败: {str(e)}'}
            
            # 生成格式化输出
            logger.info("生成格式化输出")
            try:
                output = self.format_output(result)
                logger.info("格式化输出生成完成")
            except Exception as e:
                logger.error(f"生成格式化输出时出错: {str(e)}")
                logger.error(traceback.format_exc())
                output = f"格式化输出失败: {str(e)}"
            
            # 如果需要保存结果
            if save_path:
                logger.info(f"保存分析结果到: {save_path}")
                try:
                    output_file = save_path
                    if not output_file.endswith('.txt'):
                        output_file += '.txt'
                    
                    # 确保output目录存在
                    os.makedirs('output', exist_ok=True)
                    output_path = os.path.join('output', output_file)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(output)
                        # 保存完整的新闻信息
                        f.write("\n\n完整新闻列表:\n")
                        for i, news in enumerate(result['news'], 1):
                            f.write(f"{i}. {news['title']}\n")
                            f.write(f"   链接: {news['link']}\n")
                            f.write(f"   摘要: {news['snippet']}\n\n")
                    
                    logger.info(f"分析结果已保存到: {output_path}")
                    result['output_path'] = output_path
                except Exception as e:
                    logger.error(f"保存分析结果时出错: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # 添加格式化输出到结果中
            result['formatted_output'] = output
            
            logger.info("NewsAnalyzer分析完成")
            return result
            
        except Exception as e:
            logger.error(f"新闻分析器运行出错: {str(e)}")
            logger.error(traceback.format_exc())
            return {'status': 'error', 'message': f'新闻分析失败: {str(e)}'}
    
