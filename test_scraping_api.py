#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试ScrapingBee API功能
"""

import os
import json
import logging
import sys
from utils.scraping_api import ScrapingBeeAPI, get_scraping_api
from utils.google_api import get_google_search_api

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_scraping_api():
    """测试ScrapingBee API功能"""
    
    # 获取API实例
    api = get_scraping_api()
    
    # 测试不同的爬取功能
    test_scraping_functions(api)
    
def test_scraping_functions(api):
    """测试各种爬取功能"""
    
    # 测试1: 基本爬取
    print("\n===== 测试1: 基本网页爬取 =====")
    try:
        url = "https://www.baidu.com"
        result = api.scrape_page(url=url)
        
        if result.get('status_code') == 200:
            print(f"✅ 成功！状态码: {result.get('status_code')}")
            content = result.get('content', '')
            if content:
                print(f"内容长度: {len(content)}")
                print(f"内容预览: {content[:100]}...")
            else:
                print("⚠️ 响应成功但无内容返回")
        else:
            print(f"❌ 失败！状态码: {result.get('status_code')}")
            print(f"响应头: {json.dumps(result.get('headers', {}), indent=2)}")
            
    except Exception as e:
        print(f"❌ 基本爬取失败: {str(e)}")
    
    # 测试2: 提取财经新闻
    print("\n===== 测试2: 提取财经新闻 =====")
    try:
        # 使用一个稳定的财经新闻URL
        url = "https://finance.sina.com.cn/stock/marketresearch/2023-08-07/doc-imfzpmir1531371.shtml"
        result = api.extract_financial_news(url)
        
        if 'data' in result:
            data = result['data']
            print("✅ 成功提取数据！")
            print(f"标题: {data.get('title', '未找到标题')}")
            print(f"发布日期: {data.get('publish_date', '未找到日期')}")
            print(f"作者/来源: {data.get('author', '未找到作者')}")
            content = data.get('content', '')
            if content:
                print(f"内容长度: {len(content)}")
                print(f"内容预览: {content[:200]}...")
            else:
                print("⚠️ 未提取到正文内容")
        else:
            print("❌ 提取结构化数据失败")
            print(f"状态码: {result.get('status_code')}")
            if 'content' in result:
                print(f"返回内容预览: {result['content'][:200] if result['content'] else '无内容'}")
    except Exception as e:
        print(f"❌ 提取财经新闻失败: {str(e)}")
    
    # 测试3: 与Google搜索集成
    print("\n===== 测试3: 与Google搜索集成 =====")
    try:
        google_api = get_google_search_api()
        
        if not google_api.api_key or not google_api.cx:
            print("❌ Google搜索API未配置，无法测试集成功能")
        else:
            print("正在进行搜索并深度爬取，这可能需要一些时间...")
            # 搜索并深度爬取
            results = google_api.search_stock_info(
                stock_code="600519", 
                stock_name="贵州茅台", 
                max_results=1,  # 只搜索一个结果，节省API调用
                deep_crawl=True,
                deep_crawl_limit=1  # 只深度爬取一个结果，节省API调用
            )
            
            if results:
                print(f"✅ 成功！找到 {len(results)} 条结果")
                for i, result in enumerate(results, 1):
                    print(f"\n结果 {i}:")
                    print(f"  标题: {result.get('title', '未找到标题')}")
                    print(f"  链接: {result.get('link', '未找到链接')}")
                    
                    if 'extracted_content' in result and result.get('extraction_success', False):
                        content = result.get('extracted_content', '')
                        print(f"  ✅ 深度爬取成功!")
                        print(f"  内容长度: {len(content)} 字符")
                        print(f"  内容预览: {content[:150]}..." if content else "  无内容")
                    else:
                        print("  ❌ 未成功进行深度爬取或爬取失败")
                        if 'extraction_error' in result:
                            print(f"  错误: {result['extraction_error']}")
            else:
                print("❌ 搜索未返回结果")
    except Exception as e:
        print(f"❌ Google搜索集成测试失败: {str(e)}")
    
    # 测试4: 检查API使用情况
    print("\n===== 测试4: 检查API使用情况 =====")
    try:
        usage = api.get_usage_info()
        if 'error' in usage:
            print(f"❌ 获取使用情况失败: {usage['error']}")
        else:
            print("✅ 成功获取API使用情况:")
            print(json.dumps(usage, indent=2))
    except Exception as e:
        print(f"❌ 获取API使用情况失败: {str(e)}")
    
    print("\n===== 测试完成 =====")
    print("如果测试失败，请检查API密钥是否正确，或者ScrapingBee服务是否可用。")
    print("有关更多信息，请参考docs/scrapingbee_setup.md文档。")

if __name__ == "__main__":
    test_scraping_api() 