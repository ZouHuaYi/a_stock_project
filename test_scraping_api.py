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
        url = "https://finance.sina.com.cn/"  # 更新为新浪财经首页，更稳定
        result = api.extract_financial_news(url)
        
        if result.get('status_code') == 200:
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
                print("✅ 成功访问但未提取到结构化数据")
                print(f"内容长度: {len(result.get('content', ''))}")
        else:
            print("❌ 提取结构化数据失败")
            print(f"状态码: {result.get('status_code')}")
            if 'error' in result:
                print(f"错误: {result.get('error')[:200]}")
    except Exception as e:
        print(f"❌ 提取财经新闻失败: {str(e)}")
    
    # 测试3: 单独测试深度爬取
    print("\n===== 测试3: 深度爬取测试 =====")
    try:
        # 创建一个简单的测试URL列表
        test_urls = [
            {
                "title": "新浪财经首页",
                "link": "https://finance.sina.com.cn/",
                "type": "财经网站"
            },
            {
                "title": "百度首页",
                "link": "https://www.baidu.com",
                "type": "搜索引擎"
            }
        ]
        
        for url_info in test_urls:
            url = url_info["link"]
            print(f"\n测试URL: {url_info['title']} ({url})")
            
            if "财经" in url_info["type"]:
                result = api.extract_financial_news(url)
            else:
                result = api.extract_article_content(url)
                
            if result.get('status_code') == 200:
                print(f"✅ 爬取成功！状态码: {result.get('status_code')}")
                
                # 检查是否提取到内容
                if 'data' in result and result['data']:
                    data = result['data']
                    print("内容提取情况:")
                    
                    # 标题
                    title = data.get('title', '')
                    if title:
                        if isinstance(title, list):
                            title = title[0] if title else ''
                        print(f"- 标题: {title}")
                    else:
                        print("- 标题: 未提取到")
                        
                    # 正文
                    content = data.get('content', '')
                    if content:
                        if isinstance(content, list):
                            content = "\n".join(content) if content else ''
                        content_len = len(content)
                        print(f"- 内容: 已提取 ({content_len} 字符)")
                        print(f"  预览: {content[:100]}..." if content_len > 100 else f"  全文: {content}")
                    else:
                        print("- 内容: 未提取到")
                        
                    # 发布日期
                    date = data.get('publish_date', '')
                    if date:
                        print(f"- 发布日期: {date}")
                    
                    # 作者/来源
                    author = data.get('author', '')
                    if author:
                        print(f"- 作者/来源: {author}")
                else:
                    print("❌ 未能提取到结构化内容")
                    content = result.get('content', '')
                    print(f"原始内容长度: {len(content)}")
                    print(f"内容预览: {content[:100]}..." if content else "无内容")
            else:
                print(f"❌ 爬取失败！状态码: {result.get('status_code')}")
                if 'error' in result:
                    print(f"错误信息: {result.get('error')[:200]}")
    except Exception as e:
        print(f"❌ 深度爬取测试失败: {str(e)}")
    
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