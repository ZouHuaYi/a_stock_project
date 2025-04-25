#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Google搜索API功能
"""

import os
import logging
from utils.google_api import GoogleSearchAPI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_google_search():
    """测试Google搜索功能"""
    # 创建API实例
    api = GoogleSearchAPI()
    
    # 测试不同的搜索功能
    test_search_functions(api)
    
def test_search_functions(api):
    """测试各种搜索功能"""
    
    # 测试1: 基本搜索
    print("\n===== 测试1: 基本搜索 =====")
    try:
        results = api.get_search_results("中国股市 上证指数", max_results=3)
        print_results(results)
    except Exception as e:
        print(f"基本搜索失败: {str(e)}")
    
    # 测试2: 特定股票搜索
    print("\n===== 测试2: 特定股票搜索 =====")
    try:
        results = api.search_stock_info(
            stock_code="600519", 
            stock_name="贵州茅台",
            max_results=3
        )
        print_results(results)
    except Exception as e:
        print(f"特定股票搜索失败: {str(e)}")
    
    # 测试3: 限制特定网站搜索
    print("\n===== 测试3: 限制特定网站搜索 =====")
    try:
        results = api.search_stock_info(
            stock_code="000001", 
            stock_name="平安银行",
            max_results=3,
            site_list=["finance.sina.com.cn"]
        )
        print_results(results)
    except Exception as e:
        print(f"特定网站搜索失败: {str(e)}")

def print_results(results):
    """打印搜索结果"""
    if not results:
        print("没有找到结果")
        return
        
    print(f"找到 {len(results)} 条结果:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   链接: {result['link']}")
        print(f"   摘要: {result['snippet']}")
        print()

if __name__ == "__main__":
    test_google_search() 