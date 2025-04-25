#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试中文编码修复功能
"""

import os
import sys
import logging
from dotenv import load_dotenv

# 确保添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载环境变量
load_dotenv()

from utils.scraping_api import ScrapingBeeClient

# 设置日志
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_encoding_fix():
    """测试中文编码修复函数"""
    
    # 初始化客户端
    client = ScrapingBeeClient()
    
    # 测试1: 常见乱码修复
    garbled_text = "æ°æµªè´¢ç»"
    fixed_text = client._fix_chinese_text(garbled_text)
    print(f"原始文本: {garbled_text}")
    print(f"修复后的文本: {fixed_text}")
    print("-" * 50)
    
    # 测试2: 完整HTML页面
    # 模拟一个包含乱码的HTML页面
    garbled_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>æ°æµªè´¢ç» - ä¸­å½é¦é¡µ</title>
    </head>
    <body>
        <h1>æ°æµªè´¢ç»</h1>
        <p>ä¸»è¦æ–°é—»</p>
    </body>
    </html>
    """
    
    fixed_html = client._fix_chinese_encoding(garbled_html)
    print(f"原始HTML: {garbled_html[:100]}...")
    print(f"修复后的HTML: {fixed_html[:100]}...")
    print("-" * 50)
    
    # 测试3: 实际网站测试
    try:
        print("正在抓取实际网站...")
        result = client.scrape_page("https://finance.sina.com.cn/")
        print(f"网站标题: {result.get('title', '无标题')}")
        print(f"内容片段: {result.get('content', '无内容')[:200]}...")
    except Exception as e:
        print(f"抓取实际网站失败: {str(e)}")
    
if __name__ == "__main__":
    test_encoding_fix() 