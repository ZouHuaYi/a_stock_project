#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试NewsAnalyzer类的简单脚本
"""

import sys
import os
import traceback
import logging
from datetime import datetime

try:
    # 启用简单的控制台日志记录
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("test_news_analyzer")
    logger.info("开始测试NewsAnalyzer")
    
    # 尝试导入相关模块
    logger.info("导入BaseAnalyzer")
    from analyzer.base_analyzer import BaseAnalyzer
    logger.info("BaseAnalyzer导入成功")
    
    logger.info("导入NewsAnalyzer")
    from analyzer.news_analyzer import NewsAnalyzer
    logger.info("NewsAnalyzer导入成功")
    
    # 设置测试参数
    test_params = {
        'stock_code': '600120',
        'days': 30,
        'max_news_results': 5,
        'enable_deep_crawl': False  # 禁用深度爬取以加快测试
    }
    
    logger.info(f"使用参数创建NewsAnalyzer：{test_params}")
    
    # 创建实例
    analyzer = NewsAnalyzer(**test_params)
    logger.info("NewsAnalyzer创建成功")
    
    logger.info("测试分析方法")
    
    # 执行分析
    result = analyzer.run_analysis()
    logger.info(f"分析结果: {result}")
    
    logger.info("测试完成")
        
except Exception as e:
    print(f"错误: {str(e)}")
    traceback.print_exc() 