#!/usr/bin/env python
# -*- coding: utf-8 -*-

import traceback

print("开始测试")

try:
    from analyzer.base_analyzer import BaseAnalyzer
    print("BaseAnalyzer导入成功")
    
    from analyzer.news_analyzer import NewsAnalyzer
    print("NewsAnalyzer导入成功")
    
    test_params = {
        'stock_code': '600120',
        'days': 30
    }
    print(f"参数: {test_params}")
    
    analyzer = NewsAnalyzer(**test_params)
    print("创建NewsAnalyzer成功")
    
    result = analyzer.run_analysis()
    print(f"结果: {result}")
    
except Exception as e:
    print(f"错误: {type(e).__name__}: {str(e)}")
    traceback.print_exc()
    
print("测试结束") 