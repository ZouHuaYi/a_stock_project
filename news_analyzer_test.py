#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立的新闻分析器测试脚本
"""

import os
import sys
import traceback
from datetime import datetime
import argparse

# 设置输出到文件
debug_log = open("debug_output.txt", "w", encoding="utf-8")

def debug_print(message):
    """输出调试信息到文件和控制台"""
    print(message)
    debug_log.write(f"{message}\n")
    debug_log.flush()

# 确保路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 进行直接的错误捕获和输出
try:
    debug_print("测试新闻分析器...")
    
    # 设置参数
    parser = argparse.ArgumentParser(description='新闻分析器测试工具')
    parser.add_argument('stock_code', help='股票代码，如600120')
    parser.add_argument('--days', type=int, default=30, help='回溯天数')
    parser.add_argument('--output', help='输出文件路径')
    
    args = parser.parse_args()
    
    debug_print(f"分析股票: {args.stock_code}, 天数: {args.days}")
    
    # 直接导入必要的模块
    try:
        from analyzer.base_analyzer import BaseAnalyzer
        debug_print("成功导入BaseAnalyzer")
    except ImportError as e:
        debug_print(f"导入BaseAnalyzer失败: {str(e)}")
        debug_print(traceback.format_exc())
        sys.exit(1)
    
    # 尝试导入分词库
    try:
        import jieba
        debug_print("成功导入jieba")
    except ImportError:
        debug_print("jieba分词库未安装，将使用简单分词替代")
        
        # 创建简单的替代实现
        class DummyJieba:
            @staticmethod
            def cut(text, cut_all=False):
                return text.split()
                
        sys.modules['jieba'] = DummyJieba()
        debug_print("已创建jieba替代实现")
    
    # 尝试导入词云库
    try:
        import wordcloud
        debug_print("成功导入wordcloud")
        HAS_WORDCLOUD = True
    except ImportError:
        debug_print("wordcloud库未安装，词云功能将被禁用")
        HAS_WORDCLOUD = False
    
    # 导入新闻分析器
    try:
        from analyzer.news_analyzer import NewsAnalyzer
        debug_print("成功导入NewsAnalyzer")
    except ImportError as e:
        debug_print(f"导入NewsAnalyzer失败: {str(e)}")
        debug_print(traceback.format_exc())
        sys.exit(1)
    
    # 创建分析器实例
    try:
        analyzer_params = {
            'stock_code': args.stock_code,
            'days': args.days,
            'max_news_results': 10,
            'enable_deep_crawl': True,
            'deep_crawl_limit': 3
        }
        
        debug_print(f"创建NewsAnalyzer, 参数: {analyzer_params}")
        analyzer = NewsAnalyzer(**analyzer_params)
        debug_print("成功创建NewsAnalyzer实例")
    except Exception as e:
        debug_print(f"创建NewsAnalyzer实例失败: {str(e)}")
        debug_print(traceback.format_exc())
        sys.exit(1)
    
    # 运行分析
    try:
        debug_print("执行run_analysis...")
        result = analyzer.run_analysis(save_path=args.output)
        
        if result:
            debug_print(f"分析结果状态: {result.get('status', '未知')}")
            
            if result.get('status') == 'success':
                if 'formatted_output' in result:
                    debug_print("\n分析结果:")
                    debug_print(result['formatted_output'])
                
                if 'output_path' in result:
                    debug_print(f"\n结果已保存到: {result['output_path']}")
            else:
                debug_print(f"分析失败: {result.get('message', '未知错误')}")
        else:
            debug_print("分析返回空结果")
            
    except Exception as e:
        debug_print(f"执行分析时出错: {str(e)}")
        debug_print(traceback.format_exc())
    
except Exception as e:
    debug_print(f"发生错误: {str(e)}")
    debug_print(traceback.format_exc())
    sys.exit(1)
finally:
    # 关闭日志文件
    debug_log.close() 