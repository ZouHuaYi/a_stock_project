# -*- coding: utf-8 -*-
"""分析器测试脚本，用于测试各类分析器功能"""

import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import (
    BaseAnalyzer,
    VolPriceAnalyzer,
    GoldenCutAnalyzer,
    DeepseekAnalyzer
)

def test_base_analyzer(stock_code, stock_name=None):
    """测试基础分析器"""
    print(f"\n===== 测试基础分析器 =====")
    analyzer = BaseAnalyzer(stock_code, stock_name)
    
    # 获取股票数据
    if analyzer.fetch_data():
        print(f"成功获取 {analyzer.stock_name}({analyzer.stock_code}) 的数据")
        print(f"数据范围: {analyzer.start_date_str} 至 {analyzer.end_date_str}")
        print(f"数据点数: {len(analyzer.daily_data)}")
        
        # 显示最新数据
        if not analyzer.daily_data.empty:
            latest = analyzer.daily_data.iloc[-1]
            print(f"\n最新交易日 ({analyzer.daily_data.index[-1].strftime('%Y-%m-%d')}) 数据:")
            print(f"  开盘: {latest['open']:.2f}")
            print(f"  最高: {latest['high']:.2f}")
            print(f"  最低: {latest['low']:.2f}")
            print(f"  收盘: {latest['close']:.2f}")
            print(f"  成交量: {latest['volume']:.0f}")
    else:
        print(f"获取数据失败")

def test_vol_price_analyzer(stock_code, stock_name=None):
    """测试量价分析器"""
    print(f"\n===== 测试量价分析器 =====")
    # 创建保存目录
    save_path = "./output/vol_price"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 创建分析器实例
    analyzer = VolPriceAnalyzer(
        stock_code=stock_code,
        stock_name=stock_name,
        days=60,
        save_path=save_path
    )
    
    # 运行分析
    print(f"开始分析 {stock_name or stock_code} 的量价关系...")
    result = analyzer.run_analysis()
    
    # 打印分析结果
    if result['status'] == 'success':
        print("\n分析结果:")
        print(f"股票: {result['stock_name']}({result['stock_code']})")
        print(f"日期: {result['date']}")
        
        if result['is_washing']:
            print(f"洗盘可能性: {result['wash_confidence']}%")
        else:
            print("未检测到明显洗盘")
            
        print("\n检测到的特征:")
        for pattern in result['patterns']:
            print(f"- {pattern['description']}")
            
        print("\n分析结论:")
        print(result['description'])
        print("\n分析图表已保存至:", save_path)
    else:
        print(f"分析失败: {result.get('message', '未知错误')}")

def test_golden_cut_analyzer(stock_code, stock_name=None):
    """测试黄金分割分析器"""
    print(f"\n===== 测试黄金分割分析器 =====")
    # 创建保存目录
    save_path = "./output/golden_cut"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 创建分析器实例
    analyzer = GoldenCutAnalyzer(
        stock_code=stock_code,
        stock_name=stock_name,
        days=365,
        save_path=save_path
    )
    
    # 运行分析
    print(f"开始分析 {stock_name or stock_code} 的斐波那契回调水平...")
    result = analyzer.run_analysis()
    
    # 打印分析结果
    if result['status'] == 'success':
        print("\n分析结果:")
        print(f"股票: {result['stock_name']}({result['stock_code']})")
        print(f"日期: {result['date']}")
        print(f"当前价格: {result['current_price']:.2f}")
        
        if result['closest_support']:
            support_name, support_price = result['closest_support']
            print(f"最近支撑位: {support_name} ({support_price:.2f})")
            print(f"距支撑位: {result['support_distance_percent']:.2f}%")
            
        if result['closest_resistance']:
            resistance_name, resistance_price = result['closest_resistance']
            print(f"最近阻力位: {resistance_name} ({resistance_price:.2f})")
            print(f"距阻力位: {result['resistance_distance_percent']:.2f}%")
            
        print("\n分析结论:")
        print(result['description'])
        print("\n分析图表已保存至:", save_path)
    else:
        print(f"分析失败: {result.get('message', '未知错误')}")

def test_deepseek_analyzer(stock_code, stock_name=None):
    """测试大模型分析器"""
    print(f"\n===== 测试大模型分析器 =====")
    # 创建保存目录
    save_path = "./output/deepseek"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 创建分析器实例
    analyzer = DeepseekAnalyzer(
        stock_code=stock_code,
        stock_name=stock_name,
        days=365,
        save_path=save_path
    )
    
    # 运行分析
    print(f"开始使用大模型分析 {stock_name or stock_code}...")
    result = analyzer.run_analysis(additional_context="请关注该股票近期的市场表现和行业动态")
    
    # 打印分析结果
    if result['status'] == 'success':
        print("\n分析结果:")
        print(f"股票: {result['stock_name']}({result['stock_code']})")
        print(f"日期: {result['date']}")
        
        # 打印技术分析摘要
        if result['technical_summary']:
            print("\n技术分析摘要:")
            for key, value in result['technical_summary'].items():
                if key not in ['股票代码', '股票名称']:
                    print(f"  {key}: {value}")
        
        # 打印AI分析摘要
        if result['ai_analysis'] and result['ai_analysis'] != "AI分析模块未加载，无法生成AI分析报告":
            print("\nAI分析摘要:")
            # 只打印前300个字符的摘要
            print(f"{result['ai_analysis'][:300]}..." if len(result['ai_analysis']) > 300 else result['ai_analysis'])
            
        if result['report_path']:
            print(f"\n完整分析报告已保存至: {result['report_path']}")
        
        print("\n分析图表已保存至:", save_path)
    else:
        print(f"分析失败: {result.get('message', '未知错误')}")

def batch_test_analyzers(stock_list):
    """批量测试多只股票的所有分析器"""
    for stock in stock_list:
        stock_code = stock['code']
        stock_name = stock.get('name')
        
        print(f"\n\n{'='*60}")
        print(f"开始分析 {stock_name or stock_code}")
        print(f"{'='*60}")
        
        try:
            # 测试基础分析器
            test_base_analyzer(stock_code, stock_name)
            
            # 测试量价分析器
            test_vol_price_analyzer(stock_code, stock_name)
            
            # 测试黄金分割分析器
            test_golden_cut_analyzer(stock_code, stock_name)
            
            # 测试大模型分析器
            test_deepseek_analyzer(stock_code, stock_name)
        except Exception as e:
            print(f"分析 {stock_name or stock_code} 时出错: {e}")

if __name__ == "__main__":
    # 测试单个股票的所有分析器
    # test_base_analyzer("000001", "平安银行")
    # test_vol_price_analyzer("000001", "平安银行")
    # test_golden_cut_analyzer("000001", "平安银行")
    # test_deepseek_analyzer("000001", "平安银行")
    
    # 批量测试多只股票
    stocks_to_test = [
        {"code": "000001", "name": "平安银行"},
        {"code": "600519", "name": "贵州茅台"},
        {"code": "000858", "name": "五粮液"}
    ]
    
    batch_test_analyzers(stocks_to_test) 