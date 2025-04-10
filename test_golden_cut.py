# -*- coding: utf-8 -*-
"""斐波那契黄金分割分析器测试脚本"""

import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入分析器
from analyzer.golden_cut_analyzer import GoldenCutAnalyzer

def test_golden_cut_analyzer():
    """测试黄金分割分析器的功能"""
    print("开始测试黄金分割分析器...")
    
    # 创建一个结束日期（使用前一个交易日以确保有数据）
    end_date = datetime.now() - timedelta(days=1)
    
    # 测试不同的股票
    test_stocks = [
        {"code": "000001", "name": "平安银行", "days": 365},
        {"code": "600519", "name": "贵州茅台", "days": 180},
        {"code": "300750", "name": "宁德时代", "days": 90}
    ]
    
    for stock in test_stocks:
        print(f"\n测试 {stock['name']}({stock['code']}) 的黄金分割分析...")
        
        # 创建分析器实例
        analyzer = GoldenCutAnalyzer(
            stock_code=stock['code'],
            stock_name=stock['name'],
            end_date=end_date,
            days=stock['days']
        )
        
        # 运行分析
        result = analyzer.run_analysis()
        
        # 显示结果
        print(f"分析结果状态: {result['status']}")
        print(f"分析描述: {result['description']}")
        
        if 'chart_path' in result:
            print(f"图表已保存至: {result['chart_path']}")
        else:
            print("未生成图表")
        
        if result['status'] == 'error':
            print(f"错误信息: {result.get('message', '未知错误')}")

if __name__ == "__main__":
    test_golden_cut_analyzer() 