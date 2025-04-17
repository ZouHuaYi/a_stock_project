# -*- coding: utf-8 -*-
"""缠论分析器测试脚本"""

import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzer.chan_making_analyzer import ChanMakingAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)

def test_chan_analyzer():
    """测试缠论分析器"""
    
    # 测试股票代码
    stock_code = "605567"  # 沪深300指数
    end_date = datetime.now()
    days = 365  # 分析120天数据
    # 创建分析器实例
    analyzer = ChanMakingAnalyzer(
        stock_code=stock_code,
        end_date=end_date,
        days=days  # 分析120天数据
    )
    
    # 运行分析
    result = analyzer.run_analysis()
    
    # 打印分析结果摘要
    print("\n=== 缠论分析结果摘要 ===")
    print(f"股票: {result['stock_name']}({result['stock_code']})")
    print(f"分析日期: {result['analysis_date']}")
    
    # 打印趋势矩阵
    print("\n--- 趋势矩阵 ---")
    for level, data in result['trend_matrix'].items():
        if level == 'index':
            continue
        print(f"{level}: 趋势={data.get('trend')}, 信号={data.get('signal')}, 背驰={data.get('beichi')}")
    
    # 打印交易信号
    print("\n--- 交易信号 ---")
    signal = result['signal']
    print(f"行动: {signal['action']}")
    print(f"理由: {signal['reason']}")
    print(f"信心度: {signal['confidence']}")
    print(f"止损位: {signal['stop_loss']}")
    
    # 打印图表路径
    print("\n--- 分析图表 ---")
    for level, level_result in result['level_results'].items():
        chart_path = level_result.get('chart_path', '')
        if chart_path:
            print(f"{level}级别图表: {chart_path}")
    
    # 打印LLM分析结果
    if 'llm_analysis' in result and 'result' in result['llm_analysis']:
        print("\n--- LLM增强分析 ---")
        print(result['llm_analysis']['result'])
    
    return result

if __name__ == "__main__":
    try:
        test_result = test_chan_analyzer()
        print("\n测试完成: 成功")
    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)
        print(f"\n测试失败: {str(e)}") 