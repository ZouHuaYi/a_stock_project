#!/usr/bin/env python
# -*- coding: utf-8 -*-

from analyzer.chan_analyzer import ChanAnalyzer

if __name__ == "__main__":
    print("开始缠论分析测试...")
    
    # 创建分析实例
    analyzer = ChanAnalyzer(
        symbol='sh000300',  # 沪深300指数
        periods=['daily', '30min', '5min'],  # 分析周期
        data_len=200  # 获取的K线数量
    )
    
    # 运行完整分析
    analyzer.run_full_analysis()
    
    # 获取交易建议
    recommendation = analyzer.get_trading_recommendation()
    
    # 绘制K线和缠论分析图
    print("绘制日线分析图...")
    analyzer.plot_level_analysis('daily', num_records=150)
    
    # 如果有30分钟数据，也绘制30分钟图
    if '30min' in analyzer.periods:
        print("绘制30分钟分析图...")
        analyzer.plot_level_analysis('30min', num_records=200)
    
    print("测试完成") 