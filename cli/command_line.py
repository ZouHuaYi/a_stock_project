# -*- coding: utf-8 -*-
"""命令行处理模块"""

import argparse
import os
import sys
from datetime import datetime

# 导入日志
from utils.logger import setup_logger

# 创建日志记录器
logger = setup_logger()

def parse_args():
    """
    解析命令行参数
    
    返回:
        argparse.Namespace: 命令行参数
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='A股分析与选股工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 选股子命令
    select_parser = subparsers.add_parser('select', help='选股功能')
    select_parser.add_argument('selector_type', choices=['volume', 'technical', 'combined'], 
                              default='volume', nargs='?', help='选股器类型')
    select_parser.add_argument('--days', type=int, help='回溯数据天数')
    select_parser.add_argument('--threshold', type=float, help='选股分数阈值')
    select_parser.add_argument('--limit', type=int, help='限制结果数量')
    select_parser.add_argument('--output', help='输出文件名')
    
    # 分析子命令
    analyze_parser = subparsers.add_parser('analyze', help='股票分析功能')
    analyze_parser.add_argument('analyzer_type', choices=['volprice', 'golden', 'deepseek'], 
                               default='volprice', nargs='?', help='分析器类型')
    analyze_parser.add_argument('stock_code', help='股票代码')
    analyze_parser.add_argument('--days', type=int, help='回溯数据天数')
    analyze_parser.add_argument('--end-date', help='结束日期，格式：YYYY-MM-DD')
    analyze_parser.add_argument('--save-chart', action='store_true', default=True, help='保存图表')
    analyze_parser.add_argument('--output', help='输出文件名(不含扩展名)')
    
    # 更新数据子命令
    update_parser = subparsers.add_parser('update', help='更新股票数据')
    update_parser.add_argument('--full', action='store_true', help='执行全量更新')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    return args


def handle_select(args):
    """
    处理选股命令
    
    参数:
        args (argparse.Namespace): 命令行参数
    """
    selector_type = args.selector_type
    logger.info(f"开始执行{selector_type}选股...")
    
    try:
        # 根据选择的选股器类型导入相应的选股器
        if selector_type == 'volume':
            from selector.volume_selector import VolumeSelector
            selector = VolumeSelector(days=args.days, threshold=args.threshold, limit=args.limit)
        elif selector_type == 'technical':
            from selector.technical_selector import TechnicalSelector
            selector = TechnicalSelector(days=args.days, threshold=args.threshold, limit=args.limit)
        elif selector_type == 'combined':
            from selector.combined_selector import CombinedSelector
            selector = CombinedSelector(days=args.days, threshold=args.threshold, limit=args.limit)
        else:
            logger.error(f"未知的选股器类型: {selector_type}")
            return
        
        # 执行选股
        results = selector.run_screening()
        
        # 如果有结果，保存
        if not results.empty:
            # 生成输出文件名
            if args.output:
                output_file = args.output
            else:
                output_file = f"{selector_type}_selection_{datetime.now().strftime('%Y%m%d')}.csv"
            
            # 保存结果
            selector.save_results(results, output_file)
        else:
            logger.warning("选股结果为空")
    except Exception as e:
        logger.error(f"执行选股过程中出错: {str(e)}")


def handle_analyze(args):
    """
    处理分析命令
    
    参数:
        args (argparse.Namespace): 命令行参数
    """
    analyzer_type = args.analyzer_type
    stock_code = args.stock_code
    
    logger.info(f"开始执行{analyzer_type}分析，股票代码: {stock_code}...")
    
    try:
        # 根据选择的分析器类型导入相应的分析器
        if analyzer_type == 'volprice':
            from analyzer.vol_price_analyzer import VolPriceAnalyzer
            analyzer = VolPriceAnalyzer(
                stock_code=stock_code, 
                end_date=args.end_date, 
                days=args.days
            )
        elif analyzer_type == 'golden':
            from analyzer.golden_cut_analyzer import GoldenCutAnalyzer
            analyzer = GoldenCutAnalyzer(
                stock_code=stock_code, 
                end_date=args.end_date, 
                days=args.days
            )
        elif analyzer_type == 'deepseek':
            from analyzer.deepseek_analyzer import DeepseekAnalyzer
            analyzer = DeepseekAnalyzer(
                stock_code=stock_code, 
                end_date=args.end_date, 
                days=args.days
            )
        else:
            logger.error(f"未知的分析器类型: {analyzer_type}")
            return
        
        # 执行分析
        result = analyzer.run_analysis()
        
        if result:
            # 打印分析结果
            if 'description' in result:
                print("\n" + result['description'])
                
            # 输出图表路径
            if 'chart_path' in result:
                print(f"\n图表已保存至: {result['chart_path']}")
                
            # 保存分析结果
            if args.output:
                output_file = args.output
            else:
                output_file = f"{stock_code}_{analyzer_type}_{datetime.now().strftime('%Y%m%d')}"
                
            # 保存结果
            output_path = analyzer.save_results(output_file)
            if output_path:
                print(f"\n分析结果已保存至: {output_path}")
        else:
            logger.error("分析失败，未返回结果")
    except Exception as e:
        logger.error(f"执行分析过程中出错: {str(e)}")


def handle_update(args):
    """
    处理数据更新命令
    
    参数:
        args (argparse.Namespace): 命令行参数
    """
    logger.info(f"开始{'全量' if args.full else '增量'}更新股票数据...")
    
    try:
        from data.stock_data import StockDataUpdater
        
        updater = StockDataUpdater()
        if args.full:
            success = updater.full_update()
        else:
            success = updater.incremental_update()
            
        if success:
            logger.info("数据更新成功完成")
        else:
            logger.error("数据更新失败")
    except Exception as e:
        logger.error(f"执行数据更新过程中出错: {str(e)}")


def main():
    """
    主函数，入口点
    """
    # 解析命令行参数
    args = parse_args()
    
    # 根据命令执行相应功能
    if args.command == 'select':
        handle_select(args)
    elif args.command == 'analyze':
        handle_analyze(args)
    elif args.command == 'update':
        handle_update(args)
    else:
        # 如果没有指定命令，显示帮助
        print("请指定要执行的命令，使用 -h 或 --help 获取帮助")
        sys.exit(1) 