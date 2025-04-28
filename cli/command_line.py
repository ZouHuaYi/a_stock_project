# -*- coding: utf-8 -*-
"""命令行处理模块"""

import argparse
import os
import sys
from datetime import datetime
import re
import json
import traceback

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
    select_parser = subparsers.add_parser('selector', help='选股功能')
    select_parser.add_argument('selector_type', choices=['volume', 'chan', 'ma240', 'subject', 'chanbc'], 
                              default='volume', nargs='?', help='选股器类型')
    select_parser.add_argument('--days', type=int, help='回溯数据天数')
    select_parser.add_argument('--threshold', type=float, help='选股分数阈值')
    select_parser.add_argument('--limit', type=int, help='限制结果数量')
    select_parser.add_argument('--output', help='输出文件名')
    select_parser.add_argument('--text', help='题材选股的分析文本内容')
    select_parser.add_argument('--ai-type', choices=['gemini', 'openai'], 
                              default='openai', help='题材选股使用的AI类型')
    
    # 分析子命令
    analyzer_parser = subparsers.add_parser('analyzer', help='股票分析功能')
    analyzer_parser.add_argument('analyzer_type', choices=['volprice', 'golden', 'ai', 'chan', 'news'], 
                               default='volprice', nargs='?', help='分析器类型')
    analyzer_parser.add_argument('stock_code', help='股票代码，如：000001、600001等6位数字')
    analyzer_parser.add_argument('--days', type=int, help='回溯数据天数')
    analyzer_parser.add_argument('--end-date', help='结束日期，格式：YYYY-MM-DD')
    analyzer_parser.add_argument('--save-chart', action='store_true', default=True, help='保存图表')
    analyzer_parser.add_argument('--output', help='输出文件名(不含扩展名)')
    analyzer_parser.add_argument('--ai-type', choices=['gemini', 'openai'], help='AI类型')
    analyzer_parser.add_argument('--sites', help='新闻分析指定网站(逗号分隔)')
    analyzer_parser.add_argument('--max-results', type=int, default=10, help='新闻分析返回的最大结果数')
    analyzer_parser.add_argument('--deep-crawl', action='store_true', help='启用深度爬取功能')
    analyzer_parser.add_argument('--no-deep-crawl', action='store_false', dest='deep_crawl', help='禁用深度爬取功能')
    analyzer_parser.add_argument('--deep-crawl-limit', type=int, default=3, help='深度爬取的最大结果数')
    analyzer_parser.add_argument('--url', help='提取单个新闻URL的内容')
    
    # 更新数据子命令
    update_parser = subparsers.add_parser('update', help='更新股票数据')
    update_parser.add_argument('--full', action='store_true', help='执行全量更新')
    update_parser.add_argument('--basic', action='store_true', help='执行股票基本信息更新')
    update_parser.add_argument('--daily', action='store_true', help='执行股票日线数据更新')
    update_parser.add_argument('--self', action='store_true', help='执行股票自选数据更新')
    update_parser.add_argument('--start-date', help='开始日期，格式：YYYYMMDD')
    update_parser.add_argument('--end-date', help='结束日期，格式：YYYYMMDD')
    
    # 添加CSV业务分析子命令
    collect_parser = subparsers.add_parser('collect', help='选股结果汇总分析')
    collect_parser.add_argument('collect_type', choices=['business'], 
                              default='business', nargs='?', help='汇总分析类型')
    collect_parser.add_argument('--date', help='指定日期(格式：YYYYMMDD)，默认为当天')
    collect_parser.add_argument('--output', help='输出文件名(不含扩展名)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查是否缺少必要的参数
    if args.command == 'analyzer':
        # 如果参数不足2个，可能是没有提供股票代码
        if len(sys.argv) < 4:
            analyzer_parser.error("analyzer命令需要提供分析器类型和股票代码两个参数")
    
    # 检查题材选股必要参数
    if args.command == 'selector' and args.selector_type == 'subject' and not args.text:
        select_parser.error("使用题材选股器需要提供--text参数指定分析文本")
    
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
        elif selector_type == 'chan':
            from selector.chan_selector import ChanSelector
            selector = ChanSelector(days=args.days, threshold=args.threshold, limit=args.limit)
        elif selector_type == 'ma240':
            from selector.ma240_selector import Ma240Selector
            selector = Ma240Selector(days=args.days, threshold=args.threshold, limit=args.limit)
        elif selector_type == 'subject':
            from selector.subject_selector import SubjectSelector
            selector = SubjectSelector(
                days=args.days, 
                threshold=args.threshold, 
                limit=args.limit,
                text=args.text,
                ai_type=args.ai_type
            )
        elif selector_type == 'chanbc':
            from selector.chanbc_selector import ChanBackchSelector
            selector = ChanBackchSelector(days=args.days, threshold=args.threshold, limit=args.limit)
        else:
            logger.error(f"未知的选股器类型: {selector_type}")
            return
        # 生成输出文件名
        if args.output:
            output_file = args.output
        else:
            output_file = f"{selector_type}_selection_{datetime.now().strftime('%Y%m%d')}.csv"
        # 执行选股
        filepath = selector.run_screening(output_file)
        if filepath:
            logger.info(f"选股结果已保存到: {filepath}")
        else:
            logger.warning("选股结果为空")
    except Exception as e:
        logger.error(f"执行选股过程中出错: {str(e)}")


def handle_analyzer(args):
    """
    处理分析命令
    
    参数:
        args (argparse.Namespace): 命令行参数
    """
    analyzer_type = args.analyzer_type
    stock_code = args.stock_code
    
    # 验证股票代码格式
    valid_code = False
    # 验证是否为6位数字，或者以特定规则开头的代码
    if re.match(r'^\d{6}$', stock_code) or \
       re.match(r'^(sh|sz|SH|SZ)\d{6}$', stock_code) or \
       re.match(r'^(00|60|30)\d{4}$', stock_code):
        valid_code = True
    
    if not valid_code:
        logger.error(f"股票代码格式无效: {stock_code}，正确格式应为6位数字，如'000001'或'600001'")
        return
    
    logger.info(f"开始执行{analyzer_type}分析，股票代码: {stock_code}...")
    
    # 对于news分析，使用完全独立的处理流程，减少依赖关系
    if analyzer_type == 'news':
        handle_news_analyzer(args)
        return
    
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
        elif analyzer_type == 'ai':
            from analyzer.ai_analyzer import AiAnalyzer
            if args.ai_type:
                ai_type = args.ai_type
            else:
                ai_type = "openai"

            analyzer = AiAnalyzer(
                stock_code=stock_code, 
                end_date=args.end_date, 
                days=args.days,
                ai_type=ai_type
            )
        elif analyzer_type == 'chan':
            from analyzer.chan_making_analyzer import ChanMakingAnalyzer
            analyzer = ChanMakingAnalyzer(
                stock_code=stock_code, 
                end_date=args.end_date, 
                days=args.days
            )
        else:
            logger.error(f"未知的分析器类型: {analyzer_type}")
            return
        
        # 执行分析
        try:
            result = analyzer.run_analysis(save_path=args.output)
            
            if result:
                # 输出处理完成信息
                logger.info(f"处理完成")
            else:
                logger.error("分析失败，未返回结果")
        except Exception as e:
            logger.error(f"执行分析run_analysis时出错: {str(e)}")
            logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"执行分析过程中出错: {str(e)}")
        logger.error(traceback.format_exc())


def handle_news_analyzer(args):
    """
    处理新闻分析命令（作为独立函数处理）
    
    参数:
        args (argparse.Namespace): 命令行参数
    """
    stock_code = args.stock_code
    
    try:
        print(f"开始处理股票 {stock_code} 的新闻分析...")
        
        # 单独处理新闻分析
        # 1. 设置基础参数
        base_params = {
            'stock_code': stock_code,
            'days': args.days,
            'end_date': args.end_date
        }
        
        # 2. 新闻特有参数
        news_params = {
            'max_news_results': args.max_results if hasattr(args, 'max_results') else 10,
            'enable_deep_crawl': args.deep_crawl if hasattr(args, 'deep_crawl') else True,
            'deep_crawl_limit': args.deep_crawl_limit if hasattr(args, 'deep_crawl_limit') else 3
        }
        
        # 3. 添加网站参数
        if args.sites:
            sites_list = [site.strip() for site in args.sites.split(',')]
            news_params['news_sites'] = sites_list
        
        # 4. 合并所有参数
        analyzer_params = {**base_params, **news_params}
        print(f"新闻分析参数: {analyzer_params}")
        
        # 5. 检查是否是单个URL提取模式
        if args.url:
            from analyzer.news_analyzer import NewsAnalyzer
            analyzer = NewsAnalyzer(**analyzer_params)
            
            print(f"处理单个URL: {args.url}")
            result = analyzer.process_single_url(args.url, save_path=args.output)
            
            # 打印结果
            if result['status'] == 'success':
                print(f"URL处理成功: {args.url}")
                if 'formatted_output' in result:
                    print(result['formatted_output'])
            else:
                print(f"URL处理失败: {result.get('message', '未知错误')}")
                if 'error' in result:
                    print(f"错误: {result['error']}")
            
            return
            
        # 6. 执行完整分析
        try:
            # 先导入模块
            print("导入NewsAnalyzer模块...")
            from analyzer.news_analyzer import NewsAnalyzer
            
            # 创建分析器实例
            print("创建NewsAnalyzer实例...")
            analyzer = NewsAnalyzer(**analyzer_params)
            
            # 执行分析
            print("执行run_analysis方法...")
            result = analyzer.run_analysis(save_path=args.output)
            
            # 处理结果
            if result:
                print("分析完成，打印结果")
                # 打印格式化输出
                if 'formatted_output' in result:
                    print(result['formatted_output'])
                
                # 处理输出路径
                if 'output_path' in result:
                    print(f"结果已保存到: {result['output_path']}")
                
                # 处理错误结果
                if result.get('status') == 'error':
                    print(f"分析失败: {result.get('message', '未知错误')}")
            else:
                print("分析返回空结果")
                
        except ImportError as e:
            print(f"导入错误，可能缺少必要的库: {str(e)}")
            print("您可能需要安装以下库：")
            print("- jieba: 中文分词库，使用 'pip install jieba' 安装")
            print("- wordcloud: 词云生成库，使用 'pip install wordcloud' 安装")
            logger.error(f"缺少必要的库: {str(e)}")
            logger.error(traceback.format_exc())
            
        except Exception as e:
            print(f"执行新闻分析时出错: {str(e)}")
            logger.error(f"执行新闻分析时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
    except Exception as e:
        print(f"处理新闻分析命令时出错: {str(e)}")
        print(traceback.format_exc())
        logger.error(f"处理新闻分析命令时出错: {str(e)}")
        logger.error(traceback.format_exc())


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
        elif args.basic:
            success = updater.init_stock_basic()
        elif args.daily:
            success = updater.init_daily_data()
        elif args.self:
            success = updater.update_stock_self_data(args.start_date, args.end_date)
        if success:
            logger.info("数据更新成功完成")
        else:
            logger.error("数据更新失败")
    except Exception as e:
        logger.error(f"执行数据更新过程中出错: {str(e)}")


def handle_collect(args):
    """
    处理选股结果汇总分析命令
    
    参数:
        args (argparse.Namespace): 命令行参数
    """
    collect_type = args.collect_type
    logger.info(f"开始执行{collect_type}汇总分析...")
    
    try:
        if collect_type == 'business':
            from collect.csv_business import CSVBusinessAnalyzer
            
            # 创建分析器实例
            analyzer = CSVBusinessAnalyzer()
            
            # 如果指定了日期，则使用指定日期目录
            if args.date:
                # 验证日期格式
                if not re.match(r'^\d{8}$', args.date):
                    logger.error(f"日期格式无效: {args.date}，正确格式应为YYYYMMDD")
                    return
                
                custom_dir = os.path.join(analyzer.selector_path, args.date)
                if not os.path.exists(custom_dir):
                    logger.error(f"指定日期目录不存在: {custom_dir}")
                    return
                
                csv_files = analyzer.get_csv_files(custom_dir)
            else:
                # 使用当天日期
                csv_files = analyzer.get_csv_files()
            
            if not csv_files:
                logger.warning("没有找到CSV文件，分析终止")
                return
            
            # 解析CSV文件
            stocks_data = analyzer.parse_csv_files(csv_files)
            if not stocks_data:
                logger.warning("没有解析到有效的股票数据，分析终止")
                return
            
            # 分析业务关系
            analysis_result = analyzer.analyze_business_relationships(stocks_data)
            if not analysis_result:
                logger.warning("生成业务分析结果失败，分析终止")
                return
            
            # 保存分析结果
            if args.output:
                # 使用指定的输出文件名
                today = datetime.now().strftime('%Y%m%d')
                result_file = os.path.join(analyzer.collect_path, f"{args.output}_{today}.txt")
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(analysis_result)
                logger.info(f"分析结果已保存至: {result_file}")
            else:
                # 使用默认输出名称
                result_file = analyzer.save_analysis_result(analysis_result)
                if result_file:
                    logger.info(f"CSV业务分析完成，结果保存在: {result_file}")
                else:
                    logger.warning("保存分析结果失败")
        else:
            logger.error(f"未知的汇总分析类型: {collect_type}")
    except Exception as e:
        logger.error(f"执行汇总分析过程中出错: {str(e)}")


def main():
    """主函数，处理命令行参数并执行相应操作"""
    try:
        print("开始执行命令...")
        
        # 解析命令行参数
        args = parse_args()
        
        # 处理不同的命令
        if args.command == 'selector':
            handle_select(args)
        elif args.command == 'analyzer':
            print(f"执行分析命令: {args.analyzer_type} {args.stock_code}")
            handle_analyzer(args)
        elif args.command == 'update':
            handle_update(args)
        elif args.command == 'collect':
            handle_collect(args)
        else:
            logger.error(f"未知的命令: {args.command}")
            print(f"未知的命令: {args.command}")
            
        print("命令执行完成")
    except Exception as e:
        logger.error(f"执行命令时出错: {str(e)}")
        print(f"执行命令时出错: {str(e)}")
        traceback.print_exc() 