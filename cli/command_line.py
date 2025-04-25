# -*- coding: utf-8 -*-
"""命令行处理模块"""

import argparse
import os
import sys
from datetime import datetime
import re
import json

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
    
    # 添加缠论T+0训练子命令
    chantrain_parser = subparsers.add_parser('chantrain', help='缠论T+0训练系统')
    chantrain_parser.add_argument('stock_code', help='股票代码，如：000001、600001等6位数字')
    chantrain_parser.add_argument('--initial-capital', type=float, default=100000.0, help='初始资金，默认10万')
    chantrain_parser.add_argument('--days', type=int, default=10, help='训练天数，默认10天')
    chantrain_parser.add_argument('--focus-level', choices=['1min', '5min', '30min'], default='1min', help='主要关注级别')
    chantrain_parser.add_argument('--interactive', action='store_true', help='是否进行交互式训练')
    chantrain_parser.add_argument('--backtest', action='store_true', help='启用回测模式，使用历史数据进行策略回测')
    chantrain_parser.add_argument('--start-date', help='回测开始日期，格式：YYYY-MM-DD，默认为最近交易日前30天')
    chantrain_parser.add_argument('--end-date', help='回测结束日期，格式：YYYY-MM-DD，默认为最近交易日')
    
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
        elif analyzer_type == 'news':
            from analyzer.news_analyzer import NewsAnalyzer
            # 处理网站参数
            sites = None
            if args.sites:
                sites = [site.strip() for site in args.sites.split(',')]
            
            # 获取股票名称，用于搜索
            from data.stock_data import StockData
            stock_data = StockData()
            stock_info = stock_data.get_stock_info(stock_code)
            stock_name = stock_info.get('name') if stock_info else None
            
            analyzer = NewsAnalyzer(
                stock_code=stock_code,
                days=args.days,
                enable_deep_crawl=args.deep_crawl if hasattr(args, 'deep_crawl') else True,
                deep_crawl_limit=args.deep_crawl_limit if hasattr(args, 'deep_crawl_limit') else 3
            )
            
            # 检查是否是单个URL提取
            if args.url:
                logger.info(f"提取单个新闻URL: {args.url}")
                
                # 提取新闻内容
                content_data = analyzer.extract_single_news(args.url)
                
                if 'data' in content_data:
                    # 输出提取结果
                    data = content_data['data']
                    output = f"新闻提取结果:\n"
                    output += f"标题: {data.get('title', '未找到标题')}\n"
                    output += f"发布日期: {data.get('publish_date', '未找到日期')}\n"
                    output += f"作者/来源: {data.get('author', '未找到作者')}\n"
                    output += f"关键词: {data.get('keywords', '未找到关键词')}\n\n"
                    output += f"内容:\n{data.get('content', '未找到内容')}\n"
                    
                    print(output)
                    
                    # 保存结果
                    if args.output:
                        output_file = args.output
                        if not output_file.endswith('.txt'):
                            output_file += '.txt'
                        
                        # 确保output目录存在
                        os.makedirs('output', exist_ok=True)
                        output_path = os.path.join('output', output_file)
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(output)
                            
                        logger.info(f"提取结果已保存到: {output_path}")
                        
                else:
                    logger.error(f"从URL提取内容失败: {args.url}")
                    if 'error' in content_data:
                        print(f"错误: {content_data['error']}")
                
                return
            
            # 执行新闻分析
            result = analyzer.analyze(
                stock_code=stock_code, 
                stock_name=stock_name,
                max_results=args.max_results,
                days=args.days, 
                sites=sites,
                deep_crawl=args.deep_crawl if hasattr(args, 'deep_crawl') else True,
                deep_crawl_limit=args.deep_crawl_limit if hasattr(args, 'deep_crawl_limit') else 3
            )
            
            # 格式化输出
            output = analyzer.format_output(result)
            print(output)
            
            # 保存结果
            if args.output:
                output_file = args.output
                if not output_file.endswith('.txt'):
                    output_file += '.txt'
                
                # 确保output目录存在
                os.makedirs('output', exist_ok=True)
                output_path = os.path.join('output', output_file)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output)
                    # 保存完整的新闻信息
                    f.write("\n\n完整新闻列表:\n")
                    for i, news in enumerate(result['news'], 1):
                        f.write(f"{i}. {news['title']}\n")
                        f.write(f"   链接: {news['link']}\n")
                        f.write(f"   摘要: {news['snippet']}\n\n")
                
                logger.info(f"分析结果已保存到: {output_path}")
                
            return
        else:
            logger.error(f"未知的分析器类型: {analyzer_type}")
            return
        
        # 执行分析 (news分析器已经在上面单独处理)
        if analyzer_type != 'news':
            result = analyzer.run_analysis(save_path=args.output)
            
            if result:
                # 输出图表路径
                logger.info(f"处理完成")
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


def handle_chantrain(args):
    """
    处理缠论T+0训练命令
    
    参数:
        args: 命令行参数
    """
    try:
        from work.chan_work import ChanWorkTrainer
        
        # 解析参数
        stock_code = args.stock_code
        initial_capital = args.initial_capital
        days = args.days
        focus_level = args.focus_level
        interactive = args.interactive
        backtest = args.backtest
        start_date = args.start_date if hasattr(args, 'start_date') else None
        end_date = args.end_date if hasattr(args, 'end_date') else None
        
        # 创建训练器实例
        trainer = ChanWorkTrainer(
            stock_code=stock_code,
            initial_capital=initial_capital,
            day_limit=days,
            focus_level=focus_level
        )
        
        logger.info(f"开始缠论T+0训练，股票代码: {stock_code}, 初始资金: {initial_capital}, 关注级别: {focus_level}")
        
        # 根据模式选择执行方法
        if backtest:
            # 回测模式
            logger.info(f"执行回测模式，开始日期: {start_date}, 结束日期: {end_date}")
            result = trainer.run_backtest(start_date=start_date, end_date=end_date)
        elif interactive:
            # 交互式训练模式
            logger.info("执行交互式训练模式")
            result = trainer.run_interactive_training()
        else:
            # 普通训练模式
            logger.info("执行普通训练模式")
            result = trainer.run_training()
        
        if result:
            # 输出训练结果摘要
            profit_rate = result.get('profit_rate', 0)
            total_trades = result.get('total_trades', 0)
            win_rate = result.get('win_rate', 0)
            
            logger.info(f"缠论T+0训练完成 - 收益率: {profit_rate:.2f}%, 交易次数: {total_trades}, 胜率: {win_rate:.2f}%")
            return True
        else:
            logger.error("缠论T+0训练失败")
            return False
    except Exception as e:
        logger.error(f"执行缠论T+0训练时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def handle_chan_backtest(args):
    """
    处理缠论回测命令
    
    参数:
        args: 命令行参数
    """
    try:
        from work.chan_work import ChanWorkTrainer
        
        trainer = ChanWorkTrainer()
        
        # 解析日期参数
        start_date = args.start_date if hasattr(args, 'start_date') else None
        end_date = args.end_date if hasattr(args, 'end_date') else None
        
        # 解析股票代码参数
        stock_code = args.stock_code if hasattr(args, 'stock_code') else None
        
        # 如果未指定股票代码，则使用配置文件中的股票列表
        if not stock_code:
            from config.config_manager import ConfigManager
            config = ConfigManager()
            stock_list = config.get_stock_list()
            if not stock_list:
                logger.error("未指定股票代码且配置文件中不存在股票列表")
                return False
        else:
            stock_list = [stock_code]
        
        # 执行回测
        results = {}
        for stock in stock_list:
            logger.info(f"开始对 {stock} 进行缠论回测...")
            result = trainer.run_backtest(
                stock_code=stock,
                start_date=start_date,
                end_date=end_date
            )
            results[stock] = result
            logger.info(f"{stock} 回测完成, 结果: {result}")
        
        # 汇总并保存结果
        output_file = args.output if hasattr(args, 'output') else "chan_backtest_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"缠论回测结果已保存到 {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"执行缠论回测时出错: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """主函数，命令行入口点"""
    # 解析命令行参数
    args = parse_args()
    
    # 根据命令分发处理
    try:
        if args.command == 'selector':
            # 选股
            handle_select(args)
        elif args.command == 'analyzer':
            # 分析
            handle_analyzer(args)
        elif args.command == 'update':
            # 更新数据
            handle_update(args)
        elif args.command == 'collect':
            # 汇总分析
            handle_collect(args)
        elif args.command == 'chantrain':
            # 缠论T+0训练
            handle_chantrain(args)
        elif args.command == 'chan_backtest':
            # 缠论回测
            handle_chan_backtest(args)
        else:
            print("请指定命令: selector, analyzer, update, collect, 或 chantrain")
            print("使用 -h 或 --help 查看帮助")
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        print("\n程序已终止")
    except Exception as e:
        logger.error(f"执行命令时出错: {str(e)}")
        print(f"执行命令时出错: {str(e)}")
        import traceback
        traceback.print_exc() 