# -*- coding: utf-8 -*-
"""
缠论T+0训练系统 - 训练器模块

该模块整合了缠论交易系统的各个组件，包括：
1. 数据获取与处理
2. 缠论分析与信号生成
3. 交易执行与记录
4. 结果可视化与报告

作为系统的中央控制器，提供训练环境和用户交互功能。
"""

import os
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import time
import logging
from logging.handlers import RotatingFileHandler
import argparse
import sys
import cmd

# 导入自定义模块
try:
    from .chan_data import ChanDataLoader
    from .chan_analysis import ChanAnalyzer
    from .chan_trading import ChanTrader
    from .chan_visualization import ChanVisualizer
except ImportError:
    from chan_data import ChanDataLoader
    from chan_analysis import ChanAnalyzer
    from chan_trading import ChanTrader
    from chan_visualization import ChanVisualizer

# 设置日志
def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """设置日志器"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # 控制台输出
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

class ChanTrainer:
    """
    缠论训练器类
    
    整合了数据加载、分析、交易和可视化功能，提供完整的交易训练环境。
    """
    
    def __init__(self, 
                 stock_code: str, 
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.0003,
                 slippage: float = 0.002,
                 data_dir: str = 'data',
                 output_dir: str = 'output',
                 log_dir: str = 'logs',
                 start_date: str = None,
                 end_date: str = None,
                 freq: str = '30min'):
        """
        初始化缠论训练器
        
        参数:
            stock_code (str): 股票代码，如 '600519' 或 'sh600519'
            initial_capital (float, 可选): 初始资金，默认100000元
            commission_rate (float, 可选): 佣金率，默认0.0003 (万3)
            slippage (float, 可选): 滑点，默认0.002
            data_dir (str, 可选): 数据目录，默认'data'
            output_dir (str, 可选): 输出目录，默认'output'
            log_dir (str, 可选): 日志目录，默认'logs'
            start_date (str, 可选): 开始日期，如 '2023-01-01'
            end_date (str, 可选): 结束日期，如 '2023-12-31'
            freq (str, 可选): 数据频率，默认'30min'，可选值: '1min', '5min', '15min', '30min', '60min', 'D'
        """
        # 基本参数
        self.stock_code = stock_code
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.freq = freq
        
        # 日期过滤
        self.start_date = start_date
        self.end_date = end_date
        
        # 目录设置
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.log_dir = log_dir
        
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(
            f'chan_trainer_{stock_code}',
            os.path.join(log_dir, f'chan_trainer_{stock_code}.log')
        )
        
        # 模块初始化
        self.data_loader = ChanDataLoader(
            data_dir=data_dir, 
            stock_code=stock_code,
            freq=freq,
            logger=self.logger
        )
        
        self.analyzer = ChanAnalyzer(
            stock_code=stock_code,
            logger=self.logger
        )
        
        self.trader = ChanTrader(
            stock_code=stock_code,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage=slippage,
            save_path=os.path.join(output_dir, 'trades'),
            logger=self.logger
        )
        
        self.visualizer = ChanVisualizer(
            save_path=os.path.join(output_dir, 'charts')
        )
        
        # 交易数据
        self.data = None  # 原始数据
        self.analyzed_data = None  # 分析后的数据
        self.current_idx = 0  # 当前处理的数据索引
        self.is_training = False  # 是否处于训练模式
        
        self.logger.info(f"ChanTrainer初始化完成，股票代码: {stock_code}, 初始资金: {initial_capital}")
    
    def load_data(self, reload: bool = False) -> pd.DataFrame:
        """
        加载股票数据
        
        参数:
            reload (bool, 可选): 是否强制重新下载数据，默认False
            
        返回:
            pd.DataFrame: 加载的数据
        """
        self.logger.info(f"开始加载{self.stock_code}的{self.freq}数据")
        
        # 通过数据加载器获取数据
        self.data = self.data_loader.load_data(reload=reload)
        
        # 日期过滤
        if self.start_date:
            self.data = self.data[self.data.index >= self.start_date]
            self.logger.info(f"数据过滤，开始日期: {self.start_date}")
        
        if self.end_date:
            self.data = self.data[self.data.index <= self.end_date]
            self.logger.info(f"数据过滤，结束日期: {self.end_date}")
        
        self.logger.info(f"数据加载完成，共{len(self.data)}条记录")
        return self.data
    
    def analyze_data(self) -> Dict[str, Any]:
        """
        分析数据，生成缠论分析结果
        
        返回:
            Dict[str, Any]: 分析结果，包含笔、线段、中枢等信息
        """
        if self.data is None:
            self.logger.warning("尚未加载数据，先加载数据")
            self.load_data()
        
        self.logger.info("开始缠论分析")
        self.analyzed_data = self.analyzer.analyze(self.data)
        self.logger.info("缠论分析完成")
        
        return self.analyzed_data
    
    def start_training(self, 
                       interactive: bool = True, 
                       auto_trade: bool = True, 
                       batch_size: int = 1) -> None:
        """
        开始训练过程
        
        参数:
            interactive (bool, 可选): 是否交互式训练，默认True
            auto_trade (bool, 可选): 是否自动交易，默认True
            batch_size (int, 可选): 每次处理的K线数量，默认1
        """
        if self.data is None:
            self.logger.warning("尚未加载数据，先加载数据")
            self.load_data()
        
        if self.analyzed_data is None:
            self.logger.warning("尚未分析数据，先分析数据")
            self.analyze_data()
        
        self.logger.info(f"开始训练，交互式: {interactive}, 自动交易: {auto_trade}, 批次大小: {batch_size}")
        self.is_training = True
        self.current_idx = 0
        
        # 重置交易状态
        self.trader.reset()
        
        data_length = len(self.data)
        
        if interactive:
            # 创建交互式命令处理器并启动
            cmd_processor = ChanCommandProcessor(self, auto_trade)
            cmd_processor.cmdloop()
        else:
            # 非交互模式，自动处理所有数据
            while self.current_idx < data_length:
                end_idx = min(self.current_idx + batch_size, data_length)
                self._process_batch(self.current_idx, end_idx, auto_trade)
                self.current_idx = end_idx
                
                # 非交互模式下添加进度信息
                progress = (self.current_idx / data_length) * 100
                self.logger.info(f"训练进度: {progress:.2f}% ({self.current_idx}/{data_length})")
        
        self.logger.info("训练结束")
        self.is_training = False
        
        # 展示训练结果
        self.show_training_results()
    
    def _process_batch(self, start_idx: int, end_idx: int, auto_trade: bool = True) -> None:
        """
        处理一批数据
        
        参数:
            start_idx (int): 开始索引
            end_idx (int): 结束索引
            auto_trade (bool, 可选): 是否自动交易，默认True
        """
        # 更新当前数据视图
        current_view = self.data.iloc[:end_idx]
        
        # 对当前视图数据进行缠论分析
        current_analysis = self.analyzer.analyze(current_view)
        
        # 根据分析结果检测信号
        signals = self.analyzer.detect_signals(current_analysis)
        
        # 处理最新的K线数据
        for i in range(start_idx, end_idx):
            current_time = self.data.index[i]
            current_bar = self.data.iloc[i]
            
            # 提取当前时间的信号
            current_signals = []
            for signal in signals:
                if signal['time'] == current_time:
                    current_signals.append(signal)
            
            # 执行交易（如果有信号且启用自动交易）
            if auto_trade and current_signals:
                for signal in current_signals:
                    if signal['action'] == 'buy':
                        price = current_bar['close'] * (1 + self.slippage)  # 考虑滑点
                        volume = 100 * int(min(self.trader.get_account_status()['cash'] / price / 100, 10))  # 最多买10手
                        
                        if volume >= 100:  # 至少1手
                            self.trader.buy(price, volume, signal['time'], reason=signal['reason'])
                            self.logger.info(f"自动交易: 买入 {self.stock_code}, 价格: {price:.2f}, 数量: {volume}, 时间: {signal['time']}, 原因: {signal['reason']}")
                    
                    elif signal['action'] == 'sell':
                        price = current_bar['close'] * (1 - self.slippage)  # 考虑滑点
                        position = self.trader.get_account_status()['position']
                        
                        if position > 0:
                            # 卖出所有持仓
                            self.trader.sell(price, position, signal['time'], reason=signal['reason'])
                            self.logger.info(f"自动交易: 卖出 {self.stock_code}, 价格: {price:.2f}, 数量: {position}, 时间: {signal['time']}, 原因: {signal['reason']}")
    
    def show_current_status(self) -> Dict[str, Any]:
        """
        显示当前状态
        
        返回:
            Dict[str, Any]: 当前状态信息
        """
        if not self.is_training or self.data is None:
            self.logger.warning("未处于训练状态或数据未加载")
            return {}
        
        current_time = self.data.index[self.current_idx - 1] if self.current_idx > 0 else None
        current_bar = self.data.iloc[self.current_idx - 1] if self.current_idx > 0 else None
        account_status = self.trader.get_account_status()
        
        status = {
            "current_time": current_time,
            "current_price": current_bar['close'] if current_bar is not None else None,
            "account": account_status
        }
        
        # 打印当前状态
        self.logger.info(f"当前状态: 时间={current_time}, 价格={current_bar['close'] if current_bar is not None else 'N/A'}")
        self.logger.info(f"账户状态: 现金={account_status['cash']:.2f}, 持仓={account_status['position']}, 净值={account_status['total_value']:.2f}")
        
        return status
    
    def manual_trade(self, action: str) -> bool:
        """
        进行手动交易
        
        参数:
            action (str): 交易动作，"buy" 或 "sell"
            
        返回:
            bool: 交易是否成功
        """
        if not self.is_training or self.current_idx <= 0:
            self.logger.warning("未处于训练状态或数据不足")
            return False
        
        current_time = self.data.index[self.current_idx - 1]
        current_bar = self.data.iloc[self.current_idx - 1]
        
        if action.lower() == "buy":
            result = self.trader.manual_buy(current_bar, current_time)
            if result:
                self.logger.info(f"手动交易: 买入 {self.stock_code}, 价格: {result['price']:.2f}, 数量: {result['volume']}, 时间: {current_time}")
                return True
                
        elif action.lower() == "sell":
            result = self.trader.manual_sell(current_bar, current_time)
            if result:
                self.logger.info(f"手动交易: 卖出 {self.stock_code}, 价格: {result['price']:.2f}, 数量: {result['volume']}, 时间: {current_time}")
                return True
        
        return False
    
    def next_batch(self, steps: int = 1, auto_trade: bool = True) -> Dict[str, Any]:
        """
        前进指定步数
        
        参数:
            steps (int, 可选): 前进的步数，默认1
            auto_trade (bool, 可选): 是否自动交易，默认True
            
        返回:
            Dict[str, Any]: 前进后的状态
        """
        if not self.is_training:
            self.logger.warning("未处于训练状态")
            return {}
        
        data_length = len(self.data)
        
        # 计算下一批次的结束位置
        end_idx = min(self.current_idx + steps, data_length)
        
        if self.current_idx >= data_length:
            self.logger.info("已到达数据末尾")
            return self.show_current_status()
        
        # 处理这一批次
        self._process_batch(self.current_idx, end_idx, auto_trade)
        self.current_idx = end_idx
        
        # 返回处理后的状态
        return self.show_current_status()
    
    def show_training_results(self) -> Dict[str, Any]:
        """
        显示训练结果
        
        返回:
            Dict[str, Any]: 训练结果信息
        """
        # 获取交易摘要
        trade_summary = self.trader.get_trade_summary()
        trade_records = self.trader.trades
        
        self.logger.info("========== 交易训练结果 ==========")
        self.logger.info(f"股票代码: {self.stock_code}")
        self.logger.info(f"初始资金: {trade_summary['initial_capital']:.2f}")
        self.logger.info(f"最终资金: {trade_summary['current_capital']:.2f}")
        self.logger.info(f"净利润: {trade_summary['net_profit']:.2f}")
        self.logger.info(f"收益率: {trade_summary['profit_rate']:.2f}%")
        self.logger.info(f"交易次数: {trade_summary['total_trades']}")
        self.logger.info(f"胜率: {trade_summary['win_rate']:.2f}%")
        self.logger.info(f"平均持仓时间: {trade_summary['avg_holding_time']:.2f}小时")
        self.logger.info("================================")
        
        # 生成交易报告
        if trade_records:
            self.logger.info("生成交易报告...")
            report_path = self.visualizer.generate_trade_report(
                trade_summary,
                trade_records,
                output_path=os.path.join(self.output_dir, 'reports')
            )
            self.logger.info(f"交易报告已生成: {report_path}")
            
            # 生成交易图表
            self.logger.info("生成交易图表...")
            chart_path = self.visualizer.plot_kline_with_signals(
                self.data,
                trade_records,
                title=f"{self.stock_code} 交易图表",
                stock_code=self.stock_code,
                save_fig=True,
                show_fig=False
            )
            self.logger.info(f"交易图表已生成: {chart_path}")
            
            # 生成绩效图表
            performance_path = self.visualizer.plot_performance(
                trade_summary,
                trade_records,
                title=f"{self.stock_code} 交易绩效",
                stock_code=self.stock_code,
                save_fig=True,
                show_fig=False
            )
            self.logger.info(f"绩效图表已生成: {performance_path}")
            
            # 生成缠论分析图表
            if self.analyzed_data and 'segments' in self.analyzed_data:
                segments = self.analyzed_data.get('segments', [])
                central_zones = self.analyzed_data.get('central_zones', [])
                
                chan_chart_path = self.visualizer.plot_chan_analysis(
                    self.data,
                    segments,
                    central_zones,
                    title=f"{self.stock_code} 缠论分析",
                    stock_code=self.stock_code,
                    save_fig=True,
                    show_fig=False
                )
                self.logger.info(f"缠论分析图已生成: {chan_chart_path}")
        
        return trade_summary

# 命令行交互处理器
class ChanCommandProcessor(cmd.Cmd):
    """
    缠论训练系统命令行处理器
    """
    
    intro = '''
    ====================================
    缠论T+0训练系统 - 交互式命令行界面
    ====================================
    
    可用命令:
    help - 显示帮助信息
    next [steps=1] - 前进指定步数
    status - 显示当前状态
    buy - 手动买入
    sell - 手动卖出
    auto - 切换自动交易模式
    chart - 显示当前图表
    result - 显示训练结果
    quit - 退出训练
    
    键入 help <命令> 获取详细帮助。
    '''
    
    prompt = 'ChanTrainer> '
    
    def __init__(self, trainer: ChanTrainer, auto_trade: bool = True):
        """
        初始化命令行处理器
        
        参数:
            trainer (ChanTrainer): 缠论训练器实例
            auto_trade (bool): 是否自动交易
        """
        super().__init__()
        self.trainer = trainer
        self.auto_trade = auto_trade
        self.logger = trainer.logger
    
    def do_next(self, arg: str) -> None:
        """
        前进指定步数
        
        用法: next [steps=1]
        """
        try:
            steps = int(arg) if arg else 1
            if steps <= 0:
                print("步数必须为正整数")
                return
                
            status = self.trainer.next_batch(steps, self.auto_trade)
            
            if status:
                print(f"\n当前时间: {status['current_time']}")
                print(f"当前价格: {status['current_price']:.2f}")
                print(f"现金: {status['account']['cash']:.2f}")
                print(f"持仓: {status['account']['position']}")
                print(f"总资产: {status['account']['total_value']:.2f}")
                print(f"收益率: {status['account']['profit_rate']:.2f}%\n")
        except ValueError:
            print("无效的步数，请输入整数")
    
    def help_next(self) -> None:
        """next命令的帮助信息"""
        print('前进指定步数，默认为1步')
        print('用法: next [steps=1]')
    
    def do_status(self, arg: str) -> None:
        """
        显示当前状态
        
        用法: status
        """
        status = self.trainer.show_current_status()
        
        if status:
            print("\n==== 当前状态 ====")
            print(f"时间: {status['current_time']}")
            print(f"价格: {status['current_price']:.2f}")
            print(f"现金: {status['account']['cash']:.2f}")
            print(f"持仓: {status['account']['position']}")
            print(f"总资产: {status['account']['total_value']:.2f}")
            print(f"收益率: {status['account']['profit_rate']:.2f}%")
            print("=================\n")
    
    def help_status(self) -> None:
        """status命令的帮助信息"""
        print('显示当前状态，包括时间、价格、账户信息等')
        print('用法: status')
    
    def do_buy(self, arg: str) -> None:
        """
        手动买入操作
        
        用法: buy
        """
        if self.trainer.manual_trade("buy"):
            print("买入成功")
            self.do_status("")
        else:
            print("买入失败，请检查是否有足够资金或当前是否处于训练状态")
    
    def help_buy(self) -> None:
        """buy命令的帮助信息"""
        print('执行手动买入操作')
        print('用法: buy')
    
    def do_sell(self, arg: str) -> None:
        """
        手动卖出操作
        
        用法: sell
        """
        if self.trainer.manual_trade("sell"):
            print("卖出成功")
            self.do_status("")
        else:
            print("卖出失败，请检查是否有持仓或当前是否处于训练状态")
    
    def help_sell(self) -> None:
        """sell命令的帮助信息"""
        print('执行手动卖出操作')
        print('用法: sell')
    
    def do_auto(self, arg: str) -> None:
        """
        切换自动交易模式
        
        用法: auto [on|off]
        """
        if arg.lower() in ('on', 'true', '1'):
            self.auto_trade = True
            print("自动交易模式已开启")
        elif arg.lower() in ('off', 'false', '0'):
            self.auto_trade = False
            print("自动交易模式已关闭")
        else:
            # 切换当前状态
            self.auto_trade = not self.auto_trade
            print(f"自动交易模式已{'开启' if self.auto_trade else '关闭'}")
    
    def help_auto(self) -> None:
        """auto命令的帮助信息"""
        print('切换自动交易模式')
        print('用法: auto [on|off]')
        print('不带参数则切换当前状态')
    
    def do_chart(self, arg: str) -> None:
        """
        显示当前图表
        
        用法: chart
        """
        if not self.trainer.is_training or self.trainer.data is None:
            print("未处于训练状态或数据未加载")
            return
        
        # 使用当前数据和交易记录生成图表
        print("正在生成图表...")
        chart_path = self.trainer.visualizer.plot_kline_with_signals(
            self.trainer.data.iloc[:self.trainer.current_idx],
            self.trainer.trader.trades,
            title=f"{self.trainer.stock_code} 当前交易图表",
            stock_code=self.trainer.stock_code,
            save_fig=True,
            show_fig=True  # 显示图表
        )
        print(f"图表已生成: {chart_path}")
    
    def help_chart(self) -> None:
        """chart命令的帮助信息"""
        print('显示当前图表，包括K线和交易信号')
        print('用法: chart')
    
    def do_result(self, arg: str) -> None:
        """
        显示训练结果
        
        用法: result
        """
        result = self.trainer.show_training_results()
        
        if result:
            print("\n==== 训练结果 ====")
            print(f"股票代码: {self.trainer.stock_code}")
            print(f"初始资金: {result['initial_capital']:.2f}")
            print(f"最终资金: {result['current_capital']:.2f}")
            print(f"净利润: {result['net_profit']:.2f}")
            print(f"收益率: {result['profit_rate']:.2f}%")
            print(f"交易次数: {result['total_trades']}")
            print(f"胜率: {result['win_rate']:.2f}%")
            print(f"平均持仓时间: {result['avg_holding_time']:.2f}小时")
            print("=================\n")
    
    def help_result(self) -> None:
        """result命令的帮助信息"""
        print('显示训练结果，包括收益率、交易次数、胜率等')
        print('用法: result')
    
    def do_quit(self, arg: str) -> bool:
        """
        退出训练
        
        用法: quit
        """
        print("退出训练")
        return True
    
    def help_quit(self) -> None:
        """quit命令的帮助信息"""
        print('退出训练系统')
        print('用法: quit')
    
    # 别名
    do_q = do_quit
    do_exit = do_quit
    do_n = do_next
    do_s = do_status
    do_b = do_buy
    do_sl = do_sell
    do_a = do_auto
    do_c = do_chart
    do_r = do_result

# 命令行入口
def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='缠论T+0训练系统')
    
    parser.add_argument('stock_code', help='股票代码，如 600519 或 sh600519')
    parser.add_argument('--capital', '-c', type=float, default=100000.0, help='初始资金，默认100000元')
    parser.add_argument('--commission', type=float, default=0.0003, help='佣金率，默认0.0003 (万3)')
    parser.add_argument('--slippage', type=float, default=0.002, help='滑点，默认0.002')
    parser.add_argument('--data-dir', default='data', help='数据目录，默认data')
    parser.add_argument('--output-dir', default='output', help='输出目录，默认output')
    parser.add_argument('--log-dir', default='logs', help='日志目录，默认logs')
    parser.add_argument('--start-date', help='开始日期，格式：YYYY-MM-DD')
    parser.add_argument('--end-date', help='结束日期，格式：YYYY-MM-DD')
    parser.add_argument('--freq', default='30min', choices=['1min', '5min', '15min', '30min', '60min', 'D'], 
                        help='数据频率，默认30min')
    parser.add_argument('--non-interactive', '-n', action='store_true', help='非交互式模式')
    parser.add_argument('--no-auto-trade', '-a', action='store_true', help='关闭自动交易')
    parser.add_argument('--reload-data', '-r', action='store_true', help='强制重新下载数据')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = ChanTrainer(
        stock_code=args.stock_code,
        initial_capital=args.capital,
        commission_rate=args.commission,
        slippage=args.slippage,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        freq=args.freq
    )
    
    # 加载数据
    trainer.load_data(reload=args.reload_data)
    
    # 分析数据
    trainer.analyze_data()
    
    # 开始训练
    trainer.start_training(
        interactive=not args.non_interactive,
        auto_trade=not args.no_auto_trade
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 