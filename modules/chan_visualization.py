# -*- coding: utf-8 -*-
"""
缠论T+0训练系统 - 可视化模块

该模块提供缠论交易系统的可视化功能，包括：
1. K线图与交易信号可视化
2. 回测绩效报告生成
3. 交易记录可视化
4. 缠论分析图表生成

用于直观展示交易过程、结果和缠论分析。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
import mplfinance as mpf
from typing import Dict, List, Optional, Union, Any, Tuple
import datetime
import locale
import warnings
from pathlib import Path
import jinja2
import json
import time

# 设置中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    locale.setlocale(locale.LC_TIME, 'zh_CN.UTF-8')
except:
    warnings.warn("无法设置中文环境，图表中文可能无法正确显示")

class ChanVisualizer:
    """
    缠论可视化器类
    
    提供缠论交易系统的各种可视化功能，包括K线图绘制、交易信号标记、
    绩效报告生成、交易记录可视化等。
    """
    
    def __init__(self, save_path: str = 'output/charts'):
        """
        初始化缠论可视化器
        
        参数:
            save_path (str, 可选): 图表保存路径，默认为'output/charts'
        """
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        # 尝试设置中文字体
        try:
            self.chinese_font = FontProperties(family=['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'])
        except:
            self.chinese_font = None
            warnings.warn("无法设置中文字体，图表中文可能无法正确显示")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_kline_with_signals(self, 
                               data: pd.DataFrame, 
                               trade_records: List[Dict[str, Any]], 
                               title: str = "K线图与交易信号",
                               stock_code: str = "",
                               save_fig: bool = True,
                               show_fig: bool = True) -> Optional[str]:
        """
        绘制K线图并标记交易信号
        
        参数:
            data (pd.DataFrame): K线数据，必须包含OHLCV
            trade_records (List[Dict[str, Any]]): 交易记录列表
            title (str, 可选): 图表标题
            stock_code (str, 可选): 股票代码
            save_fig (bool, 可选): 是否保存图表
            show_fig (bool, 可选): 是否显示图表
            
        返回:
            Optional[str]: 如果save_fig为True，返回保存的文件路径
        """
        # 确保索引是日期时间类型
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据索引必须是DatetimeIndex类型")
        
        # 准备数据
        df = data.copy()
        
        # 转换为mplfinance可用的格式
        df.index.name = 'Date'
        df.columns = [c.lower() for c in df.columns]
        
        # 创建买卖信号
        buys = []
        sells = []
        
        for trade in trade_records:
            if trade['action'] == 'buy':
                buys.append((pd.to_datetime(trade['time']), trade['price']))
            elif trade['action'] == 'sell':
                sells.append((pd.to_datetime(trade['time']), trade['price']))
        
        # 创建mpf_style
        mpf_style = mpf.make_mpf_style(
            base_mpf_style='charles', 
            marketcolors=mpf.make_marketcolors(
                up='red', down='green',
                edge='inherit',
                wick='inherit',
                volume='inherit'
            ),
            gridaxis='both',
            gridstyle=':',
            y_on_right=False
        )
        
        # 创建图表
        fig, axes = mpf.plot(
            df,
            type='candle',
            style=mpf_style,
            title=f"{title} - {stock_code}" if stock_code else title,
            ylabel='价格',
            figsize=(15, 10),
            volume=True,
            returnfig=True,
            panel_ratios=(4, 1),  # 主图和成交量图的比例
            warn_too_much_data=10000
        )
        
        fig = axes[0]
        ax1 = axes[1]  # 主K线图
        ax2 = axes[2]  # 成交量图
        
        # 设置标题字体
        ax1.set_title(
            f"{title} - {stock_code}" if stock_code else title,
            fontproperties=self.chinese_font,
            fontsize=16
        )
        
        # 添加买入信号
        for buy_time, buy_price in buys:
            if buy_time in df.index:
                idx = df.index.get_loc(buy_time)
                ax1.scatter(
                    idx, buy_price * 0.99,  # 稍微偏移以便更好地显示
                    s=100, 
                    color='red', 
                    marker='^', 
                    alpha=0.7, 
                    zorder=5,
                    label='买入' if '买入' not in ax1.get_legend_handles_labels()[1] else ''
                )
        
        # 添加卖出信号
        for sell_time, sell_price in sells:
            if sell_time in df.index:
                idx = df.index.get_loc(sell_time)
                ax1.scatter(
                    idx, sell_price * 1.01,  # 稍微偏移以便更好地显示
                    s=100, 
                    color='green', 
                    marker='v', 
                    alpha=0.7, 
                    zorder=5,
                    label='卖出' if '卖出' not in ax1.get_legend_handles_labels()[1] else ''
                )
        
        # 添加图例
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend(handles, labels, loc='best', prop=self.chinese_font)
        
        # 保存或显示图表
        save_path = None
        if save_fig:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{stock_code}_kline_signals_{timestamp}.png" if stock_code else f"kline_signals_{timestamp}.png"
            save_path = os.path.join(self.save_path, filename)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
            
        return save_path
    
    def plot_performance(self, 
                        trade_summary: Dict[str, Any], 
                        trade_records: List[Dict[str, Any]],
                        title: str = "交易绩效报告",
                        stock_code: str = "",
                        save_fig: bool = True,
                        show_fig: bool = True) -> Optional[str]:
        """
        绘制交易绩效报告
        
        参数:
            trade_summary (Dict[str, Any]): 交易摘要信息
            trade_records (List[Dict[str, Any]]): 交易记录列表
            title (str, 可选): 图表标题
            stock_code (str, 可选): 股票代码
            save_fig (bool, 可选): 是否保存图表
            show_fig (bool, 可选): 是否显示图表
            
        返回:
            Optional[str]: 如果save_fig为True，返回保存的文件路径
        """
        if not trade_records:
            warnings.warn("没有交易记录，无法生成绩效报告")
            return None
        
        # 创建图表
        fig = plt.figure(figsize=(18, 15))
        fig.suptitle(
            f"{title} - {stock_code}" if stock_code else title,
            fontproperties=self.chinese_font,
            fontsize=20
        )
        
        # 创建网格布局
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # 1. 资金曲线
        ax_capital = fig.add_subplot(gs[0, :])
        ax_capital.set_title('资金曲线', fontproperties=self.chinese_font)
        
        # 处理资金曲线数据
        capital_data = {}
        initial_capital = trade_summary['initial_capital']
        capital_value = initial_capital
        
        for trade in trade_records:
            time = pd.to_datetime(trade['time'])
            if trade['action'] == 'buy':
                capital_value -= trade['price'] * trade['volume'] * (1 + trade['commission_rate'])
            elif trade['action'] == 'sell':
                capital_value += trade['price'] * trade['volume'] * (1 - trade['commission_rate'])
            
            capital_data[time] = capital_value
        
        # 排序并转换为DataFrame
        if capital_data:
            capital_df = pd.Series(capital_data).sort_index()
            capital_df = pd.DataFrame({'capital': capital_df})
            
            # 添加初始资金点
            if len(capital_df) > 0:
                first_time = capital_df.index[0] - pd.Timedelta(days=1)
                capital_df.loc[first_time] = initial_capital
                capital_df = capital_df.sort_index()
            
            # 绘制资金曲线
            ax_capital.plot(
                capital_df.index, 
                capital_df['capital'], 
                color='blue', 
                linewidth=2,
                marker='.',
                markersize=8
            )
            
            # 设置x轴格式
            ax_capital.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_capital.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # 添加网格
            ax_capital.grid(True, linestyle='--', alpha=0.7)
            
            # 设置标签
            ax_capital.set_ylabel('资金', fontproperties=self.chinese_font)
            ax_capital.set_xlabel('日期', fontproperties=self.chinese_font)
        
        # 2. 盈亏饼图
        ax_pie = fig.add_subplot(gs[1, 0])
        ax_pie.set_title('交易盈亏比例', fontproperties=self.chinese_font)
        
        # 计算盈利和亏损交易数
        win_trades = len([t for t in trade_summary.get('trade_details', []) if t['profit'] > 0])
        loss_trades = len([t for t in trade_summary.get('trade_details', []) if t['profit'] <= 0])
        
        if win_trades + loss_trades > 0:
            # 绘制饼图
            ax_pie.pie(
                [win_trades, loss_trades],
                labels=['盈利', '亏损'],
                colors=['#2ca02c', '#d62728'],
                autopct='%1.1f%%',
                startangle=90,
                explode=(0.05, 0),
                textprops={'fontproperties': self.chinese_font}
            )
        
        # 3. 盈亏分布直方图
        ax_hist = fig.add_subplot(gs[1, 1])
        ax_hist.set_title('交易盈亏分布', fontproperties=self.chinese_font)
        
        # 从交易明细中提取盈亏数据
        profits = [t['profit'] for t in trade_summary.get('trade_details', [])]
        
        if profits:
            # 绘制直方图
            bins = min(10, len(profits))
            ax_hist.hist(
                profits,
                bins=bins,
                color='#3274A1',
                edgecolor='white',
                alpha=0.7
            )
            
            # 设置标签
            ax_hist.set_xlabel('盈亏金额', fontproperties=self.chinese_font)
            ax_hist.set_ylabel('交易次数', fontproperties=self.chinese_font)
            
            # 添加网格
            ax_hist.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 4. 持仓时间分布
        ax_hold = fig.add_subplot(gs[1, 2])
        ax_hold.set_title('持仓时间分布', fontproperties=self.chinese_font)
        
        # 从交易明细中提取持仓时间
        hold_times = [t['holding_time'] for t in trade_summary.get('trade_details', [])]
        
        if hold_times:
            # 绘制直方图
            bins = min(10, len(hold_times))
            ax_hold.hist(
                hold_times,
                bins=bins,
                color='#E377C2',
                edgecolor='white',
                alpha=0.7
            )
            
            # 设置标签
            ax_hold.set_xlabel('持仓时间（小时）', fontproperties=self.chinese_font)
            ax_hold.set_ylabel('交易次数', fontproperties=self.chinese_font)
            
            # 添加网格
            ax_hold.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 5. 累计盈亏曲线
        ax_cumul = fig.add_subplot(gs[2, :])
        ax_cumul.set_title('累计盈亏曲线', fontproperties=self.chinese_font)
        
        # 生成累计盈亏数据
        cumul_profit = {}
        current_profit = 0
        
        # 按时间排序交易记录
        sorted_trade_details = sorted(
            trade_summary.get('trade_details', []),
            key=lambda x: pd.to_datetime(x.get('close_time') or x.get('open_time'))
        )
        
        for trade in sorted_trade_details:
            time = pd.to_datetime(trade.get('close_time') or trade.get('open_time'))
            current_profit += trade['profit']
            cumul_profit[time] = current_profit
        
        # 排序并转换为DataFrame
        if cumul_profit:
            cumul_df = pd.Series(cumul_profit).sort_index()
            cumul_df = pd.DataFrame({'cumul_profit': cumul_df})
            
            # 添加起始点
            if len(cumul_df) > 0:
                first_time = cumul_df.index[0] - pd.Timedelta(days=1)
                cumul_df.loc[first_time] = 0
                cumul_df = cumul_df.sort_index()
            
            # 绘制累计盈亏曲线
            ax_cumul.plot(
                cumul_df.index,
                cumul_df['cumul_profit'],
                color='green',
                linewidth=2,
                marker='.',
                markersize=8
            )
            
            # 绘制零线
            ax_cumul.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # 设置x轴格式
            ax_cumul.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_cumul.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # 添加网格
            ax_cumul.grid(True, linestyle='--', alpha=0.7)
            
            # 设置标签
            ax_cumul.set_ylabel('累计盈亏', fontproperties=self.chinese_font)
            ax_cumul.set_xlabel('日期', fontproperties=self.chinese_font)
        
        # 添加绩效摘要文本
        summary_text = (
            f"初始资金: {trade_summary['initial_capital']:.2f}\n"
            f"最终资金: {trade_summary['current_capital']:.2f}\n"
            f"净利润: {trade_summary['net_profit']:.2f}\n"
            f"收益率: {trade_summary['profit_rate']:.2f}%\n"
            f"交易次数: {trade_summary['total_trades']}\n"
            f"胜率: {trade_summary['win_rate']:.2f}%\n"
            f"平均持仓时间: {trade_summary['avg_holding_time']:.2f}小时"
        )
        
        fig.text(
            0.13, 0.02,
            summary_text,
            fontproperties=self.chinese_font,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # 保存或显示图表
        save_path = None
        if save_fig:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{stock_code}_performance_{timestamp}.png" if stock_code else f"performance_{timestamp}.png"
            save_path = os.path.join(self.save_path, filename)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
            
        return save_path
    
    def generate_trade_report(self, 
                             trade_summary: Dict[str, Any], 
                             trade_records: List[Dict[str, Any]],
                             template_file: Optional[str] = None,
                             output_path: str = 'output/reports') -> Optional[str]:
        """
        生成交易报告HTML文件
        
        参数:
            trade_summary (Dict[str, Any]): 交易摘要信息
            trade_records (List[Dict[str, Any]]): 交易记录列表
            template_file (Optional[str], 可选): HTML模板文件路径，如果为None则使用默认模板
            output_path (str, 可选): 输出目录路径
            
        返回:
            Optional[str]: 生成的报告文件路径
        """
        if not trade_records:
            warnings.warn("没有交易记录，无法生成交易报告")
            return None
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 准备数据
        stock_code = trade_summary.get('stock_code', '')
        report_data = {
            'title': f'{stock_code} 交易报告' if stock_code else '交易报告',
            'generation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': trade_summary,
            'trades': sorted(
                trade_summary.get('trade_details', []),
                key=lambda x: pd.to_datetime(x.get('close_time') or x.get('open_time')),
                reverse=True
            )
        }
        
        # 默认HTML模板
        default_template = '''
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
            <style>
                body {
                    font-family: Arial, "Microsoft YaHei", sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                    background-color: #f8f9fa;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: #fff;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 5px;
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid #eee;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .summary-box {
                    background-color: #f8f9fa;
                    border-left: 4px solid #2c3e50;
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 3px;
                }
                .summary-grid {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 15px;
                }
                .summary-item {
                    padding: 15px;
                    background-color: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }
                .summary-item h3 {
                    margin-top: 0;
                    font-size: 16px;
                    color: #666;
                }
                .summary-item p {
                    margin: 0;
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }
                .profit {
                    color: #e74c3c;
                }
                .loss {
                    color: #2ecc71;
                }
                .footer {
                    text-align: center;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    color: #7f8c8d;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{{ title }}</h1>
                    <p>生成时间: {{ generation_time }}</p>
                </div>
                
                <section>
                    <h2>交易摘要</h2>
                    <div class="summary-grid">
                        <div class="summary-item">
                            <h3>初始资金</h3>
                            <p>{{ summary.initial_capital | round(2) }}</p>
                        </div>
                        <div class="summary-item">
                            <h3>最终资金</h3>
                            <p>{{ summary.current_capital | round(2) }}</p>
                        </div>
                        <div class="summary-item">
                            <h3>净利润</h3>
                            <p class="{{ 'profit' if summary.net_profit > 0 else 'loss' }}">{{ summary.net_profit | round(2) }}</p>
                        </div>
                        <div class="summary-item">
                            <h3>收益率</h3>
                            <p class="{{ 'profit' if summary.profit_rate > 0 else 'loss' }}">{{ summary.profit_rate | round(2) }}%</p>
                        </div>
                        <div class="summary-item">
                            <h3>交易次数</h3>
                            <p>{{ summary.total_trades }}</p>
                        </div>
                        <div class="summary-item">
                            <h3>胜率</h3>
                            <p>{{ summary.win_rate | round(2) }}%</p>
                        </div>
                        <div class="summary-item">
                            <h3>平均持仓时间</h3>
                            <p>{{ summary.avg_holding_time | round(2) }} 小时</p>
                        </div>
                    </div>
                </section>
                
                <section>
                    <h2>交易记录</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>序号</th>
                                <th>开仓时间</th>
                                <th>开仓价格</th>
                                <th>开仓量</th>
                                <th>平仓时间</th>
                                <th>平仓价格</th>
                                <th>持仓时间</th>
                                <th>盈亏</th>
                                <th>收益率</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for trade in trades %}
                            <tr>
                                <td>{{ loop.index }}</td>
                                <td>{{ trade.open_time }}</td>
                                <td>{{ trade.open_price | round(2) }}</td>
                                <td>{{ trade.volume }}</td>
                                <td>{{ trade.close_time or '未平仓' }}</td>
                                <td>{{ trade.close_price | round(2) if trade.close_price else '-' }}</td>
                                <td>{{ trade.holding_time | round(2) if trade.holding_time else '-' }}</td>
                                <td class="{{ 'profit' if trade.profit > 0 else 'loss' }}">{{ trade.profit | round(2) }}</td>
                                <td class="{{ 'profit' if trade.profit_rate > 0 else 'loss' }}">{{ trade.profit_rate | round(2) }}%</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </section>
                
                <div class="footer">
                    <p>缠论T+0训练系统 - 交易报告</p>
                </div>
            </div>
        </body>
        </html>
        '''
        
        # 设置Jinja2环境
        template_loader = jinja2.FileSystemLoader(searchpath='.')
        template_env = jinja2.Environment(loader=template_loader)
        
        # 使用提供的模板或默认模板
        if template_file and os.path.exists(template_file):
            # 从文件加载模板
            template = template_env.get_template(template_file)
        else:
            # 使用默认模板
            template = jinja2.Template(default_template)
        
        # 渲染HTML
        html_output = template.render(**report_data)
        
        # 生成输出文件名
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{stock_code}_report_{timestamp}.html" if stock_code else f"trade_report_{timestamp}.html"
        output_file = os.path.join(output_path, file_name)
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_output)
        
        return output_file
    
    def plot_chan_analysis(self, 
                          data: pd.DataFrame, 
                          segments: List[Dict[str, Any]], 
                          central_zones: List[Dict[str, Any]],
                          title: str = "缠论分析图",
                          stock_code: str = "",
                          save_fig: bool = True,
                          show_fig: bool = True) -> Optional[str]:
        """
        绘制缠论分析图
        
        参数:
            data (pd.DataFrame): K线数据，必须包含OHLCV
            segments (List[Dict[str, Any]]): 线段列表
            central_zones (List[Dict[str, Any]]): 中枢列表
            title (str, 可选): 图表标题
            stock_code (str, 可选): 股票代码
            save_fig (bool, 可选): 是否保存图表
            show_fig (bool, 可选): 是否显示图表
            
        返回:
            Optional[str]: 如果save_fig为True，返回保存的文件路径
        """
        # 确保索引是日期时间类型
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据索引必须是DatetimeIndex类型")
        
        # 准备数据
        df = data.copy()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 绘制K线收盘价
        ax.plot(df.index, df['close'], color='gray', linewidth=1, alpha=0.6, label='收盘价')
        
        # 绘制线段
        for seg in segments:
            if 'start_idx' in seg and 'end_idx' in seg:
                start_idx, end_idx = seg['start_idx'], seg['end_idx']
                start_val, end_val = seg['start_val'], seg['end_val']
                
                if 0 <= start_idx < len(df) and 0 <= end_idx < len(df):
                    start_time = df.index[start_idx]
                    end_time = df.index[end_idx]
                    
                    # 根据上升/下降设置颜色
                    color = 'red' if end_val > start_val else 'green'
                    
                    ax.plot(
                        [start_time, end_time], 
                        [start_val, end_val], 
                        color=color, 
                        linewidth=2, 
                        alpha=0.8
                    )
                    
                    # 标记关键点
                    ax.scatter(start_time, start_val, color=color, s=50, alpha=0.8, zorder=5)
                    ax.scatter(end_time, end_val, color=color, s=50, alpha=0.8, zorder=5)
        
        # 绘制中枢区域
        for zone in central_zones:
            if 'start_idx' in zone and 'end_idx' in zone:
                start_idx, end_idx = zone['start_idx'], zone['end_idx']
                high, low = zone['high'], zone['low']
                
                if 0 <= start_idx < len(df) and 0 <= end_idx < len(df):
                    start_time = df.index[start_idx]
                    end_time = df.index[end_idx]
                    
                    # 绘制中枢矩形
                    ax.fill_between(
                        [start_time, end_time], 
                        low, 
                        high, 
                        color='blue', 
                        alpha=0.2
                    )
                    
                    # 绘制中枢边界
                    ax.plot([start_time, end_time], [high, high], 'b--', alpha=0.5)
                    ax.plot([start_time, end_time], [low, low], 'b--', alpha=0.5)
                    
                    # 可选：添加中枢标注
                    ax.text(
                        start_time,
                        high,
                        f'中枢{zone.get("level", "")}',
                        color='blue',
                        fontsize=8,
                        fontproperties=self.chinese_font
                    )
        
        # 设置图表标题和标签
        ax.set_title(
            f"{title} - {stock_code}" if stock_code else title,
            fontproperties=self.chinese_font,
            fontsize=16
        )
        ax.set_ylabel('价格', fontproperties=self.chinese_font)
        ax.set_xlabel('日期', fontproperties=self.chinese_font)
        
        # 设置x轴日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 优化布局
        plt.tight_layout()
        
        # 旋转x轴日期标签
        plt.xticks(rotation=45)
        
        # 保存或显示图表
        save_path = None
        if save_fig:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{stock_code}_chan_analysis_{timestamp}.png" if stock_code else f"chan_analysis_{timestamp}.png"
            save_path = os.path.join(self.save_path, filename)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
            
        return save_path 