# -*- coding: utf-8 -*-
"""黄金分割分析器模块，用于识别股票的斐波那契回调水平"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

# 导入基础分析器和工具
from analyzer.base_analyzer import BaseAnalyzer
from config import ANALYZER_CONFIG, PATH_CONFIG
from utils.logger import get_logger
from utils.indicators import calculate_fibonacci_levels, plot_stock_chart, calculate_technical_indicators

# 创建日志记录器
logger = get_logger(__name__)

class GoldenCutAnalyzer(BaseAnalyzer):
    """黄金分割分析器类，用于计算和可视化股票的斐波那契回调水平"""
    
    def __init__(self, stock_code: str, stock_name: str = None, end_date: Union[str, datetime] = None, 
                 days: int = 365):
        """
        初始化黄金分割分析器
        
        参数:
            stock_code (str): 股票代码
            stock_name (str, 可选): 股票名称，如不提供则通过基类获取
            end_date (str 或 datetime, 可选): 结束日期，默认为当前日期
            days (int, 可选): 回溯天数，默认365天
        """
        super().__init__(stock_code, stock_name, end_date, days)
        
        # 初始化斐波那契相关属性
        self.fib_levels = {}
        self.plot_annotations = []
        self.swing_low_date = None
        self.swing_high_date = None
    
    def fetch_data(self) -> bool:
        """
        获取股票数据
        
        返回:
            bool: 是否成功获取数据
        """
        try:
            # 获取股票名称(如果未提供)
            if not self.stock_name or self.stock_name == self.stock_code:
                self.get_stock_name()
            
            # 获取股票日线数据
            self.daily_data = self.get_stock_daily_data()
            
            if self.daily_data.empty:
                logger.error(f"未能获取到股票 {self.stock_code} 的日线数据")
                return False
            
            # 确保日期成为索引，并且数据已排序
            if 'trade_date' in self.daily_data.columns:
                self.daily_data.set_index('trade_date', inplace=True)
            
            self.daily_data.sort_index(inplace=True)
            
            # 确保数据类型正确
            for col in ['open', 'close', 'high', 'low', 'volume']:
                if col in self.daily_data.columns:
                    self.daily_data[col] = pd.to_numeric(self.daily_data[col], errors='coerce')
            
            logger.info(f"成功获取 {self.stock_code} ({self.stock_name}) 的数据")
            return True
                
        except Exception as e:
            logger.error(f"获取数据时出错: {str(e)}")
            return False
    
    def _create_mock_data(self) -> bool:
        """
        创建模拟数据，用于测试
        
        返回:
            bool: 是否成功创建模拟数据
        """
        try:
            # 创建日期范围
            start_date = self.start_date.date()
            end_date = self.end_date.date()
            date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
                         if (start_date + timedelta(days=i)).weekday() < 5]  # 排除周末
            
            # 创建价格数据
            np.random.seed(42)  # 固定随机种子以便测试
            
            # 模拟一个趋势运动 + 随机波动
            n = len(date_range)
            trend = np.linspace(0, 30, n) + np.sin(np.linspace(0, 4*np.pi, n)) * 10
            noise = np.random.normal(0, 1, n)
            price_series = 100 + trend + noise
            
            # 计算各价格
            opens = price_series + np.random.normal(0, 0.5, n)
            highs = np.maximum(price_series, opens) + np.random.uniform(0.5, 2.0, n)
            lows = np.minimum(price_series, opens) - np.random.uniform(0.5, 2.0, n)
            closes = price_series + np.random.normal(0, 0.5, n)
            volumes = np.random.uniform(1000, 10000, n) * (1 + 0.1 * np.sin(np.linspace(0, 8*np.pi, n)))
            
            # 创建DataFrame
            self.daily_data = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }, index=date_range)
            
            logger.info(f"成功创建 {self.stock_code} 的模拟数据")
            return True
                
        except Exception as e:
            logger.error(f"创建模拟数据时出错: {str(e)}")
            return False
    
    def prepare_data(self) -> bool:
        """
        准备数据，计算必要的技术指标
        
        返回:
            bool: 是否成功准备数据
        """
        if self.daily_data is None or self.daily_data.empty:
            logger.warning(f"股票{self.stock_code}没有日线数据，将尝试创建模拟数据")
            if not self._create_mock_data():
                return False
        
        try:
            # 计算移动平均线等基本指标
            self.daily_data = calculate_technical_indicators(
                self.daily_data,
                ma_periods=[5, 10, 20, 30, 60, 120],
                vol_periods=[5, 10, 20]
            )
            
            logger.info(f"已为{self.stock_code}计算技术指标")
            return True
            
        except Exception as e:
            logger.error(f"准备数据时出错: {str(e)}")
            return False
    
    def calculate_fibonacci_levels(self) -> bool:
        """
        计算斐波那契回调水平
        
        返回:
            bool: 是否成功计算
        """
        if self.daily_data is None or self.daily_data.empty:
            logger.warning(f"股票{self.stock_code}没有日线数据，请先获取数据")
            return False
        
        try:
            # 使用指标工具计算斐波那契水平
            self.fib_levels = calculate_fibonacci_levels(self.daily_data, use_swing=True)
            
            if not self.fib_levels:
                logger.warning(f"未能计算有效的斐波那契回调水平")
                return False
            
            # 找到波段的起点和终点供绘图标记使用
            try:
                self.swing_low_date = self.daily_data['low'].idxmin()
                high_df = self.daily_data.loc[self.swing_low_date:]
                self.swing_high_date = high_df['high'].idxmax()
                
                # 准备标注信息
                self.plot_annotations = [
                    {
                        'xy': (self.swing_low_date, self.daily_data.loc[self.swing_low_date, 'low']), 
                        'text': f'波段低点\n{self.daily_data.loc[self.swing_low_date, "low"]:.2f}', 
                        'xytext': (-60, -30)
                    },
                    {
                        'xy': (self.swing_high_date, self.daily_data.loc[self.swing_high_date, 'high']), 
                        'text': f'波段高点\n{self.daily_data.loc[self.swing_high_date, "high"]:.2f}', 
                        'xytext': (10, 20)
                    }
                ]
            except Exception as e:
                logger.warning(f"计算波段端点时出错: {str(e)}")
                
            # 打印计算结果
            logger.info("计算得到的斐波那契回调位:")
            for label, price in self.fib_levels.items():
                logger.info(f"  {label}: {price:.2f}")
                
            return True
                
        except Exception as e:
            logger.error(f"计算斐波那契回调水平时出错: {str(e)}")
            return False
    
    def plot_chart(self, save_filename=None) -> bool:
        """
        绘制斐波那契回调图表
        
        参数:
            save_filename (str, 可选): 保存的文件名，默认为股票代码_斐波那契_日期.png
            
        返回:
            bool: 是否成功绘制并保存图表
        """
        if self.daily_data is None or self.daily_data.empty:
            logger.warning("无数据可绘制，请先获取数据")
            return False
        
        if not self.fib_levels:
            logger.warning("未计算斐波那契回调水平，请先计算")
            self.calculate_fibonacci_levels()
            
            if not self.fib_levels:
                logger.error("无法计算斐波那契回调水平")
                return False
        
        if save_filename is None:
            save_filename = f"{self.stock_code}_斐波那契_{self.end_date.strftime('%Y%m%d')}.png"
            
        save_path = os.path.join(self.save_path, save_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            # 使用通用绘图函数
            title = f'{self.stock_name}({self.stock_code}) 日线图与斐波那契回调'
            
            # 调用共享绘图函数
            fig, axes = plot_stock_chart(
                self.daily_data,
                title=title,
                save_path=None,  # 暂不保存，后面添加标注后再保存
                plot_ma=True,
                plot_volume=True,
                plot_fib=self.fib_levels
            )
            
            # 添加波段起止点标注
            if fig and len(axes) > 0 and self.plot_annotations:
                ax1 = axes[0]
                for ann in self.plot_annotations:
                    date = ann['xy'][0]
                    price = ann['xy'][1]
                    ax1.annotate(
                        ann['text'],
                        xy=(date, price),
                        xytext=ann['xytext'],
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2'),
                        fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
                    )
            
            # 保存图表
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"斐波那契回调图表已保存至: {save_path}")
            return True
                
        except Exception as e:
            logger.error(f"绘制图表时出错: {str(e)}")
            return False
    
    def generate_analysis_summary(self) -> str:
        """
        生成分析摘要
        
        返回:
            str: 分析摘要
        """
        if not self.fib_levels:
            logger.warning("未计算斐波那契回调水平，无法生成分析摘要")
            return "未计算斐波那契回调水平，无法生成分析摘要"
        
        try:
            # 获取最新价格
            current_price = self.daily_data['close'].iloc[-1]
            
            # 判断当前价格处于哪个斐波那契回调位附近
            price_position = "未知"
            nearest_level = None
            min_distance = float('inf')
            
            for level, price in self.fib_levels.items():
                distance = abs(current_price - price)
                if distance < min_distance:
                    min_distance = distance
                    nearest_level = level
                    
            if nearest_level:
                price_position = f"接近{nearest_level}回调位（{self.fib_levels[nearest_level]:.2f}）"
            
            # 计算从低点到高点的涨幅
            if self.swing_low_date and self.swing_high_date:
                low_price = self.daily_data.loc[self.swing_low_date, 'low']
                high_price = self.daily_data.loc[self.swing_high_date, 'high']
                swing_range_pct = (high_price / low_price - 1) * 100
            else:
                # 使用数据中的最高点和最低点
                low_price = self.daily_data['low'].min()
                high_price = self.daily_data['high'].max()
                swing_range_pct = (high_price / low_price - 1) * 100
            
            # 判断当前处于上涨还是下跌阶段
            if current_price > self.fib_levels.get('Fib 50.0%', 0):
                trend = "上涨趋势中"
            elif current_price < self.fib_levels.get('Fib 50.0%', 0):
                trend = "下跌调整中"
            else:
                trend = "处于中间位置，趋势不明确"
            
            # 支撑位和阻力位分析
            supports = []
            resistances = []
            
            for level, price in self.fib_levels.items():
                if price < current_price:
                    supports.append((level, price))
                else:
                    resistances.append((level, price))
            
            # 排序得到最近的支撑位和阻力位
            supports.sort(key=lambda x: current_price - x[1])
            resistances.sort(key=lambda x: x[1] - current_price)
            
            # 构建分析摘要
            summary = []
            summary.append(f"{self.stock_name}({self.stock_code})斐波那契回调分析")
            summary.append(f"分析日期: {self.end_date.strftime('%Y-%m-%d')}")
            summary.append(f"当前价格: {current_price:.2f}，{price_position}")
            summary.append(f"从低点到高点涨幅: {swing_range_pct:.2f}%")
            summary.append(f"当前趋势判断: {trend}")
            
            # 添加支撑位
            if supports:
                summary.append("主要支撑位:")
                for i, (level, price) in enumerate(supports[:3], 1):
                    summary.append(f"  {i}. {level} - {price:.2f}")
            
            # 添加阻力位
            if resistances:
                summary.append("主要阻力位:")
                for i, (level, price) in enumerate(resistances[:3], 1):
                    summary.append(f"  {i}. {level} - {price:.2f}")
            
            # 添加策略建议
            if current_price < self.fib_levels.get('Fib 38.2%', 0):
                summary.append("策略建议: 回调较深，可考虑分批买入")
            elif current_price > self.fib_levels.get('Fib 61.8%', 0):
                summary.append("策略建议: 上涨较多，注意风险，可考虑减仓")
            else:
                summary.append("策略建议: 处于回调中间区域，可少量试探性买入，设置止损")
            
            # 返回格式化的摘要
            return '\n'.join(summary)
                
        except Exception as e:
            error_msg = f"生成分析摘要时出错: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def run_analysis(self, save_path=None) -> Dict:
        """
        运行完整的分析流程
        
        参数:
            save_path (str, 可选): 图表保存路径，默认使用类初始化时的save_path
            
        返回:
            Dict: 分析结果
        """
        try:
            # 获取数据
            if not self.fetch_data():
                return {'status': 'error', 'message': '获取股票数据失败'}
            
            # 计算技术指标
            if not self.prepare_data():
                return {'status': 'error', 'message': '准备数据失败'}
            
            # 计算斐波那契回调水平
            if not self.calculate_fibonacci_levels():
                return {'status': 'error', 'message': '计算斐波那契回调水平失败'}
            
            # 绘制图表
            chart_path = ""
            if self.plot_chart():
                if save_path:
                    chart_path = os.path.join(self.save_path, f"{save_path}.png")
                else:
                    chart_path = os.path.join(self.save_path, f"{self.stock_code}_斐波那契_{self.end_date.strftime('%Y%m%d')}.png")
            
            # 生成分析摘要
            analysis_summary = self.generate_analysis_summary()
            
            # 整合结果
            result = {
                'status': 'success',
                'stock_code': self.stock_code,
                'stock_name': self.stock_name,
                'date': self.end_date.strftime('%Y-%m-%d'),
                'fibonacci_levels': self.fib_levels,
                'chart_path': chart_path,
                'description': analysis_summary
            }
            
            # 保存分析结果
            self.analysis_result = result
            
            logger.info(f"{self.stock_code} ({self.stock_name}) 斐波那契分析完成")
            return result
            
        except Exception as e:
            error_msg = f"运行分析流程时出错: {str(e)}"
            logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}


if __name__ == '__main__':
    # 直接运行测试
    analyzer = GoldenCutAnalyzer('000001')
    result = analyzer.run_analysis()
    print(result['description']) 