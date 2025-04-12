# -*- coding: utf-8 -*-
"""量价关系分析器模块，用于识别股票的洗盘、拉升等特征"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# 导入基础分析器和工具
from analyzer.base_analyzer import BaseAnalyzer
from config import ANALYZER_CONFIG, PATH_CONFIG
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger(__name__)

class VolPriceAnalyzer(BaseAnalyzer):
    """量价关系分析器类，用于识别股票的洗盘、拉升等特征"""
    
    def __init__(self, stock_code: str, stock_name: str = None, end_date: Union[str, datetime] = None, days: int = 60):
        """
        初始化量价关系分析器
        
        参数:
            stock_code (str): 股票代码
            stock_name (str, 可选): 股票名称，如不提供则通过基类获取
            end_date (str 或 datetime, 可选): 结束日期，默认为当前日期
            days (int, 可选): 回溯天数，默认60天
        """
        super().__init__(stock_code, stock_name, end_date, days)
    
    def prepare_data(self):
        """
        准备数据，计算各种量价指标
        
        返回:
            bool: 是否成功准备数据
        """
        if self.daily_data is None or self.daily_data.empty:
            logger.warning(f"股票{self.stock_code}没有日线数据，请先获取数据")
            return False
        
        try:
            # 计算移动平均线
            self.daily_data.loc[:, 'MA5'] = self.daily_data['close'].rolling(window=5).mean()
            self.daily_data.loc[:, 'MA10'] = self.daily_data['close'].rolling(window=10).mean()
            self.daily_data.loc[:, 'MA20'] = self.daily_data['close'].rolling(window=20).mean()
            self.daily_data.loc[:, 'MA30'] = self.daily_data['close'].rolling(window=30).mean()
            
            # 计算成交量移动平均
            self.daily_data.loc[:, 'VOL_MA5'] = self.daily_data['volume'].rolling(window=5).mean()
            self.daily_data.loc[:, 'VOL_MA10'] = self.daily_data['volume'].rolling(window=10).mean()
            
            # 计算量比（当日成交量/5日平均成交量）
            self.daily_data['volume_ratio'] = self.daily_data['volume'] / self.daily_data['VOL_MA5']
            
            # 计算涨跌幅
            self.daily_data['price_change_pct'] = self.daily_data['close'].pct_change() * 100
            
            return True
        except Exception as e:
            logger.error(f"准备量价数据时出错: {e}")
            return False
    
    def detect_wash_patterns(self, window=15):
        """
        检测洗盘特征
        
        参数:
            window (int): 检查的时间窗口，默认15天
            
        返回:
            list: 洗盘特征列表，包含特征类型和位置
        """
        if self.daily_data is None or len(self.daily_data) < window:
            logger.warning("数据不足，无法检测洗盘特征")
            return []
        
        # 复制最后window天的数据进行分析
        recent_data = self.daily_data.iloc[-window:].copy()
        patterns = []
        
        # 检测特征1：成交量显著萎缩
        vol_ratio = recent_data['volume'].mean() / self.daily_data['volume'].iloc[-(window*2):-window].mean()
        if vol_ratio < 0.7:  # 成交量萎缩到之前的70%以下
            patterns.append({
                'type': '成交量萎缩',
                'ratio': vol_ratio,
                'description': f'成交量萎缩至前期的{vol_ratio:.2f}倍'
            })
        
        # 检测特征2：价格在下跌过程中企稳（震荡但守住关键位）
        price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['low'].min()
        price_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        if price_range < 0.15 and abs(price_trend) < 0.05:  # 价格波动不大，且趋势不明显
            patterns.append({
                'type': '横盘震荡',
                'range': price_range,
                'trend': price_trend,
                'description': f'价格横盘震荡，波动率{price_range:.2%}，趋势{price_trend:.2%}'
            })
        
        # 检测特征3：下影线信号（买盘暗中接盘）
        lower_shadows = (recent_data['open'] - recent_data['low']) / recent_data['low']
        if lower_shadows.mean() > 0.015:  # 平均下影线长度超过1.5%
            patterns.append({
                'type': '下影线信号',
                'avg_shadow': lower_shadows.mean(),
                'description': f'下影线明显，平均长度{lower_shadows.mean():.2%}'
            })
        
        # 检测特征4：缩量十字星（犹豫信号，通常出现在洗盘后期）
        dojis = recent_data[abs(recent_data['close'] - recent_data['open']) / recent_data['close'] < 0.005]
        if len(dojis) >= window * 0.3:  # 30%的天数出现十字星
            doji_vol_ratio = dojis['volume'].mean() / recent_data['volume'].mean()
            if doji_vol_ratio < 0.8:  # 十字星天的成交量比平均小
                patterns.append({
                    'type': '缩量企稳',
                    'doji_count': len(dojis),
                    'vol_ratio': doji_vol_ratio,
                    'description': f'出现{len(dojis)}天十字星形态，成交量为平均的{doji_vol_ratio:.2f}倍'
                })
        
        return patterns
    
    def analyze_vol_price(self):
        """
        分析股票的量价关系，检测洗盘特征
        
        返回:
            dict: 分析结果
        """
        if not self.prepare_data():
            return {'status': 'error', 'message': '数据准备失败'}
        
        # 获取最近数据
        last_data = self.daily_data.iloc[-1]
        patterns = self.detect_wash_patterns()
        
        # 判断是否可能处于洗盘阶段
        is_washing = False
        wash_confidence = 0
        
        # 根据检测到的特征判断洗盘可能性
        if patterns:
            wash_features = [p['type'] for p in patterns]
            if '成交量萎缩' in wash_features:
                wash_confidence += 40
            if '横盘震荡' in wash_features:
                wash_confidence += 30
            if '下影线信号' in wash_features:
                wash_confidence += 20
            if '缩量企稳' in wash_features:
                wash_confidence += 10
            
            is_washing = wash_confidence >= 50
        
        # 形成分析结论
        analysis_result = {
            'status': 'success',
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'date': self.end_date.strftime('%Y-%m-%d'),
            'is_washing': is_washing,
            'wash_confidence': wash_confidence,
            'patterns': patterns,
            'latest_price': last_data['close'],
            'latest_volume': last_data['volume'],
            'volume_ratio': last_data['volume_ratio'] if 'volume_ratio' in last_data else None,
            'ma5': last_data['MA5'],
            'ma20': last_data['MA20'],
            'description': self.generate_analysis_description(is_washing, wash_confidence, patterns)
        }
        
        return analysis_result
    
    def generate_analysis_description(self, is_washing, wash_confidence, patterns):
        """
        生成分析描述文本
        
        参数:
            is_washing (bool): 是否处于洗盘阶段
            wash_confidence (float): 洗盘可能性得分
            patterns (list): 检测到的特征列表
            
        返回:
            str: 分析描述
        """
        if not patterns:
            return f"{self.stock_name}({self.stock_code})未检测到明显的洗盘特征，可能处于正常交易状态。"
        
        desc_parts = []
        if is_washing:
            desc_parts.append(f"{self.stock_name}({self.stock_code})可能处于洗盘阶段(可信度:{wash_confidence}%)。")
        else:
            desc_parts.append(f"{self.stock_name}({self.stock_code})检测到部分洗盘特征，但不足以确认(可信度:{wash_confidence}%)。")
        
        desc_parts.append("检测到的特征包括：")
        for i, p in enumerate(patterns, 1):
            desc_parts.append(f"{i}. {p['description']}")
        
        if is_washing:
            desc_parts.append("建议：可考虑观察是否出现放量上涨信号，洗盘结束通常伴随着成交量增加和股价突破。")
        
        return "".join(desc_parts)
    
    def plot_vol_price_chart(self, save_filename=None):
        """
        绘制量价关系图，突出显示洗盘特征
        
        参数:
            save_filename (str, 可选): 保存的文件名，默认为股票代码_洗盘分析_日期.png
        """
        if self.daily_data is None or self.daily_data.empty:
            logger.warning("无数据可绘制。请先获取数据。")
            return False

        if save_filename is None:
            save_filename = f"{self.stock_code}_洗盘分析_{self.end_date.strftime('%Y%m%d')}.png"
        else:
            save_filename = f"{save_filename}.png"
            
        save_path = os.path.join(self.save_path, save_filename)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 尝试再次进行量价分析
        try:
            analysis_result = self.analyze_vol_price()
            if analysis_result['status'] != 'success':
                logger.error(f"分析失败，无法绘制图表: {analysis_result.get('message', '未知错误')}")
                return False
            patterns = analysis_result.get('patterns', [])
        except Exception as e:
            logger.error(f"分析过程出错: {e}")
            # 使用空模式列表继续绘图
            patterns = []
            analysis_result = {
                'status': 'success',
                'is_washing': False,
                'wash_confidence': 0,
                'description': f"{self.stock_name}({self.stock_code})分析出错，无法显示结果。"
            }
        
        # 创建一个包含两个子图的Figure(主图和成交量图)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        title = f'{self.stock_name}({self.stock_code}) 量价关系分析'
        if analysis_result['is_washing']:
            title += f" - 疑似洗盘(可信度:{analysis_result['wash_confidence']}%)"
        fig.suptitle(title, fontsize=16)
        
        # 选择用于绘图的数据点(最后30天)
        plot_days = min(30, len(self.daily_data))
        plot_df = self.daily_data.iloc[-plot_days:].copy()
        
        # 绘制主图：收盘价和移动平均线
        ax1.plot(plot_df.index, plot_df['close'], label='收盘价', linewidth=2, color='black')
        ax1.plot(plot_df.index, plot_df['MA5'], label='5日均线', linewidth=1.5, color='red')
        ax1.plot(plot_df.index, plot_df['MA10'], label='10日均线', linewidth=1.5, color='blue')
        ax1.plot(plot_df.index, plot_df['MA20'], label='20日均线', linewidth=1.5, color='green')
        
        # 标记横盘震荡区域(如果检测到)
        if patterns and any(p['type'] == '横盘震荡' for p in patterns):
            # 获取最近15天数据
            recent_15d = self.daily_data.iloc[-15:]
            # 添加长方形标记震荡区域
            rect = Rectangle(
                (recent_15d.index[0], recent_15d['low'].min()*0.99),
                recent_15d.index[-1] - recent_15d.index[0],
                recent_15d['high'].max()*1.01 - recent_15d['low'].min()*0.99,
                linewidth=2, edgecolor='purple', facecolor='purple', alpha=0.1
            )
            ax1.add_patch(rect)
            
            # 添加文字说明
            for p in patterns:
                if p['type'] == '横盘震荡':
                    ax1.annotate(
                        p['description'], 
                        xy=(recent_15d.index[len(recent_15d)//2], (recent_15d['high'].max() + recent_15d['low'].min())/2),
                        xytext=(0, 30),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='purple'),
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3)
                    )
        
        # 标记下影线信号(如果检测到)
        if patterns and any(p['type'] == '下影线信号' for p in patterns):
            # 找出最明显的3根下影线
            recent_15d = self.daily_data.iloc[-15:]
            lower_shadows = (recent_15d['open'] - recent_15d['low']) / recent_15d['low']
            top_shadows_idx = lower_shadows.nlargest(3).index
            
            for idx in top_shadows_idx:
                if idx in plot_df.index:
                    row = self.daily_data.loc[idx]
                    ax1.plot(
                        [idx, idx], 
                        [row['low'], min(row['open'], row['close'])], 
                        color='purple', linewidth=2
                    )
                    ax1.annotate(
                        "下影线", 
                        xy=(idx, row['low']),
                        xytext=(0, -25),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='purple'),
                        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
                    )
        
        # 设置主图标题和标签
        ax1.set_title('价格走势与关键均线', fontsize=14)
        ax1.set_ylabel('价格', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='best')
        
        # 绘制成交量图
        volume_colors = ['green' if plot_df.loc[idx, 'close'] < plot_df.loc[idx, 'open'] 
                         else 'red' for idx in plot_df.index]
        
        ax2.bar(plot_df.index, plot_df['volume'], color=volume_colors, alpha=0.7)
        ax2.plot(plot_df.index, plot_df['VOL_MA5'], color='blue', linewidth=1.5, label='5日均量')
        
        # 标记成交量萎缩
        if patterns and any(p['type'] == '成交量萎缩' for p in patterns):
            # 使用更安全的方式获取数据点
            try:
                mid_point = len(plot_df) // 2
                ax2.annotate('成交量萎缩区域', 
                            xy=(plot_df.index[mid_point], plot_df['volume'].mean()),
                            xytext=(plot_df.index[mid_point], plot_df['volume'].max() * 0.8),
                            arrowprops=dict(facecolor='purple', shrink=0.05),
                            horizontalalignment='center', verticalalignment='top')
            except Exception as e:
                logger.error(f"标记成交量萎缩区域时出错: {e}")
        
        # 设置成交量图标题和标签
        ax2.set_title('成交量分析', fontsize=14)
        ax2.set_ylabel('成交量', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 调整x轴日期格式
        plt.xticks(rotation=45)
        
        # 如果索引是DatetimeIndex，使用日期格式化器
        if isinstance(self.daily_data.index, pd.DatetimeIndex):
            date_format = mdates.DateFormatter('%Y-%m-%d')
            ax2.xaxis.set_major_formatter(date_format)
            # 设置适当的日期定位器
            ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
        
        # 添加分析结论
        desc = analysis_result.get('description', f"无法获取{self.stock_name}({self.stock_code})的分析结论")
        plt.figtext(0.5, 0.01, desc, ha='center', fontsize=12, 
                  bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        try:
            # 保存图表
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"量价关系分析图已保存至: {save_path}")
            plt.close(fig)  # 关闭图形，释放内存
            return True
        except Exception as e:
            logger.error(f"保存图表失败: {e}")
            return False
    
    def run_analysis(self, save_path=None):
        """
        运行完整分析流程
        
        返回:
            dict: 分析结果
        """
        if self.get_stock_daily_data().empty:
            return {'status': 'error', 'message': '获取数据失败'}
        
        # 先尝试准备数据
        if not self.prepare_data():
            return {'status': 'error', 'message': '数据准备失败', 'patterns': []}
        
        # 分析量价关系
        analysis_result = self.analyze_vol_price()
        
        # 如果分析成功，则绘制图表
        if analysis_result['status'] == 'success':
            try:
                self.plot_vol_price_chart(save_path)
            except Exception as e:
                logger.error(f"绘制图表失败: {e}")
                analysis_result['chart_error'] = str(e)
        
        # 保存分析结果
        self.save_analysis_result(analysis_result)
        
        return analysis_result 