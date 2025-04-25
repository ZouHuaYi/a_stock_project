# -*- coding: utf-8 -*-
"""
缠论T+0训练系统模块包

该包包含以下模块：
- chan_data: 数据获取和处理
- chan_analysis: 信号分析和检测
- chan_trading: 交易执行和记录
- chan_visualization: 图表和可视化
- chan_trainer: 训练主类和交互
"""

from .chan_data import ChanDataHandler
from .chan_analysis import ChanAnalyzer
from .chan_trading import ChanTrader
from .chan_visualization import ChanVisualizer
from .chan_trainer import ChanTrainer

__all__ = [
    'ChanDataHandler',
    'ChanAnalyzer',
    'ChanTrader',
    'ChanVisualizer',
    'ChanTrainer'
] 