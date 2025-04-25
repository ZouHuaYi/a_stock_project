# -*- coding: utf-8 -*-
"""
缠论T+0训练系统 - 交易执行模块

该模块负责缠论训练系统的交易执行和记录，包括：
1. 模拟交易执行（买入、卖出）
2. 交易记录管理
3. 资金账户管理
4. 交易绩效统计
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import os
import logging

# 导入项目模块
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger(__name__)

class ChanTrader:
    """
    缠论交易器类
    
    该类负责执行交易操作和记录交易记录。
    """
    
    def __init__(self, stock_code: str, initial_capital: float = 100000.0, 
                 commission_rate: float = 0.0003, slippage: float = 0.002, 
                 save_path: str = None):
        """
        初始化交易器
        
        参数:
            stock_code (str): 股票代码
            initial_capital (float, 可选): 初始资金，默认100000.0
            commission_rate (float, 可选): 佣金率，默认0.0003
            slippage (float, 可选): 滑点率，默认0.002
            save_path (str, 可选): 交易记录保存路径
        """
        self.stock_code = stock_code
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.save_path = save_path or os.path.join('output', 'trades', stock_code)
        
        # 确保保存目录存在
        os.makedirs(self.save_path, exist_ok=True)
        
        # 持仓信息
        self.position = 0  # 持仓数量
        self.position_cost = 0.0  # 持仓成本
        self.last_deal_price = 0.0  # 最近成交价格
        
        # 交易记录
        self.trade_records = []
        
        # 交易统计
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
        self.logger = logger
        self.logger.info(f"初始化交易器: {stock_code}, 初始资金: {initial_capital}")

    def buy(self, price: float, volume: int, time: str, reason: str = "", 
            auto_adjust: bool = True, level: str = "") -> Dict[str, Any]:
        """
        执行买入操作
        
        参数:
            price (float): 买入价格
            volume (int): 买入数量
            time (str): 交易时间
            reason (str, 可选): 买入原因
            auto_adjust (bool, 可选): 是否自动调整数量以适应资金限制，默认True
            level (str, 可选): 交易级别
            
        返回:
            Dict[str, Any]: 交易结果
        """
        self.logger.info(f"尝试买入: 价格={price}, 数量={volume}, 时间={time}")
        
        # 检查是否有足够的资金
        estimated_cost = price * volume * (1 + self.commission_rate + self.slippage)
        
        if estimated_cost > self.current_capital:
            if auto_adjust:
                # 自动调整买入数量
                adjusted_volume = int(self.current_capital / (price * (1 + self.commission_rate + self.slippage)))
                adjusted_volume = adjusted_volume // 100 * 100  # 调整为100的整数倍
                
                if adjusted_volume <= 0:
                    self.logger.warning(f"资金不足，无法买入: 当前资金={self.current_capital}, 所需资金={estimated_cost}")
                    return {
                        'success': False,
                        'message': f"资金不足，无法买入: 当前资金={self.current_capital}, 所需资金={estimated_cost}",
                        'actual_volume': 0
                    }
                
                volume = adjusted_volume
                estimated_cost = price * volume * (1 + self.commission_rate + self.slippage)
                self.logger.info(f"自动调整买入数量为: {volume}")
            else:
                self.logger.warning(f"资金不足，无法买入: 当前资金={self.current_capital}, 所需资金={estimated_cost}")
                return {
                    'success': False,
                    'message': f"资金不足，无法买入: 当前资金={self.current_capital}, 所需资金={estimated_cost}",
                    'actual_volume': 0
                }
        
        # 计算实际成交价格（考虑滑点）
        actual_price = price * (1 + self.slippage)
        # 计算佣金
        commission = actual_price * volume * self.commission_rate
        # 计算总成本
        total_cost = actual_price * volume + commission
        
        # 更新资金和持仓
        self.current_capital -= total_cost
        
        # 如果已有持仓，计算新的持仓成本
        if self.position > 0:
            self.position_cost = (self.position_cost * self.position + actual_price * volume) / (self.position + volume)
        else:
            self.position_cost = actual_price
        
        self.position += volume
        self.last_deal_price = actual_price
        
        # 记录交易
        trade_record = {
            'time': time,
            'action': 'buy',
            'price': price,
            'actual_price': actual_price,
            'volume': volume,
            'commission': commission,
            'total_cost': total_cost,
            'reason': reason,
            'level': level,
            'current_capital': self.current_capital,
            'position': self.position,
            'position_cost': self.position_cost
        }
        self.trade_records.append(trade_record)
        self.total_trades += 1
        
        self.logger.info(f"买入成功: 价格={actual_price}, 数量={volume}, 总成本={total_cost}, 剩余资金={self.current_capital}")
        
        # 保存交易记录
        self._save_trade_records()
        
        return {
            'success': True,
            'message': "买入成功",
            'trade': trade_record,
            'actual_volume': volume
        }

    def sell(self, price: float, volume: int, time: str, reason: str = "", 
             level: str = "") -> Dict[str, Any]:
        """
        执行卖出操作
        
        参数:
            price (float): 卖出价格
            volume (int): 卖出数量，如果为None或超过持仓量，则卖出全部持仓
            time (str): 交易时间
            reason (str, 可选): 卖出原因
            level (str, 可选): 交易级别
            
        返回:
            Dict[str, Any]: 交易结果
        """
        self.logger.info(f"尝试卖出: 价格={price}, 数量={volume}, 时间={time}")
        
        # 检查持仓
        if self.position <= 0:
            self.logger.warning("无持仓，无法卖出")
            return {
                'success': False,
                'message': "无持仓，无法卖出",
                'actual_volume': 0
            }
        
        # 如果要卖出数量大于持仓量，则卖出全部持仓
        if volume > self.position:
            self.logger.info(f"卖出数量({volume})大于持仓量({self.position})，调整为卖出全部持仓")
            volume = self.position
        
        # 计算实际成交价格（考虑滑点）
        actual_price = price * (1 - self.slippage)
        # 计算佣金
        commission = actual_price * volume * self.commission_rate
        # 计算卖出所得
        proceeds = actual_price * volume - commission
        
        # 计算本次交易的盈亏
        profit_loss = (actual_price - self.position_cost) * volume - commission
        
        # 更新资金和持仓
        self.current_capital += proceeds
        self.position -= volume
        self.last_deal_price = actual_price
        
        # 如果全部卖出，重置持仓成本
        if self.position == 0:
            self.position_cost = 0.0
        
        # 更新交易统计
        if profit_loss > 0:
            self.winning_trades += 1
            self.total_profit += profit_loss
        else:
            self.losing_trades += 1
            self.total_loss += profit_loss
        
        # 记录交易
        trade_record = {
            'time': time,
            'action': 'sell',
            'price': price,
            'actual_price': actual_price,
            'volume': volume,
            'commission': commission,
            'proceeds': proceeds,
            'profit_loss': profit_loss,
            'reason': reason,
            'level': level,
            'current_capital': self.current_capital,
            'position': self.position,
            'position_cost': self.position_cost
        }
        self.trade_records.append(trade_record)
        self.total_trades += 1
        
        self.logger.info(f"卖出成功: 价格={actual_price}, 数量={volume}, 所得={proceeds}, "
                         f"盈亏={profit_loss}, 剩余资金={self.current_capital}, 剩余持仓={self.position}")
        
        # 保存交易记录
        self._save_trade_records()
        
        return {
            'success': True,
            'message': "卖出成功",
            'trade': trade_record,
            'actual_volume': volume,
            'profit_loss': profit_loss
        }

    def get_account_status(self) -> Dict[str, Any]:
        """
        获取账户状态
        
        返回:
            Dict[str, Any]: 账户状态信息
        """
        # 计算持仓市值
        position_value = self.position * self.last_deal_price if self.position > 0 and self.last_deal_price > 0 else 0
        # 计算总资产
        total_assets = self.current_capital + position_value
        # 计算盈亏率
        profit_rate = (total_assets - self.initial_capital) / self.initial_capital * 100 if self.initial_capital > 0 else 0
        
        return {
            'cash': self.current_capital,
            'position': self.position,
            'position_cost': self.position_cost,
            'position_value': position_value,
            'total_assets': total_assets,
            'profit_loss': total_assets - self.initial_capital,
            'profit_rate': profit_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'profit_factor': abs(self.total_profit / self.total_loss) if self.total_loss != 0 else float('inf')
        }

    def get_trade_summary(self) -> Dict[str, Any]:
        """
        获取交易摘要
        
        返回:
            Dict[str, Any]: 交易摘要信息
        """
        if not self.trade_records:
            return {
                'total_trades': 0,
                'message': "无交易记录"
            }
        
        # 获取账户状态
        account_status = self.get_account_status()
        
        # 计算平均持仓时间
        holding_times = []
        buy_time = None
        buy_volume = 0
        
        for trade in self.trade_records:
            if trade['action'] == 'buy':
                if buy_time is None:
                    buy_time = datetime.strptime(trade['time'], '%Y-%m-%d %H:%M:%S')
                    buy_volume = trade['volume']
                else:
                    # 增加持仓
                    buy_volume += trade['volume']
            elif trade['action'] == 'sell' and buy_time is not None:
                sell_time = datetime.strptime(trade['time'], '%Y-%m-%d %H:%M:%S')
                holding_time = (sell_time - buy_time).total_seconds() / 3600  # 小时为单位
                holding_times.append(holding_time)
                
                if trade['volume'] >= buy_volume:
                    # 全部卖出
                    buy_time = None
                    buy_volume = 0
                else:
                    # 部分卖出
                    buy_volume -= trade['volume']
        
        avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
        
        # 计算最大回撤
        cumulative_returns = []
        current_capital = self.initial_capital
        for trade in self.trade_records:
            if trade['action'] == 'buy':
                current_capital -= trade['total_cost']
            elif trade['action'] == 'sell':
                current_capital += trade['proceeds']
            cumulative_returns.append(current_capital)
        
        max_drawdown = 0
        peak = self.initial_capital
        for capital in cumulative_returns:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'stock_code': self.stock_code,
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': account_status['win_rate'],
            'profit_factor': account_status['profit_factor'],
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'net_profit': self.total_profit + self.total_loss,
            'profit_rate': account_status['profit_rate'],
            'max_drawdown': max_drawdown * 100,  # 转换为百分比
            'avg_holding_time': avg_holding_time,  # 小时
            'position': self.position,
            'position_cost': self.position_cost,
            'last_deal_price': self.last_deal_price
        }

    def _save_trade_records(self) -> None:
        """
        保存交易记录到文件
        """
        if not self.save_path:
            return
        
        file_path = os.path.join(self.save_path, f"{self.stock_code}_trades.json")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.trade_records, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"交易记录已保存到: {file_path}")
        except Exception as e:
            self.logger.error(f"保存交易记录失败: {str(e)}")
    
    def save_trade_summary(self) -> None:
        """
        保存交易摘要到文件
        """
        if not self.save_path:
            return
        
        summary = self.get_trade_summary()
        file_path = os.path.join(self.save_path, f"{self.stock_code}_summary.json")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            self.logger.info(f"交易摘要已保存到: {file_path}")
        except Exception as e:
            self.logger.error(f"保存交易摘要失败: {str(e)}")
    
    def load_trade_records(self) -> bool:
        """
        从文件加载交易记录
        
        返回:
            bool: 是否成功加载
        """
        if not self.save_path:
            return False
        
        file_path = os.path.join(self.save_path, f"{self.stock_code}_trades.json")
        if not os.path.exists(file_path):
            self.logger.warning(f"交易记录文件不存在: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.trade_records = json.load(f)
            
            # 重新计算统计信息
            self._recalculate_statistics()
            
            self.logger.info(f"成功加载交易记录，共{len(self.trade_records)}条")
            return True
        except Exception as e:
            self.logger.error(f"加载交易记录失败: {str(e)}")
            return False
    
    def _recalculate_statistics(self) -> None:
        """
        重新计算交易统计信息
        """
        self.total_trades = len(self.trade_records)
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
        for trade in self.trade_records:
            if trade['action'] == 'sell':
                if 'profit_loss' in trade:
                    if trade['profit_loss'] > 0:
                        self.winning_trades += 1
                        self.total_profit += trade['profit_loss']
                    else:
                        self.losing_trades += 1
                        self.total_loss += trade['profit_loss']
        
        # 获取最后一条交易记录的状态
        if self.trade_records:
            last_trade = self.trade_records[-1]
            if 'current_capital' in last_trade:
                self.current_capital = last_trade['current_capital']
            if 'position' in last_trade:
                self.position = last_trade['position']
            if 'position_cost' in last_trade:
                self.position_cost = last_trade['position_cost']
            if 'actual_price' in last_trade:
                self.last_deal_price = last_trade['actual_price']
        
        self.logger.info("交易统计信息已重新计算")
    
    def reset(self) -> None:
        """
        重置交易器状态
        """
        self.current_capital = self.initial_capital
        self.position = 0
        self.position_cost = 0.0
        self.last_deal_price = 0.0
        self.trade_records = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
        self.logger.info("交易器状态已重置")
    
    def adjust_initial_capital(self, new_capital: float) -> None:
        """
        调整初始资金
        
        参数:
            new_capital (float): 新的初始资金
        """
        if new_capital <= 0:
            self.logger.warning(f"初始资金必须为正数: {new_capital}")
            return
        
        # 如果已经有交易，不允许调整
        if self.trade_records:
            self.logger.warning("已有交易记录，不能调整初始资金")
            return
        
        self.initial_capital = new_capital
        self.current_capital = new_capital
        self.logger.info(f"初始资金已调整为: {new_capital}")
    
    def manual_buy(self, price: float, time: str, available_capital: float = None) -> Dict[str, Any]:
        """
        手动买入（用户交互）
        
        参数:
            price (float): 买入价格
            time (str): 交易时间
            available_capital (float, 可选): 可用资金，默认为当前资金
            
        返回:
            Dict[str, Any]: 交易结果
        """
        if available_capital is None:
            available_capital = self.current_capital
        
        max_volume = int(available_capital / (price * (1 + self.commission_rate + self.slippage)))
        max_volume = max_volume // 100 * 100  # 调整为100的整数倍
        
        self.logger.info(f"当前价格: {price}, 可买入最大数量: {max_volume}, 可用资金: {available_capital}")
        
        return {
            'max_volume': max_volume,
            'price': price,
            'time': time
        }
    
    def manual_sell(self, price: float, time: str) -> Dict[str, Any]:
        """
        手动卖出（用户交互）
        
        参数:
            price (float): 卖出价格
            time (str): 交易时间
            
        返回:
            Dict[str, Any]: 交易信息
        """
        if self.position <= 0:
            self.logger.warning("无持仓，无法卖出")
            return {
                'position': 0,
                'price': price,
                'time': time
            }
        
        self.logger.info(f"当前价格: {price}, 当前持仓: {self.position}, 持仓成本: {self.position_cost}")
        
        # 计算预期盈亏
        expected_profit = (price * (1 - self.slippage) - self.position_cost) * self.position
        expected_profit_rate = (price * (1 - self.slippage) / self.position_cost - 1) * 100
        
        return {
            'position': self.position,
            'price': price,
            'cost': self.position_cost,
            'expected_profit': expected_profit,
            'expected_profit_rate': expected_profit_rate,
            'time': time
        } 