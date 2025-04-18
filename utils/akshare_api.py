# -*- coding: utf-8 -*-
"""AkShare API接口封装"""

import json
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Union, Any

# 导入配置和日志
from config import DATA_CONFIG
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger(__name__)

class AkshareAPI:
    """AkShare API接口封装，提供股票数据获取功能"""
    def __init__(self):
        """初始化AkShare API封装"""
        self.retry_count = DATA_CONFIG.get('retry_count', 3)
        self.retry_interval = DATA_CONFIG.get('retry_interval', 5)
    
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取A股股票列表
        
        返回:
            pd.DataFrame: 股票列表数据框
        """
        logger.info("正在获取A股股票列表...")
        
        try:
            # 重试机制
            for i in range(self.retry_count):
                try:
                    # 通过 AkShare 获取股票列表
                    stock_list = ak.stock_zh_a_spot_em()
                    
                    if not stock_list.empty:
                        # 提取需要的列
                        result = stock_list[['代码', '名称', '总市值']].copy()
                        result.columns = ['stock_code', 'stock_name', 'market_cap']
                        
                        # 添加交易所和行业信息
                        result['exchange'] = result['stock_code'].apply(
                            lambda x: 'SH' if x.startswith('6') else 'SZ'
                        )
                        
                        # 获取行业信息(可选，速度较慢)
                        # result = self._add_industry_info(result)
                        
                        # 设置默认值
                        result['industry'] = '未知'
                        result['list_date'] = datetime.now().strftime('%Y-%m-%d')
                        
                        logger.info(f"成功获取 {len(result)} 只股票信息")
                        return result
                    
                    logger.warning(f"获取股票列表失败，第 {i+1} 次重试中...")
                    time.sleep(self.retry_interval)
                    
                except Exception as e:
                    logger.error(f"获取股票列表出错 (尝试 {i+1}/{self.retry_count}): {str(e)}")
                    time.sleep(self.retry_interval)
            
            logger.error(f"获取股票列表失败，已重试 {self.retry_count} 次")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取股票列表时出错: {str(e)}")
            return pd.DataFrame()
    
    def get_stock_name(self, stock_code: str) -> str:
        """
        根据股票代码获取股票名称
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            str: 股票名称
        """
        logger.info(f"正在获取股票 {stock_code} 的名称...")
        
        try:
            # 通过 AkShare 获取股票名称
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            stock_name = stock_info.iloc[1, 1]
            if stock_name:
                logger.info(f"成功获取股票 {stock_code} 的名称: {stock_name}")
                return stock_name
            else:
                logger.warning(f"未找到股票代码 {stock_code} 对应的名称")
                return stock_code
                
        except Exception as e:
            logger.error(f"获取股票名称时出错: {str(e)}")
            return stock_code
    
    def get_stock_history(self, stock_code: str, period: str = "daily", 
                         years: int = 1, adjust: str = "qfq") -> pd.DataFrame:
        """
        获取股票历史数据
        
        参数:
            stock_code (str): 股票代码
            period (str): 周期，如 "daily", "weekly", "monthly"
            years (int): 获取多少年的数据
            adjust (str): 复权方式，如 "qfq"(前复权), "hfq"(后复权), 或 None(不复权)
            
        返回:
            pd.DataFrame: 股票历史数据
        """
        logger.info(f"正在获取股票 {stock_code} 的历史数据...")
        
        try:
            # 计算日期范围
            end_date = datetime.now() + timedelta(days=1)
            start_date = end_date - timedelta(days=years * 365)
            
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = end_date.strftime('%Y%m%d')
            
            # 重试机制
            for i in range(self.retry_count):
                try:
                    # 通过 AkShare 获取历史数据
                    stock_data = ak.stock_zh_a_hist(
                        symbol=stock_code,
                        period=period,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        adjust=adjust
                    )
                    
                    if not stock_data.empty:
                        # 标准化列名
                        column_map = {
                            '股票代码': 'stock_code',
                            '日期': 'trade_date',
                            '开盘': 'open',
                            '收盘': 'close',
                            '最高': 'high',
                            '最低': 'low',
                            '成交量': 'volume',
                            '成交额': 'amount',
                            '振幅': 'amplitude',
                            '涨跌幅': 'change_percent',
                            '涨跌额': 'change_amount',
                            '换手率': 'turnover_rate'
                        }
                        stock_data.rename(columns=column_map, inplace=True)
                        
                        # 转换日期格式
                        stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
                        
                        # 确保数值列类型正确
                        for col in ['open', 'close', 'high', 'low', 'volume', 'amount']:
                            if col in stock_data.columns:
                                stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                                
                        logger.info(f"成功获取 {len(stock_data)} 条 {stock_code} 历史数据")
                        return stock_data
                    
                    logger.warning(f"获取股票 {stock_code} 历史数据失败，第 {i+1} 次重试中...")
                    time.sleep(self.retry_interval)
                    
                except Exception as e:
                    logger.error(f"获取股票 {stock_code} 历史数据出错 (尝试 {i+1}/{self.retry_count}): {str(e)}")
                    time.sleep(self.retry_interval)
            
            logger.error(f"获取股票 {stock_code} 历史数据失败，已重试 {self.retry_count} 次")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取股票历史数据时出错: {str(e)}")
            return pd.DataFrame()
        
    def get_stock_history_min(self, stock_code: str, period: str = "30", 
                              days: int = 7) -> pd.DataFrame:
        """
        获取股票分钟级别历史数据
        
        参数:
            stock_code (str): 股票代码
            period (str): 周期，如 {'1', '5', '15', '30', '60'}
            
        """
        logger.info(f"正在获取股票 {stock_code} 的分钟级别历史数据...")

        try:
            # 计算日期范围
            end_date = datetime.now() + timedelta(days=1)
            start_date = end_date - timedelta(days)
            # 将日期转换为字符串格式 到分钟 开始是 00:00:00 结束是 23:59:59
            start_date_str = start_date.strftime('%Y%m%d 00:00:00')
            end_date_str = end_date.strftime('%Y%m%d 23:59:59')

            # 重试机制
            for i in range(self.retry_count):
                try:
                    # 通过 AkShare 获取分钟级别历史数据
                    stock_data = ak.index_zh_a_hist_min_em(
                        symbol=stock_code,
                        period=period,
                        start_date=start_date_str,
                        end_date=end_date_str
                    )
                    if not stock_data.empty:    
                        # 标准化列名
                        column_map = {
                            '时间': 'trade_date',
                            '开盘': 'open',
                            '收盘': 'close',
                            '最高': 'high',
                            '最低': 'low',
                            '成交量': 'volume',
                            '成交额': 'amount'
                        }
                        logger.info(f"请求 {stock_code} 分钟级别历史数据成功，数据长度为 {len(stock_data)}")
                        # 确保数值列类型正确
                        for col in ['open', 'close', 'high', 'low', 'volume', 'amount']:
                            if col in stock_data.columns:
                                stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')   

                         # 只保留上面的 column_map 中的列
                        stock_data = stock_data[column_map.keys()]
                        stock_data.rename(columns=column_map, inplace=True)
                        
                        # 将数据按分钟分组
                        stock_data['stock_code'] = stock_code
                        
                        # 如果 stock_data[open] 位空 或者 等于 0 ，0.00 则那么这个值就等于 前一行的close, 如果前一个close不存在则删除这行
                        # 处理开盘价为空的记录
                        for index, row in stock_data.iterrows():
                            if pd.isna(row['open']) or row['open'] == 0 or row['open'] == 0.00:
                                if index > 0:
                                    # 使用前一个收盘价填充开盘价
                                    stock_data.at[index, 'open'] = stock_data.at[index - 1, 'close']
                                

                        # 如果 stock_data[open] 位空 或者 等于 0 ，0.00 则那么这个值就等于 前一行的close, 如果前一个close不存在则删除这行
                        stock_data = stock_data[stock_data['open'].notna() & (stock_data['open'] != 0) & (stock_data['open'] != 0.00)]

                        logger.info(f"成功获取 {len(stock_data)} 条 {stock_code} 分钟级别历史数据")
                        return stock_data
                    
                    logger.warning(f"获取股票 {stock_code} 分钟级别历史数据失败，第 {i+1} 次重试中...")
                    time.sleep(self.retry_interval)
    
                except Exception as e:
                    logger.error(f"获取股票 {stock_code} 分钟级别历史数据出错 (尝试 {i+1}/{self.retry_count}): {str(e)}")
                    time.sleep(self.retry_interval)
            
            logger.error(f"获取股票 {stock_code} 分钟级别历史数据失败，已重试 {self.retry_count} 次")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取股票分钟级别历史数据时出错: {str(e)}")
            return pd.DataFrame()

    def get_financial_reports(self, stock_code: str, start_year: str = None) -> pd.DataFrame:
        """
        获取财务报表数据
        
        参数:
            stock_code (str): 股票代码
            start_year (str, 可选): 开始年份
            
        返回:
            pd.DataFrame: 财务报表数据
        """
        logger.info(f"正在获取股票 {stock_code} 的财务数据...")
        
        try:
            # 去除可能的前缀
            stock_code = stock_code.replace('sh', '').replace('sz', '').strip()
                
            # 重试机制
            for i in range(self.retry_count):
                try:
                    # 使用 AkShare 的 stock_financial_analysis_indicator 获取财务数据
                    financial_data = ak.stock_financial_analysis_indicator(symbol=stock_code, start_year=start_year)
                    
                    if not financial_data.empty:
                        logger.info(f"成功获取 {stock_code} 财务数据，共 {len(financial_data)} 条记录")
                        return financial_data
                    
                    logger.warning(f"获取股票 {stock_code} 财务数据失败，第 {i+1} 次重试中...")
                    time.sleep(self.retry_interval)
                    
                except Exception as e:
                    logger.error(f"获取股票 {stock_code} 财务数据出错 (尝试 {i+1}/{self.retry_count}): {str(e)}")
                    time.sleep(self.retry_interval)
            
            logger.error(f"获取股票 {stock_code} 财务数据失败，已重试 {self.retry_count} 次")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取财务报表数据时出错: {str(e)}")
            return pd.DataFrame()
  
    def generate_technical_summary(self, df: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict:
        """
        生成技术分析摘要
        
        参数:
            df (pd.DataFrame): 股票价格数据
            indicators (Dict[str, pd.Series]): 技术指标
            
        返回:
            Dict: 技术分析摘要
        """
        summary = {}
        try:
            # 获取最新数据
            current_price = df['close'].iloc[-1] if 'close' in df.columns else None
            
            if current_price is None:
                logger.error("无法获取当前价格，无法生成技术分析摘要")
                return {}
                
            # 基本价格信息
            summary['当前价格'] = round(current_price, 2)
           
            if 'stock_code' in df.columns or df.get('stock_code') is not None:
                stock_code = df['stock_code'].iloc[0] if 'stock_code' in df.columns else df.get('stock_code')
                summary['股票代码'] = stock_code

            # 价格变动
            if len(df) > 1:
                prev_close = df['close'].iloc[-2]
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100
                
                summary['涨跌额'] = round(change, 2)
                summary['涨跌幅'] = f"{round(change_percent, 2)}%"
            
            # 技术指标分析
            tech_signals = []
            
            # 均线分析
            if all(f'MA{period}' in indicators for period in [5, 20]):
                ma5 = indicators['MA5'].iloc[-1]
                ma20 = indicators['MA20'].iloc[-1]
                
                if ma5 > ma20:
                    tech_signals.append("均线多头排列")
                    if ma5 > ma20 and indicators['MA5'].iloc[-2] <= indicators['MA20'].iloc[-2]:
                        tech_signals.append("5日均线上穿20日均线，黄金交叉")
                else:
                    tech_signals.append("均线空头排列")
                    if ma5 < ma20 and indicators['MA5'].iloc[-2] >= indicators['MA20'].iloc[-2]:
                        tech_signals.append("5日均线下穿20日均线，死亡交叉")
            
            # MACD分析
            if all(key in indicators for key in ['MACD_DIF', 'MACD_DEA', 'MACD_BAR']):
                macd = indicators['MACD_DIF'].iloc[-1]
                signal = indicators['MACD_DEA'].iloc[-1]
                hist = indicators['MACD_BAR'].iloc[-1]
                
                if macd > signal:
                    tech_signals.append("MACD金叉")
                else:
                    tech_signals.append("MACD死叉")
                    
                if hist > 0 and indicators['MACD_BAR'].iloc[-2] <= 0:
                    tech_signals.append("MACD柱状图由负转正")
                elif hist < 0 and indicators['MACD_BAR'].iloc[-2] >= 0:
                    tech_signals.append("MACD柱状图由正转负")
            
            # KDJ分析
            if all(key in indicators for key in ['KDJ_K', 'KDJ_D', 'KDJ_J']):
                k = indicators['KDJ_K'].iloc[-1]
                d = indicators['KDJ_D'].iloc[-1]
                
                if k > d:
                    tech_signals.append("KDJ金叉")
                    if k < 20:
                        tech_signals.append("KDJ超卖区域金叉")
                else:
                    tech_signals.append("KDJ死叉")
                    if k > 80:
                        tech_signals.append("KDJ超买区域死叉")
            
            # RSI分析
            if 'RSI_14' in indicators or 'RSI_12' in indicators:
                rsi_key = 'RSI_14' if 'RSI_14' in indicators else 'RSI_12'
                rsi = indicators[rsi_key].iloc[-1]

                if rsi > 70:
                    tech_signals.append(f"{rsi_key}超买({round(rsi, 2)})")
                elif rsi < 30:
                    tech_signals.append(f"{rsi_key}超卖({round(rsi, 2)})")
                else:
                    tech_signals.append(f"{rsi_key}中位({round(rsi, 2)})")
            
            # 将信号添加到摘要中
            summary['技术信号'] = tech_signals
            
            # 整体研判
            if len(tech_signals) > 0:
                bullish_signals = sum(1 for signal in tech_signals if any(term in signal for term in ['多头', '金叉', '上穿', '超卖']))
                bearish_signals = sum(1 for signal in tech_signals if any(term in signal for term in ['空头', '死叉', '下穿', '超买']))
                
                if bullish_signals > bearish_signals:
                    summary['整体研判'] = "偏多"
                elif bearish_signals > bullish_signals:
                    summary['整体研判'] = "偏空"
                else:
                    summary['整体研判'] = "震荡"
            
            logger.info("技术分析摘要生成完成")
            return summary
            
        except Exception as e:
            logger.error(f"生成技术分析摘要时出错: {str(e)}")
            return {}
    
    def generate_financial_summary(self, stock_code: str, stock_name: str, financial_reports: pd.DataFrame) -> Dict:
        """
        生成财务分析摘要
        
        参数:
            stock_code (str): 股票代码
            stock_name (str): 股票名称
            financial_reports (pd.DataFrame): 财务报表数据
            
        返回:
            Dict: 财务分析摘要
        """
        summary = {
            '股票代码': stock_code,
            '股票名称': stock_name,
            '分析日期': datetime.now().strftime('%Y-%m-%d')
        }
        
        try:
            # 检查财务数据是否为空
            if financial_reports is None or not isinstance(financial_reports, pd.DataFrame) or financial_reports.empty:
                logger.warning(f"股票 {stock_code} 的财务数据为空或格式不正确")
                summary['财务状况'] = '无财务数据'
                return summary
                
            # 获取最新一期的财务数据
            latest_data = financial_reports.iloc[0]  # 假设数据是按时间降序排列的
            
            # 提取关键财务指标
            key_metrics = {}
            possible_metrics = {
                'roe': '净资产收益率(%)',  # 盈利能力
                'net_profit_margin': '销售净利率(%)',  # 盈利能力
                'gross_profit_margin': '销售毛利率(%)',  # 盈利能力
                'debt_to_asset_ratio': '资产负债率(%)',  # 偿债能力
                'current_ratio': '流动比率',  # 短期偿债能力
                'quick_ratio': '速动比率',  # 短期偿债能力
                'total_asset_turnover': '总资产周转率(次)',  # 营运能力
                'eps': '摊薄每股收益(元)',  # 每股指标
                'net_profit_growth_rate': '净利润增长率(%)'  # 成长能力
            }
            
            # 尝试找到对应的列名
            for key, col_name in possible_metrics.items():
                if col_name in latest_data.index:
                    key_metrics[key] = latest_data[col_name]
                    logger.debug(f"找到指标: {col_name} = {key_metrics[key]}")
                else:
                    key_metrics[key] = 'N/A'
                    logger.debug(f"未找到指标: {col_name}")
            
            # 生成财务分析摘要
            if latest_data.name:
                summary['报告期'] = str(latest_data.name)
                
            # 添加盈利能力指标
            profit_metrics = {}
            if key_metrics['roe'] != 'N/A':
                profit_metrics['净资产收益率'] = key_metrics['roe']
            if key_metrics['net_profit_margin'] != 'N/A':
                profit_metrics['销售净利率'] = key_metrics['net_profit_margin']
            if key_metrics['gross_profit_margin'] != 'N/A':
                profit_metrics['销售毛利率'] = key_metrics['gross_profit_margin']
            if profit_metrics:
                summary['盈利能力'] = profit_metrics
            
            # 添加偿债能力指标
            debt_metrics = {}
            if key_metrics['debt_to_asset_ratio'] != 'N/A':
                debt_metrics['资产负债率'] = key_metrics['debt_to_asset_ratio']
            if key_metrics['current_ratio'] != 'N/A':
                debt_metrics['流动比率'] = key_metrics['current_ratio']
            if key_metrics['quick_ratio'] != 'N/A':
                debt_metrics['速动比率'] = key_metrics['quick_ratio']
            if debt_metrics:
                summary['偿债能力'] = debt_metrics
            
            # 添加营运能力指标
            if key_metrics['total_asset_turnover'] != 'N/A':
                summary['营运能力'] = {'总资产周转率': key_metrics['total_asset_turnover']}
            
            # 添加每股指标
            if key_metrics['eps'] != 'N/A':
                summary['每股指标'] = {'摊薄每股收益': key_metrics['eps']}
            
            # 添加成长能力指标
            if key_metrics['net_profit_growth_rate'] != 'N/A':
                summary['成长能力'] = {'净利润增长率': key_metrics['net_profit_growth_rate']}
            
            # 如果未找到任何指标，添加提示
            if len(summary) <= 3:  # 只有基本的三个字段
                summary['财务状况'] = '未能提取到有效财务指标'
            
            # 添加财务评价
            roe_value = pd.to_numeric(key_metrics['roe'], errors='coerce')
            debt_ratio_value = pd.to_numeric(key_metrics['debt_to_asset_ratio'], errors='coerce')
            
            evaluation = []
            if pd.notna(roe_value):
                if roe_value > 15:
                    evaluation.append("盈利能力较强 (ROE > 15%)")
                elif roe_value > 10:
                    evaluation.append("盈利能力良好 (ROE > 10%)")
                elif roe_value > 5:
                    evaluation.append("盈利能力一般 (ROE > 5%)")
                else:
                    evaluation.append("盈利能力较弱 (ROE < 5%)")
                    
            if pd.notna(debt_ratio_value):
                if debt_ratio_value < 40:
                    evaluation.append("财务结构稳健 (资产负债率 < 40%)")
                elif debt_ratio_value < 60:
                    evaluation.append("财务结构一般 (资产负债率 < 60%)")
                else:
                    evaluation.append("负债水平较高 (资产负债率 > 60%)")
                    
            if evaluation:
                summary['财务评价'] = evaluation
            
            logger.info(f"生成 {stock_code} 财务分析摘要完成，包含字段: {list(summary.keys())}")
            return summary
            
        except Exception as e:
            logger.error(f"生成财务分析摘要时出错: {str(e)}")
            # 即使出错也返回基本信息
            summary['出错信息'] = str(e)
            return summary
    
    def get_news_sentiment(self, stock_code: str, stock_name: str = None, days: int = 30) -> pd.DataFrame:
        """
        获取股票相关新闻及其情感分析
        
        参数:
            stock_code (str): 股票代码
            stock_name (str, 可选): 股票名称，如不提供则获取
            days (int): 获取多少天内的新闻
            
        返回:
            pd.DataFrame: 新闻数据，包含标题、内容、日期、情感得分等
        """
        logger.info(f"正在获取股票 {stock_code} 的新闻舆情数据...")
        
        if not stock_name:
            stock_name = self.get_stock_name(stock_code)
        
        try:
            # 通过 AkShare 获取股票新闻
            # 注意：此功能在akshare中可能需要付费或不完全支持，这里提供一个简化实现
            try:
                # 获取股票相关新闻
                stock_news = ak.stock_news_em(symbol=stock_code)
                
                # 如果能够获取到新闻
                if not stock_news.empty:
                    # 标准化列名
                    stock_news.columns = ['title', 'content', 'date', 'url']
                    
                    # 过滤指定天数内的新闻
                    if 'date' in stock_news.columns:
                        # 转换日期格式
                        stock_news['date'] = pd.to_datetime(stock_news['date'])
                        
                        # 过滤最近days天的新闻
                        cutoff_date = datetime.now() - timedelta(days=days)
                        stock_news = stock_news[stock_news['date'] >= cutoff_date]
                    
                    # 添加简单的情感分析（需要实际实现）
                    # 这里使用一个随机值作为示例
                    np.random.seed(42)  # 固定随机种子以便测试
                    stock_news['sentiment'] = np.random.uniform(-1, 1, size=len(stock_news))
                    
                    logger.info(f"成功获取 {len(stock_news)} 条 {stock_code} 相关新闻")
                    return stock_news
                    
            except Exception as e:
                logger.warning(f"通过AkShare获取新闻失败: {str(e)}，使用模拟数据")
            
            # 如果AkShare获取失败，使用模拟数据
            mock_news = pd.DataFrame({
                'title': [f"{stock_name}发布新产品", f"{stock_name}季度业绩超预期", 
                         f"分析师看好{stock_name}未来发展", f"{stock_name}获得政府补贴"],
                'content': ["内容详情...", "内容详情...", "内容详情...", "内容详情..."],
                'date': [datetime.now() - timedelta(days=i) for i in range(4)],
                'url': ["http://example.com"] * 4,
                'sentiment': [0.8, 0.5, 0.3, 0.6]  # 模拟情感分数（-1到1之间，越大越积极）
            })
            
            logger.info(f"使用模拟数据生成 {len(mock_news)} 条 {stock_code} 相关新闻")
            return mock_news
            
        except Exception as e:
            logger.error(f"获取股票新闻舆情数据时出错: {str(e)}")
            return pd.DataFrame()
    
    def generate_news_summary(self, news_data: pd.DataFrame) -> Dict:
        """
        生成新闻舆情摘要
        
        参数:
            news_data (pd.DataFrame): 新闻数据
            
        返回:
            Dict: 新闻舆情摘要
        """
        summary = {}
        
        try:
            if news_data is None or news_data.empty:
                logger.warning("没有可用的新闻数据，无法生成摘要")
                return {'新闻数量': 0, '舆情倾向': '无数据'}
            
            # 新闻数量
            summary['新闻数量'] = len(news_data)
            
            # 平均情感得分
            if 'sentiment' in news_data.columns:
                avg_sentiment = news_data['sentiment'].mean()
                summary['平均情感得分'] = round(avg_sentiment, 2)
                
                # 舆情倾向
                if avg_sentiment > 0.2:
                    summary['舆情倾向'] = '积极'
                elif avg_sentiment < -0.2:
                    summary['舆情倾向'] = '消极'
                else:
                    summary['舆情倾向'] = '中性'
            
            # 最新新闻
            if len(news_data) > 0 and 'title' in news_data.columns:
                summary['最新新闻标题'] = news_data.iloc[0]['title']
            
            # 热点词频统计（简化版）
            if 'title' in news_data.columns:
                # 简单分词（在实际应用中应使用jieba等专业工具）
                titles = ' '.join(news_data['title'].tolist())
                words = titles.split()
                # 热点词统计
                word_count = {}
                for word in words:
                    if len(word) > 1:  # 忽略单字词
                        word_count[word] = word_count.get(word, 0) + 1
                
                # 获取出现频率最高的词
                top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:5]
                summary['近期热点话题'] = [word for word, _ in top_words]
            
            logger.info("新闻舆情摘要生成完成")
            return summary
            
        except Exception as e:
            logger.error(f"生成新闻舆情摘要时出错: {str(e)}")
            return {'错误': str(e)}
            
    def get_stock_info(self, stock_code: str) -> Dict:
        """
        获取股票基本信息
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            Dict: 股票基本信息
        """
        logger.info(f"正在获取股票 {stock_code} 的基本信息...")
        
        try:
            # 通过 AkShare 获取股票信息
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            
            if not stock_info.empty:
                # 转换为字典格式
                info_dict = {}
                for _, row in stock_info.iterrows():
                    info_dict[row['item']] = row['value']
                
                logger.info(f"成功获取 {stock_code} 的基本信息")
                return info_dict
            
            logger.warning(f"未找到股票 {stock_code} 的基本信息")
            return {}
            
        except Exception as e:
            logger.error(f"获取股票基本信息时出错: {str(e)}")
            return {} 