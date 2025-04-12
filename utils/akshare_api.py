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
            end_date = datetime.now()
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
            # 如果未指定开始年份，使用前一年
            if start_year is None:
                start_year = str(datetime.now().year - 1)
                
            # 重试机制
            for i in range(self.retry_count):
                try:
                    # 通过 AkShare 获取财务数据
                    # 可以根据需要选择不同的财务报表，这里使用资产负债表作为示例
                    balance_sheet = ak.stock_financial_report_sina(stock=stock_code, symbol="资产负债表")
                    income_statement = ak.stock_financial_report_sina(stock=stock_code, symbol="利润表")
                    cash_flow = ak.stock_financial_report_sina(stock=stock_code, symbol="现金流量表")
                    
                    # 合并数据
                    combined_data = {
                        '资产负债表': balance_sheet,
                        '利润表': income_statement,
                        '现金流量表': cash_flow
                    }
                    
                    # 可以根据需要进行更多处理，比如数据清理、格式化等
                    
                    logger.info(f"成功获取 {stock_code} 财务数据")
                    return combined_data
                    
                except Exception as e:
                    logger.error(f"获取股票 {stock_code} 财务数据出错 (尝试 {i+1}/{self.retry_count}): {str(e)}")
                    time.sleep(self.retry_interval)
            
            logger.error(f"获取股票 {stock_code} 财务数据失败，已重试 {self.retry_count} 次")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取财务报表数据时出错: {str(e)}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算技术指标
        
        参数:
            df (pd.DataFrame): 股票价格数据
            
        返回:
            Dict[str, pd.Series]: 技术指标字典
        """
        # 确保数据框包含所需的列
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        
        # 转换列名为标准名称
        renamed = False
        if not all(col in df.columns for col in required_columns):
            column_map = {
                'open': 'Open', 
                'high': 'High', 
                'low': 'Low', 
                'close': 'Close', 
                'volume': 'Volume',
                'stock_code': '股票代码',
                'stock_name': '股票名称',
                '开盘': 'Open', 
                '最高': 'High', 
                '最低': 'Low', 
                '收盘': 'Close', 
                '成交量': 'Volume'
            }
            # 创建一个新的列映射，仅包含数据框中存在的列
            mapping = {k: v for k, v in column_map.items() if k in df.columns}
            if mapping:
                df = df.rename(columns=mapping)
                renamed = True
        
        # 检查是否有必要的列
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logger.error(f"缺少计算技术指标所需的列: {missing_columns}")
            return {}
            
        # 计算结果字典
        indicators = {}
        
        try:
            # 排序数据（按日期升序）
            if 'trade_date' in df.columns and df.index.name != 'trade_date':
                df = df.set_index('trade_date').sort_index()
            elif not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("数据框没有日期索引，可能影响指标计算")
            
            # 计算简单移动平均线
            for period in [5, 10, 20, 30, 60]:
                indicators[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            
            # 计算MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            indicators['MACD'] = macd
            indicators['MACD_signal'] = signal
            indicators['MACD_hist'] = macd - signal
            
            # 计算KDJ
            low_min = df['Low'].rolling(window=9).min()
            high_max = df['High'].rolling(window=9).max()
            
            # 避免除零错误
            rsv = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-10))
            
            indicators['KDJ_K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
            indicators['KDJ_D'] = indicators['KDJ_K'].ewm(alpha=1/3, adjust=False).mean()
            indicators['KDJ_J'] = 3 * indicators['KDJ_K'] - 2 * indicators['KDJ_D']
            
            # 计算RSI
            for period in [6, 12, 24]:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                
                # 避免除零错误
                rs = avg_gain / (avg_loss + 1e-10)
                indicators[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # 计算Bollinger Bands
            for period in [20]:
                mid = df['Close'].rolling(window=period).mean()
                std = df['Close'].rolling(window=period).std()
                
                indicators[f'BB_mid_{period}'] = mid
                indicators[f'BB_upper_{period}'] = mid + 2 * std
                indicators[f'BB_lower_{period}'] = mid - 2 * std
            
            # 计算成交量指标
            for period in [5, 10, 20]:
                indicators[f'VWAP_{period}'] = (df['Close'] * df['Volume']).rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
                indicators[f'VOL_MA_{period}'] = df['Volume'].rolling(window=period).mean()
            
            logger.info("技术指标计算完成")
            return indicators
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {str(e)}")
            return {}
    
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
            if all(f'SMA_{period}' in indicators for period in [5, 20]):
                ma5 = indicators['SMA_5'].iloc[-1]
                ma20 = indicators['SMA_20'].iloc[-1]
                
                if ma5 > ma20:
                    tech_signals.append("均线多头排列")
                    if ma5 > ma20 and indicators['SMA_5'].iloc[-2] <= indicators['SMA_20'].iloc[-2]:
                        tech_signals.append("5日均线上穿20日均线，黄金交叉")
                else:
                    tech_signals.append("均线空头排列")
                    if ma5 < ma20 and indicators['SMA_5'].iloc[-2] >= indicators['SMA_20'].iloc[-2]:
                        tech_signals.append("5日均线下穿20日均线，死亡交叉")
            
            # MACD分析
            if all(key in indicators for key in ['MACD', 'MACD_signal', 'MACD_hist']):
                macd = indicators['MACD'].iloc[-1]
                signal = indicators['MACD_signal'].iloc[-1]
                hist = indicators['MACD_hist'].iloc[-1]
                
                if macd > signal:
                    tech_signals.append("MACD金叉")
                else:
                    tech_signals.append("MACD死叉")
                    
                if hist > 0 and indicators['MACD_hist'].iloc[-2] <= 0:
                    tech_signals.append("MACD柱状图由负转正")
                elif hist < 0 and indicators['MACD_hist'].iloc[-2] >= 0:
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
    
    def generate_financial_summary(self, stock_code: str, stock_name: str, financial_reports: Dict) -> Dict:
        """
        生成财务分析摘要
        
        参数:
            stock_code (str): 股票代码
            stock_name (str): 股票名称
            financial_reports (Dict): 财务报表数据
            
        返回:
            Dict: 财务分析摘要
        """
        summary = {
            '股票代码': stock_code,
            '股票名称': stock_name,
            '分析日期': datetime.now().strftime('%Y-%m-%d')
        }
        
        try:
            # 检查是否有有效的财务数据
            if not financial_reports or all(df.empty for df in financial_reports.values() if isinstance(df, pd.DataFrame)):
                logger.warning(f"股票 {stock_code} 没有有效的财务数据")
                return summary
            
            # 处理资产负债表
            if '资产负债表' in financial_reports and not financial_reports['资产负债表'].empty:
                balance_sheet = financial_reports['资产负债表']
                # 提取最新的资产负债表数据
                latest_balance = balance_sheet.iloc[:, :2]  # 假设第一列是项目名，第二列是最新数据
                # 提取关键指标
                balance_summary = {}
                for index, row in latest_balance.iterrows():
                    item_name = row.iloc[0]
                    if item_name in ['总资产', '总负债', '所有者权益(或股东权益)合计', '流动资产合计', '流动负债合计']:
                        balance_summary[item_name] = row.iloc[1]
                        
                summary['资产负债'] = balance_summary
                
            # 处理利润表
            if '利润表' in financial_reports and not financial_reports['利润表'].empty:
                income_statement = financial_reports['利润表']
                # 提取最新的利润表数据
                latest_income = income_statement.iloc[:, :2]  # 假设第一列是项目名，第二列是最新数据
                
                # 提取关键指标
                income_summary = {}
                for index, row in latest_income.iterrows():
                    item_name = row.iloc[0]
                    if item_name in ['营业总收入', '营业利润', '利润总额', '净利润', '基本每股收益']:
                        income_summary[item_name] = row.iloc[1]
                        
                summary['盈利能力'] = income_summary
            
            # 处理现金流量表
            if '现金流量表' in financial_reports and not financial_reports['现金流量表'].empty:
                cash_flow = financial_reports['现金流量表']
                # 提取最新的现金流量表数据
                latest_cash = cash_flow.iloc[:, :2]  # 假设第一列是项目名，第二列是最新数据
                
                # 提取关键指标
                cash_summary = {}
                for index, row in latest_cash.iterrows():
                    item_name = row.iloc[0]
                    if item_name in ['经营活动产生的现金流量净额', '投资活动产生的现金流量净额', '筹资活动产生的现金流量净额']:
                        cash_summary[item_name] = row.iloc[1]
                        
                summary['现金流'] = cash_summary
            
            logger.info(f"生成 {stock_code} 财务分析摘要完成")
            return summary
            
        except Exception as e:
            logger.error(f"生成财务分析摘要时出错xxxx: {str(e)}")
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