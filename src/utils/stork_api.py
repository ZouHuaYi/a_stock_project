import akshare as ak
import baostock as bs
import pandas as pd
import requests
from datetime import datetime
import json
from typing import Optional, Tuple, Dict
import atexit

from src.utils.logger import logger

class StockAPI:
    def __init__(self):
        """初始化数据接口"""
        # 初始化baostock
        bs.login()
        # 注册退出时的清理函数
        atexit.register(self.cleanup)
    
    def cleanup(self):
        """清理资源"""
        try:
            bs.logout()
        except:
            pass
    
    def __del__(self):
        """析构函数"""
        self.cleanup()

    def get_stock_info(self, stock_code: str) -> Optional[Dict]:
        """
        获取股票基本信息
        
        参数:
            stock_code (str): 股票代码（如：600000）
        """
        try:
            # 统一代码格式（移除可能的后缀如.SH）
            pure_codes = stock_code.split('.')
            
            # 通过AKShare获取股票信息
            stock_info = ak.stock_individual_info_em(symbol=f"{pure_codes[0]}")
            if not stock_info.empty:
                return {
                    'stock_code': stock_code, 
                    'exchange': pure_codes[1], # 交易所
                    'stock_name': stock_info.iloc[1, 1], # 名称
                    'industry': stock_info.iloc[6, 1], # 行业
                    'list_date': stock_info.iloc[7, 1], # 上市日期
                    'market_cap': stock_info.iloc[4, 1], # 总市值
                }
        except Exception as e:
            logger.warning(f"通过AKShare获取股票信息失败: {e}")

        return None

    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取股票日线数据
        
        参数:
            stock_code (str): 股票代码（如：600000.SH）
            start_date (str): 开始日期（YYYYMMDD）
            end_date (str): 结束日期（YYYYMMDD）
        """
        try:
            # 尝试通过AKShare获取数据
            pure_code = stock_code.split('.')[0]
            daily_data = ak.stock_zh_a_hist(symbol=pure_code, 
                                          start_date=start_date, 
                                          end_date=end_date,
                                          adjust="qfq")  # 前复权数据
            
            if not daily_data.empty:
                # 创建新的DataFrame而不是修改原始数据
                result_data = pd.DataFrame()
                # 重命名列以统一格式
                result_data['trade_date'] = pd.to_datetime(daily_data['日期']).dt.strftime('%Y%m%d')
                result_data['open'] = daily_data['开盘']
                result_data['close'] = daily_data['收盘']
                result_data['high'] = daily_data['最高']
                result_data['low'] = daily_data['最低']
                result_data['volume'] = daily_data['成交量']
                result_data['amount'] = daily_data['成交额']
                result_data['stock_code'] = stock_code
                
                # 添加其他指标
                result_data['amplitude'] = daily_data['振幅']
                result_data['change_percent'] = daily_data['涨跌幅']
                result_data['change_amount'] = daily_data['涨跌额']
                result_data['turnover_rate'] = daily_data['换手率']
                
                return result_data
            
        except Exception as e:
            logger.warning(f"通过AKShare获取日线数据失败: {e}")
        
        try:
            # 备用方案：通过Baostock获取数据
            bs_code = f"sh.{pure_code}" if stock_code.endswith('.SH') else f"sz.{pure_code}"
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"  # 前复权
            )
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if data_list:
                df = pd.DataFrame(data_list, columns=['trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
                df['stock_code'] = stock_code
                # 确保数据类型正确
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
                
        except Exception as e:
            logger.warning(f"通过Baostock获取日线数据失败: {e}")
        
        return None

    def get_stock_indicators(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取股票技术指标数据
        
        参数:
            stock_code (str): 股票代码（如：600000.SH）
            start_date (str): 开始日期（YYYYMMDD）
            end_date (str): 结束日期（YYYYMMDD）
        """
        try:
            # 获取日线数据
            daily_data = self.get_daily_data(stock_code, start_date, end_date)
            if daily_data is None or daily_data.empty:
                return None
            
            # 创建新的DataFrame来存储指标数据
            df = pd.DataFrame()
            df['stock_code'] = daily_data['stock_code']
            df['trade_date'] = pd.to_datetime(daily_data['trade_date'])
            
            # 确保数据类型正确
            close_series = pd.to_numeric(daily_data['close'], errors='coerce')
            volume_series = pd.to_numeric(daily_data['volume'], errors='coerce')
            
            # 计算MA5, MA10, MA20
            df['ma5'] = close_series.rolling(window=5).mean()
            df['ma10'] = close_series.rolling(window=10).mean()
            df['ma20'] = close_series.rolling(window=20).mean()
            
            # 计算成交量MA5, MA10
            df['volume_ma5'] = volume_series.rolling(window=5).mean()
            df['volume_ma10'] = volume_series.rolling(window=10).mean()
            
            # 计算MACD
            exp1 = close_series.ewm(span=12, adjust=False).mean()
            exp2 = close_series.ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['signal']
            
            # 计算RSI
            delta = close_series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            return df
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return None

    def get_stock_list(self):
        """获取A股股票列表"""
        try:
            # 使用akshare获取A股列表
            stock_list = ak.stock_info_a_code_name()
            
            # 检查返回的数据结构
            logger.debug(f"原始数据列名: {stock_list.columns.tolist()}")
            
            # 重命名列（根据实际返回的列名调整）
            column_mapping = {
                'code': 'stock_code',
                'name': 'stock_name'
            }
            
            # 重命名列
            stock_list = stock_list.rename(columns=column_mapping)
            
            # 添加交易所信息
            stock_list['exchange'] = stock_list['stock_code'].apply(
                lambda x: 'SH' if str(x).startswith('6') else 'SZ'
            )
            
            # 格式化股票代码（添加.SH或.SZ后缀）
            stock_list['stock_code'] = stock_list.apply(
                lambda x: f"{str(x['stock_code']).zfill(6)}.{x['exchange']}", axis=1
            )

            # 只保留 0，60，300开头的股票
            stock_list = stock_list[stock_list['stock_code'].str.startswith(('0', '60', '300'))]
            
            logger.info(f"成功获取{len(stock_list)}只股票的基本信息")
            return stock_list
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return None

        """获取股票技术指标数据"""
        try:
            # 首先获取日线数据
            daily_data = self.get_stock_daily(stock_code, start_date, end_date)
            
            if daily_data is None or daily_data.empty:
                return None
            
            # 计算技术指标
            indicators = pd.DataFrame()
            indicators['trade_date'] = daily_data['trade_date']
            indicators['stock_code'] = daily_data['stock_code']
            
            # 计算MA5, MA10, MA20
            for period in [5, 10, 20]:
                indicators[f'ma{period}'] = daily_data['close'].rolling(window=period).mean()
            
            # 计算成交量MA5, MA10
            for period in [5, 10]:
                indicators[f'vol_ma{period}'] = daily_data['volume'].rolling(window=period).mean()
            
            # 删除包含NaN的行
            indicators = indicators.dropna()
            
            logger.info(f"成功计算股票{stock_code}的{len(indicators)}条技术指标数据")
            return indicators
            
        except Exception as e:
            logger.error(f"计算股票{stock_code}技术指标失败: {e}")
            return None
# 创建全局实例
stock_api = StockAPI() 