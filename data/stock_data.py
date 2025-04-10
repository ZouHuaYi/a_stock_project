# -*- coding: utf-8 -*-
"""股票数据获取和更新模块"""

import akshare as ak
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# 导入配置和日志
from config import DATA_CONFIG, TABLE_SCHEMAS
from data.db_manager import DatabaseManager
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger(__name__)

class StockDataUpdater:
    """股票数据更新类，负责获取和更新股票数据"""
    
    def __init__(self):
        """初始化股票数据更新器"""
        self.db_manager = DatabaseManager()
        self.batch_size = DATA_CONFIG['batch_size']
        self.sleep_time = DATA_CONFIG['sleep_time']
        
    def _ensure_tables_exist(self) -> bool:
        """
        确保必要的数据表存在
        
        返回:
            bool: 所有表是否存在
        """
        for table_name, schema in TABLE_SCHEMAS.items():
            if not self.db_manager.table_exists(table_name):
                logger.info(f"创建表 {table_name}")
                if not self.db_manager.execute_ddl(schema):
                    logger.error(f"创建表 {table_name} 失败")
                    return False
        return True
        
    def _fetch_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        返回:
            pd.DataFrame: 股票列表数据框
        """
        logger.info("获取A股股票列表")
        try:
            # 使用akshare获取股票列表
            stock_list = ak.stock_zh_a_spot_em()
            
            # 整理数据
            result = stock_list[['代码', '名称', '总市值']].copy()
            result.columns = ['stock_code', 'stock_name', 'market_cap']
            
            # 添加交易所和行业信息
            result['exchange'] = result['stock_code'].apply(
                lambda x: 'SH' if x.startswith('6') else 'SZ'
            )
            
            # 设置默认行业
            result['industry'] = '未知'
            result['list_date'] = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"获取到 {len(result)} 只股票")
            return result
        except Exception as e:
            logger.error(f"获取股票列表时出错: {str(e)}")
            return pd.DataFrame()
            
    def _update_stock_basic(self, stocks: pd.DataFrame) -> bool:
        """
        更新股票基本信息
        
        参数:
            stocks (pd.DataFrame): 股票信息数据框
            
        返回:
            bool: 是否成功更新
        """
        logger.info("更新股票基本信息")
        
        try:
            # 如果表不存在，创建表
            if not self.db_manager.table_exists('stock_basic'):
                self.db_manager.execute_ddl(TABLE_SCHEMAS['stock_basic'])
            
            # 更新股票基本信息
            return self.db_manager.to_sql(stocks, 'stock_basic', if_exists='replace')
        except Exception as e:
            logger.error(f"更新股票基本信息时出错: {str(e)}")
            return False
            
    def _fetch_stock_daily(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取单只股票的日线数据
        
        参数:
            stock_code (str): 股票代码
            start_date (str): 开始日期，格式：YYYYMMDD
            end_date (str): 结束日期，格式：YYYYMMDD
            
        返回:
            pd.DataFrame: 股票日线数据框
        """
        try:
            # 使用akshare获取日线数据
            daily_data = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            # 检查数据
            if daily_data.empty:
                logger.warning(f"未获取到股票 {stock_code} 的日线数据")
                return pd.DataFrame()
                
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
            daily_data.rename(columns=column_map, inplace=True)
            
            # 添加股票代码
            daily_data['stock_code'] = stock_code
            
            # 转换日期格式
            daily_data['trade_date'] = pd.to_datetime(daily_data['trade_date'])
            
            # 确保数据类型正确
            for col in ['open', 'close', 'high', 'low', 'volume', 'amount']:
                if col in daily_data.columns:
                    daily_data[col] = daily_data[col].astype(float)
                    
            return daily_data
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 日线数据时出错: {str(e)}")
            return pd.DataFrame()
            
    def _calculate_indicators(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        参数:
            daily_data (pd.DataFrame): 日线数据
            
        返回:
            pd.DataFrame: 技术指标数据框
        """
        if daily_data.empty:
            return pd.DataFrame()
            
        # 获取必要的列
        df = daily_data[['stock_code', 'trade_date', 'close', 'volume']].copy()
        
        # 排序数据
        df.sort_values('trade_date', inplace=True)
        
        # 计算均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        # 计算成交量均线
        df['vol_ma5'] = df['volume'].rolling(window=5).mean()
        df['vol_ma10'] = df['volume'].rolling(window=10).mean()
        
        # 选择需要的列
        result = df[['stock_code', 'trade_date', 'ma5', 'ma10', 'ma20', 'vol_ma5', 'vol_ma10']]
        
        return result
        
    def _update_stock_daily(self, stock_codes: List[str], start_date: str, end_date: str) -> int:
        """
        更新股票日线数据
        
        参数:
            stock_codes (List[str]): 股票代码列表
            start_date (str): 开始日期，格式：YYYYMMDD
            end_date (str): 结束日期，格式：YYYYMMDD
            
        返回:
            int: 更新的记录数
        """
        # 确保表存在
        if not self.db_manager.table_exists('stock_daily'):
            self.db_manager.execute_ddl(TABLE_SCHEMAS['stock_daily'])
            
        if not self.db_manager.table_exists('stock_daily_indicator'):
            self.db_manager.execute_ddl(TABLE_SCHEMAS['stock_daily_indicator'])
            
        updated_count = 0
        
        # 分批处理
        for i in range(0, len(stock_codes), self.batch_size):
            batch = stock_codes[i:i+self.batch_size]
            logger.info(f"处理第 {i//self.batch_size + 1} 批，共 {len(batch)} 只股票")
            
            for stock_code in batch:
                try:
                    # 获取日线数据
                    daily_data = self._fetch_stock_daily(stock_code, start_date, end_date)
                    
                    if not daily_data.empty:
                        # 更新日线数据
                        self.db_manager.to_sql(daily_data, 'stock_daily', if_exists='replace')
                        
                        # 计算并更新技术指标
                        indicators = self._calculate_indicators(daily_data)
                        if not indicators.empty:
                            self.db_manager.to_sql(indicators, 'stock_daily_indicator', if_exists='replace')
                        
                        updated_count += len(daily_data)
                        logger.debug(f"更新股票 {stock_code} 数据成功，{len(daily_data)} 条记录")
                except Exception as e:
                    logger.error(f"更新股票 {stock_code} 数据时出错: {str(e)}")
                    
            # 避免频繁请求
            if i + self.batch_size < len(stock_codes):
                logger.info(f"休眠 {self.sleep_time} 秒后继续")
                time.sleep(self.sleep_time)
                
        return updated_count
        
    def _log_sync(self, sync_type: str, start_time: datetime, end_time: datetime, 
                 status: str, records_count: int, error_message: str = None) -> bool:
        """
        记录数据同步日志
        
        参数:
            sync_type (str): 同步类型
            start_time (datetime): 开始时间
            end_time (datetime): 结束时间
            status (str): 状态
            records_count (int): 记录数
            error_message (str, 可选): 错误信息
            
        返回:
            bool: 是否成功记录
        """
        sql = """
            INSERT INTO data_sync_log 
            (sync_type, start_time, end_time, status, records_count, error_message)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = (
            sync_type,
            start_time.strftime('%Y-%m-%d %H:%M:%S'),
            end_time.strftime('%Y-%m-%d %H:%M:%S'),
            status,
            records_count,
            error_message
        )
        
        return self.db_manager.execute_and_commit(sql, params)
        
    def _get_last_sync_date(self, sync_type: str) -> Optional[datetime]:
        """
        获取最后同步日期
        
        参数:
            sync_type (str): 同步类型
            
        返回:
            Optional[datetime]: 最后同步日期
        """
        sql = """
            SELECT MAX(end_time) as last_sync
            FROM data_sync_log
            WHERE sync_type = %s
            AND status = 'SUCCESS'
        """
        result = self.db_manager.fetch_one(sql, (sync_type,))
        
        if result and result['last_sync']:
            return result['last_sync']
        return None
        
    def incremental_update(self) -> bool:
        """
        增量更新股票数据
        
        返回:
            bool: 是否成功更新
        """
        logger.info("开始增量更新股票数据")
        start_time = datetime.now()
        
        try:
            # 确保表存在
            if not self._ensure_tables_exist():
                raise Exception("创建必要的表失败")
                
            # 获取上次同步时间
            last_sync = self._get_last_sync_date('DAILY_DATA')
            
            # 计算增量更新的日期范围
            end_date = datetime.now()
            
            if last_sync:
                # 从上次同步的第二天开始
                start_date = last_sync + timedelta(days=1)
                if start_date >= end_date:
                    logger.info("数据已是最新，无需更新")
                    self._log_sync(
                        'DAILY_DATA', 
                        start_time, 
                        datetime.now(), 
                        'SUCCESS', 
                        0
                    )
                    return True
            else:
                # 如果没有同步记录，使用默认的初始天数
                start_date = end_date - timedelta(days=DATA_CONFIG['initial_days'])
            
            # 格式化日期
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = end_date.strftime('%Y%m%d')
            
            logger.info(f"增量更新日期范围: {start_date_str} 至 {end_date_str}")
            
            # 获取股票列表
            stock_list = self._fetch_stock_list()
            if stock_list.empty:
                raise Exception("获取股票列表失败")
                
            # 更新股票基本信息
            if not self._update_stock_basic(stock_list):
                logger.warning("更新股票基本信息失败")
                
            # 获取所有股票代码
            stock_codes = stock_list['stock_code'].tolist()
            
            # 测试模式限制股票数量
            if 'test_stock_limit' in DATA_CONFIG and DATA_CONFIG['test_stock_limit'] > 0:
                stock_codes = stock_codes[:DATA_CONFIG['test_stock_limit']]
                logger.info(f"测试模式: 限制处理 {len(stock_codes)} 只股票")
                
            # 更新股票日线数据
            updated_count = self._update_stock_daily(stock_codes, start_date_str, end_date_str)
            
            # 记录同步日志
            self._log_sync(
                'DAILY_DATA', 
                start_time, 
                datetime.now(), 
                'SUCCESS', 
                updated_count
            )
            
            logger.info(f"增量更新完成，共更新 {updated_count} 条记录")
            return True
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"增量更新失败: {error_message}")
            
            # 记录同步失败日志
            self._log_sync(
                'DAILY_DATA', 
                start_time, 
                datetime.now(), 
                'FAILED', 
                0, 
                error_message
            )
            
            return False
        finally:
            # 关闭数据库连接
            self.db_manager.close()
            
    def full_update(self) -> bool:
        """
        全量更新股票数据
        
        返回:
            bool: 是否成功更新
        """
        logger.info("开始全量更新股票数据")
        start_time = datetime.now()
        
        try:
            # 确保表存在
            if not self._ensure_tables_exist():
                raise Exception("创建必要的表失败")
                
            # 计算全量更新的日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=DATA_CONFIG['initial_days'])
            
            # 格式化日期
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = end_date.strftime('%Y%m%d')
            
            logger.info(f"全量更新日期范围: {start_date_str} 至 {end_date_str}")
            
            # 获取股票列表
            stock_list = self._fetch_stock_list()
            if stock_list.empty:
                raise Exception("获取股票列表失败")
                
            # 更新股票基本信息
            if not self._update_stock_basic(stock_list):
                logger.warning("更新股票基本信息失败")
                
            # 获取所有股票代码
            stock_codes = stock_list['stock_code'].tolist()
            
            # 测试模式限制股票数量
            if 'test_stock_limit' in DATA_CONFIG and DATA_CONFIG['test_stock_limit'] > 0:
                stock_codes = stock_codes[:DATA_CONFIG['test_stock_limit']]
                logger.info(f"测试模式: 限制处理 {len(stock_codes)} 只股票")
                
            # 更新股票日线数据
            updated_count = self._update_stock_daily(stock_codes, start_date_str, end_date_str)
            
            # 记录同步日志
            self._log_sync(
                'FULL_DATA', 
                start_time, 
                datetime.now(), 
                'SUCCESS', 
                updated_count
            )
            
            logger.info(f"全量更新完成，共更新 {updated_count} 条记录")
            return True
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"全量更新失败: {error_message}")
            
            # 记录同步失败日志
            self._log_sync(
                'FULL_DATA', 
                start_time, 
                datetime.now(), 
                'FAILED', 
                0, 
                error_message
            )
            
            return False
        finally:
            # 关闭数据库连接
            self.db_manager.close() 