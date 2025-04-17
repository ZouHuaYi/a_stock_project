# -*- coding: utf-8 -*-
"""股票数据获取和更新模块"""

import akshare as ak
import pandas as pd
import numpy as np
import time
import os
import concurrent.futures
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# 导入配置和日志
from config import DATA_CONFIG, TABLE_SCHEMAS
from data.db_manager import DatabaseManager
from utils.logger import get_logger
from utils.task_runner import TaskRunner

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
        
        # 从数据库获取数据
        try:
            # 确保表存在
            if not self.db_manager.table_exists('stock_basic'):
                self.db_manager.execute_ddl(TABLE_SCHEMAS['stock_basic'])
            # 检查数据库中是否已有股票基本信息
            existing_stocks = self.db_manager.fetch_one("SELECT COUNT(*) as count FROM stock_basic")
            if existing_stocks and existing_stocks.get('count', 0) > 0:
                logger.info("从数据库中获取股票基本信息")
                stocks = self.db_manager.read_sql("SELECT stock_code, stock_name, market_cap, exchange, industry, list_date FROM stock_basic")
                if not stocks.empty:
                    return stocks
                    
            # 如果数据库中没有数据，从AKShare获取
            logger.info("从AKShare获取股票列表")
            try:
                stock_list = ak.stock_info_a_code_name()
                # 重命名列（根据实际返回的列名调整）
                column_mapping = {
                    'code': 'stock_code',
                    'name': 'stock_name'
                }
                stock_list = stock_list.rename(columns=column_mapping)

                # 只保留 0，60，300开头的股票
                stock_list = stock_list[stock_list['stock_code'].str.startswith(('0', '60', '300'))]

                stock_list['exchange'] = stock_list['stock_code'].apply(
                    lambda x: 'SH' if x.startswith('6') else 'SZ'
                )

                # 设置默认行业
                stock_list['industry'] = '未知'
                stock_list['list_date'] = datetime.now().strftime('%Y-%m-%d')
                
                # 向数据库中插入数据
                self.db_manager.to_sql(stock_list.copy(), 'stock_basic', if_exists='append')

                logger.info(f"数据库中股票数量：{len(stock_list)}")
                return stock_list
            except Exception as e:
                logger.error(f"获取股票列表时出错, {str(e)}")
                
        except Exception as e:
            logger.error(f"获取股票列表失败: {str(e)}")
            
        # 如果所有尝试都失败，返回空DataFrame
        return pd.DataFrame()
        
    def _update_stock_basic(self, stocks: pd.DataFrame) -> bool:
        """
        更新股票基本信息
        
        参数:
            stocks (pd.DataFrame): 股票信息数据框
            
        返回:
            bool: 是否成功更新
        """
        logger.info(f"更新股票 market_cap industry list_date 字段, {len(stocks)} 只股票")
        try:
            error_update_list = []
            # 批量处理更新数据库
            batch_size = 50  # 每批处理的股票数量
            total_stocks = len(stocks)
            
            # 最多10个线程
            max_workers = min(10, (total_stocks // batch_size) + 1)
            
            def process_batch(batch_stocks):
                """处理一批股票的函数"""
                batch_data = []
                local_error_list = []
                
                for index, row in batch_stocks.iterrows():
                    stock_code = row['stock_code']
                    try:
                        # 获取股票基本信息
                        stock_info = ak.stock_individual_info_em(symbol=f"{stock_code}", timeout=6000)
                        # 这里数据如果边了需要重新整理
                        industry = stock_info.iloc[6, 1]
                        market_cap = stock_info.iloc[4, 1]    
                        list_date = stock_info.iloc[7, 1]
                        # market_cap 是 decimal 类型，需要转换为 float 类型
                        market_cap = float(market_cap)
                        
                        # 将数据添加到批次列表
                        batch_data.append((industry, market_cap, list_date, stock_code))
                        
                    except Exception as e:
                        logger.error(f"获取股票 {stock_code} 基本信息时出错: {str(e)}")
                        local_error_list.append(stock_code)
                
                # 如果批次有数据，则执行批量更新
                update_results = {}
                if batch_data:
                    local_db_manager = DatabaseManager()  # 为每个线程创建单独的数据库连接
                    try:
                        # 使用参数化查询进行批量更新
                        update_sql = "UPDATE stock_basic SET industry = %s, market_cap = %s, list_date = %s WHERE stock_code = %s"
                        success = local_db_manager.executemany_and_commit(update_sql, batch_data)
                        if success:
                            logger.info(f"批量更新 {len(batch_data)} 只股票的基本信息成功")
                        else:
                            # 如果批量更新失败，尝试单个更新
                            for industry, market_cap, list_date, stock_code in batch_data:
                                try:
                                    single_update_sql = "UPDATE stock_basic SET industry = %s, market_cap = %s, list_date = %s WHERE stock_code = %s"
                                    single_success = local_db_manager.execute_and_commit(single_update_sql, (industry, market_cap, list_date, stock_code))
                                    if not single_success:
                                        local_error_list.append(stock_code)
                                except Exception as se:
                                    logger.error(f"单独更新股票 {stock_code} 基本信息时出错: {str(se)}")
                                    local_error_list.append(stock_code)
                    except Exception as e:
                        logger.error(f"批量更新股票基本信息时出错: {str(e)}")
                        # 批量更新失败，记录所有股票为错误
                        local_error_list.extend([data[3] for data in batch_data if data[3] not in local_error_list])
                    finally:
                        local_db_manager.close()
                
                update_results['errors'] = local_error_list
                update_results['processed'] = len(batch_data)
                return update_results
            
            # 将股票列表分批
            batches = []
            for batch_start in range(0, total_stocks, batch_size):
                batch_end = min(batch_start + batch_size, total_stocks)
                batches.append(stocks.iloc[batch_start:batch_end])
            
            # 使用TaskRunner处理批次任务
            processed_count = 0
            with TaskRunner(use_threads=True, max_workers=max_workers) as task_runner:
                # 提交所有批次任务
                task_args_list = [(batch,) for batch in batches]
                results = task_runner.run_tasks(process_batch, task_args_list)
                
                # 处理结果
                for i, result in enumerate(results):
                    if result:
                        processed_count += result['processed']
                        if result['errors']:
                            error_update_list.extend(result['errors'])
                            logger.warning(f"批次 {i+1} 中有 {len(result['errors'])} 只股票更新失败")
                    else:
                        logger.error(f"批次 {i+1} 处理失败或超时")
            
            if error_update_list:
                logger.warning(f"更新失败的股票列表: {error_update_list}")
                # 再一次处理 error_update_list 中的股票
                self._update_stock_basic(pd.DataFrame(error_update_list, columns=['stock_code']))
            
            logger.info(f"共处理 {processed_count} 只股票，其中 {len(error_update_list)} 只更新失败")
            return True
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
        # 首先获取已有的数据，用于后续只获取缺失的数据
        existing_data = pd.DataFrame()
        try:
            start_date_fmt = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
            end_date_fmt = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
            
            sql = f"""
                SELECT * FROM stock_daily 
                WHERE stock_code = '{stock_code}' 
                AND trade_date BETWEEN '{start_date_fmt}' AND '{end_date_fmt}'
            """
            existing_data = self.db_manager.read_sql(sql)
            if not existing_data.empty:
                logger.info(f"从数据库获取到股票 {stock_code} 的 {len(existing_data)} 条日线数据")
                # 如果已有的数据覆盖了整个日期范围，直接返回
                if self._is_date_range_covered(existing_data, start_date_fmt, end_date_fmt):
                    logger.info(f"股票 {stock_code} 在日期范围内的数据已完整，无需更新")
                    return existing_data
                
                logger.info(f"股票 {stock_code} 日期范围内的数据不完整，将获取缺失数据")
        except Exception as db_e:
            logger.warning(f"从数据库获取股票 {stock_code} 日线数据时出错: {str(db_e)}")
        
        # 从API获取数据
        max_retries = 3
        retry_delay = 2
        for retry in range(max_retries):
            try:
                # 使用akshare获取日线数据
                logger.info(f"从AKShare获取股票 {stock_code} 日线数据")
                
                # 如果已有数据，则只获取缺失的日期范围
                if not existing_data.empty:
                    missing_ranges = self._get_missing_date_ranges(existing_data, start_date_fmt, end_date_fmt)
                    if not missing_ranges:
                        logger.info(f"股票 {stock_code} 没有缺失的日期范围，无需从API获取数据")
                        return existing_data
                    
                    # 合并多个时间段的数据
                    all_new_data = pd.DataFrame()
                    for missing_start, missing_end in missing_ranges:
                        missing_start_str = missing_start.strftime('%Y%m%d')
                        missing_end_str = missing_end.strftime('%Y%m%d')
                        logger.info(f"获取股票 {stock_code} 缺失日期 {missing_start_str} 至 {missing_end_str} 的数据")
                        
                        new_data = ak.stock_zh_a_hist(
                            symbol=stock_code,
                            period="daily",
                            start_date=missing_start_str,
                            end_date=missing_end_str,
                            adjust="qfq"  # 前复权
                        )
                        
                        if not new_data.empty:
                            all_new_data = pd.concat([all_new_data, new_data])
                else:
                    # 如果没有已有数据，则获取整个日期范围
                    pure_code = stock_code.split('.')[0]
                    all_new_data = ak.stock_zh_a_hist(
                        symbol=pure_code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"  # 前复权
                    )
                
                # 检查数据
                if all_new_data.empty:
                    logger.warning(f"未获取到股票 {stock_code} 的日线数据")
                    break
                    
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
                # 这个方法有问题，我只想保留 column_map 中的列名
                all_new_data = all_new_data[column_map.keys()]
                all_new_data.rename(columns=column_map, inplace=True)
                
                # 添加股票代码
                all_new_data['stock_code'] = stock_code
                
                # 转换日期格式
                all_new_data['trade_date'] = pd.to_datetime(all_new_data['trade_date'])
                
                # 确保数据类型正确
                for col in ['open', 'close', 'high', 'low', 'volume', 'amount']:
                    if col in all_new_data.columns:
                        all_new_data[col] = all_new_data[col].astype(float)
                        
                # 合并已有数据和新数据
                if not existing_data.empty:
                    # 确保一致的日期格式
                    if 'trade_date' in existing_data.columns:
                        existing_data['trade_date'] = pd.to_datetime(existing_data['trade_date'])
                    
                    # 删除重复的日期（以新数据为准）
                    merged_data = pd.concat([existing_data, all_new_data])
                    merged_data = merged_data.drop_duplicates(subset=['stock_code', 'trade_date'], keep='last')
                    return merged_data
                else:
                    return all_new_data
                
            except Exception as e:
                logger.error(f"获取股票 {stock_code} 日线数据时出错: {str(e)}")
                if retry < max_retries - 1:
                    logger.info(f"等待 {retry_delay} 秒后重试 ({retry+1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
        
        # 所有方法都失败，返回已有数据或空DataFrame
        return existing_data if not existing_data.empty else pd.DataFrame()
    
    def _is_date_range_covered(self, df: pd.DataFrame, start_date: str, end_date: str) -> bool:
        """
        检查数据框是否完整覆盖了指定的日期范围
        
        参数:
            df (pd.DataFrame): 数据框
            start_date (str): 开始日期，格式：YYYY-MM-DD
            end_date (str): 结束日期，格式：YYYY-MM-DD
            
        返回:
            bool: 是否完整覆盖
        """
        if df.empty:
            return False
        
        # 确保日期列是datetime类型
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 转换日期字符串为datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 获取交易日历（周一至周五）
        all_days = pd.date_range(start=start_dt, end=end_dt, freq='D')
        trading_days = [day for day in all_days if day.weekday() < 5]  # 排除周末
        
        # 检查每个交易日是否都有数据
        # 注意：这里可能需要考虑节假日，但简化处理
        existing_dates = set(df['trade_date'].dt.date)
        missing_dates = [day for day in trading_days if day.date() not in existing_dates]
        
        # 如果缺失日期数量少于总日期的5%，认为基本完整（容忍节假日）
        tolerance_ratio = 0.05
        if len(missing_dates) <= len(trading_days) * tolerance_ratio:
            return True
        
        return False
    
    def _get_missing_date_ranges(self, df: pd.DataFrame, start_date: str, end_date: str) -> List[Tuple[datetime, datetime]]:
        """
        获取数据框缺失的日期范围
        
        参数:
            df (pd.DataFrame): 数据框
            start_date (str): 开始日期，格式：YYYY-MM-DD
            end_date (str): 结束日期，格式：YYYY-MM-DD
            
        返回:
            List[Tuple[datetime, datetime]]: 缺失的日期范围列表，每个元素为(开始日期,结束日期)
        """
        if df.empty:
            # 如果数据为空，返回整个日期范围
            return [(pd.to_datetime(start_date), pd.to_datetime(end_date))]
        
        # 确保日期列是datetime类型
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 转换日期字符串为datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 获取交易日历（周一至周五）
        all_days = pd.date_range(start=start_dt, end=end_dt, freq='D')
        trading_days = [day for day in all_days if day.weekday() < 5]  # 排除周末
        
        # 获取已有的日期
        existing_dates = set(df['trade_date'].dt.date)
        
        # 找出缺失的日期
        missing_dates = [day for day in trading_days if day.date() not in existing_dates]
        
        # 如果没有缺失日期，返回空列表
        if not missing_dates:
            return []
        
        # 将缺失日期组合成连续的日期范围
        missing_ranges = []
        range_start = missing_dates[0]
        prev_date = missing_dates[0]
        
        for i in range(1, len(missing_dates)):
            curr_date = missing_dates[i]
            # 如果当前日期与前一个日期不连续，结束当前范围并开始新范围
            if (curr_date - prev_date).days > 3:  # 允许2-3天的间隔（可能是周末）
                missing_ranges.append((range_start, prev_date))
                range_start = curr_date
            prev_date = curr_date
        
        # 添加最后一个范围
        missing_ranges.append((range_start, prev_date))
        
        return missing_ranges
    
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
                        # 更新日线数据，使用append而不是replace
                        self.db_manager.to_sql(daily_data, 'stock_daily', if_exists='append', index=False)
                        
                        # 删除重复数据
                        sql = """
                            DELETE t1 FROM stock_daily t1
                            INNER JOIN (
                                SELECT stock_code, trade_date, MAX(id) as max_id
                                FROM stock_daily
                                GROUP BY stock_code, trade_date
                                HAVING COUNT(*) > 1
                            ) t2 ON t1.stock_code = t2.stock_code AND t1.trade_date = t2.trade_date
                            WHERE t1.id < t2.max_id
                        """
                        self.db_manager.execute_and_commit(sql)
                        
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
        try:
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
        except Exception as e:
            logger.error(f"记录同步日志失败: {str(e)}")
            return False
        
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
    
    def init_stock_basic(self) -> bool:
        """
        初始化股票基本信息
        
        返回:
            bool: 是否成功初始化
        """
        logger.info("开始初始化股票基本信息")
        start_time = datetime.now()
        try:
             # 确保表存在
            if not self._ensure_tables_exist():
                raise Exception("创建必要的表失败")
            
            # 获取上次同步时间
            last_sync = self._get_last_sync_date('BASIC_DATA')
            # 计算增量更新的日期范围
            end_date = datetime.now()
            
            if last_sync:
                # 从上次同步的第二天开始
                start_date = last_sync + timedelta(days=1)
                if start_date >= end_date:
                    logger.info("数据已是最新，无需更新")
                    return True
           
            # 获取股票列表
            stock_list = self._fetch_stock_list()
            if stock_list.empty:
                raise Exception("获取股票列表失败")
            
            # 更新股票基本信息
            if not self._update_stock_basic(stock_list):
                logger.warning("更新股票基本信息失败")
                return False

            # 记录同步日志
            try:
                self._log_sync(
                    'BASIC_DATA', 
                    start_time, 
                    datetime.now(), 
                    'SUCCESS', 
                    stock_list.shape[0])
            except Exception as log_e:
                logger.error(f"记录成功日志时出错: {str(log_e)}")

            return True
        except Exception as e:
            logger.error(f"初始化股票基本信息时出错: {str(e)}")
            return False
        finally:
            # 关闭数据库连接
            self.db_manager.close()
        
    def init_daily_data(self) -> bool:
        """
        初始化股票日线数据
        
        返回:
            bool: 是否成功更新
        """
        logger.info("开始初始化股票日线数据")
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
                
            # 获取所有股票代码
            stock_codes = stock_list['stock_code'].tolist()
            
            # 测试模式限制股票数量
            if 'test_stock_limit' in DATA_CONFIG and DATA_CONFIG['test_stock_limit'] > 0:
                stock_codes = stock_codes[:DATA_CONFIG['test_stock_limit']]
                logger.info(f"测试模式: 限制处理 {len(stock_codes)} 只股票")
                
            # 更新股票日线数据
            updated_count = self._update_stock_daily(stock_codes, start_date_str, end_date_str)
            
            # 记录同步日志
            try:
                self._log_sync(
                    'DAILY_DATA', 
                    start_time, 
                    datetime.now(), 
                    'SUCCESS', 
                    updated_count
                )
            except Exception as log_e:
                logger.error(f"记录成功日志时出错: {str(log_e)}")
            
            logger.info(f"增量更新完成，共更新 {updated_count} 条记录")
            return True
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"增量更新失败: {error_message}")
            
            # 记录同步失败日志
            try:
                self._log_sync(
                    'DAILY_DATA', 
                    start_time, 
                    datetime.now(), 
                    'FAILED', 
                    0, 
                    error_message
                )
            except Exception as log_e:
                logger.error(f"记录失败日志时出错: {str(log_e)}")
            
            return False
        finally:
            # 关闭数据库连接
            self.db_manager.close()

    def update_stock_self_data(self, start_date: str, end_date: str) -> bool:
        """
        更新股票指定范围内的日线数据
        
        参数:
            stock_code (str): 股票代码
            start_date (str): 开始日期，格式：YYYYMMDD
            end_date (str): 结束日期，格式：YYYYMMDD
            
        返回:
            bool: 是否成功更新
        """
        logger.info("开始更新股票指定范围内的日线数据")

        try:
            # 确保表存在
            if not self._ensure_tables_exist():
                raise Exception("创建必要的表失败")
            
            if start_date is None or end_date is None:
                raise Exception("开始日期和结束日期不能为空")

            # 获取股票列表
            stock_list = self._fetch_stock_list()
            if stock_list.empty:
                raise Exception("获取股票列表失败")

            # 获取所有股票代码
            stock_codes = stock_list['stock_code'].tolist()
            
            # 测试模式限制股票数量
            if 'test_stock_limit' in DATA_CONFIG and DATA_CONFIG['test_stock_limit'] > 0:
                stock_codes = stock_codes[:DATA_CONFIG['test_stock_limit']]
                logger.info(f"测试模式: 限制处理 {len(stock_codes)} 只股票")

            # 更新股票日线数据
            updated_count = self._update_stock_daily(stock_codes, start_date, end_date)
            logger.info(f"更新股票指定范围内的日线数据完成，共更新 {updated_count} 条记录")
            return True
        except Exception as e:
            logger.error(f"更新股票指定范围内的日线数据时出错: {str(e)}")
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
        
        try:
            if not self.init_stock_basic():
                raise Exception("初始化股票基本信息失败")
            
            if not self.init_daily_data():
                raise Exception("初始化股票日线数据失败")
            
            return True  
        except Exception as e:
            logger.error(f"全量更新失败: {str(e)}")
            return False
        finally:
            # 关闭数据库连接
            self.db_manager.close() 