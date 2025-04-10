# -*- coding: utf-8 -*-
"""数据库管理器模块"""

import pymysql
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime
import os
from typing import List, Dict, Optional, Union, Tuple, Any

# 导入配置和日志
from config import DB_CONFIG, TABLE_SCHEMAS
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger(__name__)

class DatabaseManager:
    """数据库管理器，负责数据库连接和操作"""
    
    def __init__(self):
        """初始化数据库管理器"""
        self.engine = None
        self.connection = None
        self.connect()
    
    def connect(self) -> bool:
        """
        创建数据库连接
        
        返回:
            bool: 是否成功连接
        """
        try:
            # 创建数据库引擎
            self.engine = create_engine(DB_CONFIG['connection_string'])
            
            # 测试连接
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("数据库连接创建成功")
            return True
            
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False
    
    def close(self) -> None:
        """关闭数据库连接"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")
    
    def execute_and_commit(self, sql: str, params: tuple = None) -> bool:
        """
        执行SQL语句并提交
        
        参数:
            sql (str): SQL语句
            params (tuple, 可选): SQL参数
            
        返回:
            bool: 是否成功执行
        """
        try:
            with self.engine.connect() as connection:
                if params:
                    connection.execute(text(sql), params)
                else:
                    connection.execute(text(sql))
                connection.commit()
            return True
        except Exception as e:
            logger.error(f"执行SQL失败: {e}")
            return False
    
    def fetch_one(self, sql: str, params: tuple = None) -> Optional[Dict]:
        """
        执行查询并返回单条记录
        
        参数:
            sql (str): SQL查询语句
            params (tuple, 可选): SQL参数
            
        返回:
            Optional[Dict]: 查询结果，未找到时返回None
        """
        try:
            with self.engine.connect() as connection:
                if params:
                    result = connection.execute(text(sql), params).fetchone()
                else:
                    result = connection.execute(text(sql)).fetchone()
                
                if result:
                    # 转换为字典
                    keys = result.keys()
                    return {key: result[key] for key in keys}
                return None
        except Exception as e:
            logger.error(f"查询单条记录失败: {e}")
            return None
    
    def fetch_all(self, sql: str, params: tuple = None) -> List[Dict]:
        """
        执行查询并返回所有记录
        
        参数:
            sql (str): SQL查询语句
            params (tuple, 可选): SQL参数
            
        返回:
            List[Dict]: 查询结果列表
        """
        try:
            with self.engine.connect() as connection:
                if params:
                    results = connection.execute(text(sql), params).fetchall()
                else:
                    results = connection.execute(text(sql)).fetchall()
                
                # 转换为字典列表
                if results:
                    keys = results[0].keys()
                    return [{key: row[key] for key in keys} for row in results]
                return []
        except Exception as e:
            logger.error(f"查询多条记录失败: {e}")
            return []
    
    def read_sql(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """
        执行查询并返回DataFrame
        
        参数:
            sql (str): SQL查询语句
            params (tuple, 可选): SQL参数
            
        返回:
            pd.DataFrame: 查询结果DataFrame
        """
        try:
            with self.engine.connect() as connection:
                if params:
                    # 确保SQL是text对象
                    sql_text = text(sql)
                    return pd.read_sql(sql_text, connection, params=params)
                else:
                    # 确保SQL是text对象
                    sql_text = text(sql)
                    return pd.read_sql(sql_text, connection)
        except Exception as e:
            logger.error(f"读取SQL到DataFrame失败: {e}")
            return pd.DataFrame()
    
    def to_sql(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append', index: bool = False) -> bool:
        """
        将DataFrame写入数据库表
        
        参数:
            df (pd.DataFrame): 数据框
            table_name (str): 表名
            if_exists (str): 如果表存在时的处理方式：'fail', 'replace', 'append'
            index (bool): 是否写入索引
            
        返回:
            bool: 是否成功写入
        """
        try:
            # 处理日期列
            date_columns = df.select_dtypes(include=['datetime64']).columns
            for col in date_columns:
                df[col] = pd.to_datetime(df[col])
            
            # 写入数据
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)
            logger.info(f"成功写入 {len(df)} 条记录到表 {table_name}")
            return True
        except Exception as e:
            logger.error(f"写入数据到表 {table_name} 失败: {e}")
            return False
    
    def execute_ddl(self, ddl: str) -> bool:
        """
        执行DDL语句（如CREATE TABLE）
        
        参数:
            ddl (str): DDL语句
            
        返回:
            bool: 是否成功执行
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(text(ddl))
                connection.commit()
            logger.info("DDL语句执行成功")
            return True
        except Exception as e:
            logger.error(f"执行DDL语句失败: {e}")
            return False
    
    def table_exists(self, table_name: str) -> bool:
        """
        检查表是否存在
        
        参数:
            table_name (str): 表名
            
        返回:
            bool: 表是否存在
        """
        try:
            with self.engine.connect() as connection:
                # MySQL特定查询
                sql = f"""
                    SELECT COUNT(*) as count
                    FROM information_schema.tables
                    WHERE table_schema = '{DB_CONFIG['database']}'
                    AND table_name = '{table_name}'
                """
                result = connection.execute(text(sql)).fetchone()
                count = result[0] if result else 0
                return count > 0
        except Exception as e:
            logger.error(f"检查表 {table_name} 是否存在失败: {e}")
            return False
    
    def drop_table(self, table_name: str) -> bool:
        """
        删除表
        
        参数:
            table_name (str): 表名
            
        返回:
            bool: 是否成功删除
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                connection.commit()
            logger.info(f"表 {table_name} 已删除")
            return True
        except Exception as e:
            logger.error(f"删除表 {table_name} 失败: {e}")
            return False
    
    def drop_tables(self, table_names: List[str] = None) -> bool:
        """
        删除多个表
        
        参数:
            table_names (List[str], 可选): 表名列表，为None时删除所有表
            
        返回:
            bool: 是否成功删除所有表
        """
        try:
            if table_names is None:
                # 删除所有定义的表
                table_names = list(TABLE_SCHEMAS.keys())
                
            with self.engine.connect() as connection:
                for table_name in table_names:
                    connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                    logger.info(f"表 {table_name} 已删除")
                connection.commit()
            return True
        except Exception as e:
            logger.error(f"删除表失败: {e}")
            return False
    
    def create_tables(self, table_names: List[str] = None) -> bool:
        """
        创建多个表
        
        参数:
            table_names (List[str], 可选): 表名列表，为None时创建所有表
            
        返回:
            bool: 是否成功创建所有表
        """
        try:
            if table_names is None:
                # 创建所有定义的表
                schemas_to_create = TABLE_SCHEMAS
            else:
                # 创建指定的表
                schemas_to_create = {name: TABLE_SCHEMAS[name] for name in table_names if name in TABLE_SCHEMAS}
                
            with self.engine.connect() as connection:
                for table_name, schema in schemas_to_create.items():
                    connection.execute(text(schema))
                    logger.info(f"表 {table_name} 已创建")
                connection.commit()
            return True
        except Exception as e:
            logger.error(f"创建表失败: {e}")
            return False
    
    def check_data_exists(self, table_name: str, stock_code: str, trade_date: Union[str, datetime]) -> bool:
        """
        检查特定股票在特定日期的数据是否存在
        
        参数:
            table_name (str): 表名
            stock_code (str): 股票代码
            trade_date (str 或 datetime): 交易日期
            
        返回:
            bool: 数据是否存在
        """
        try:
            # 确保trade_date是字符串格式
            if isinstance(trade_date, datetime):
                trade_date = trade_date.strftime('%Y-%m-%d')
            
            # 构建SQL查询
            sql = f"""
                SELECT COUNT(*) as count
                FROM {table_name}
                WHERE stock_code = '{stock_code}'
                AND trade_date = '{trade_date}'
            """
            
            # 执行查询
            with self.engine.connect() as connection:
                result = connection.execute(text(sql)).fetchone()
                count = result[0] if result else 0
                
                return count > 0
        except Exception as e:
            logger.error(f"检查数据存在性失败: {e}")
            # 如果查询失败，返回False以允许插入尝试
            return False
    
    def log_sync_status(self, sync_type: str, start_date: Union[str, datetime], 
                        end_date: Union[str, datetime], status: str, 
                        affected_rows: int = 0, error_msg: str = None) -> bool:
        """
        记录数据同步状态
        
        参数:
            sync_type (str): 同步类型
            start_date (str 或 datetime): 开始日期
            end_date (str 或 datetime): 结束日期
            status (str): 状态
            affected_rows (int, 可选): 影响的行数
            error_msg (str, 可选): 错误信息
            
        返回:
            bool: 是否成功记录
        """
        try:
            # 确保日期是字符串格式
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
                
            sync_date = datetime.now().strftime('%Y-%m-%d')
            
            # 创建日志记录
            log_df = pd.DataFrame({
                'sync_type': [sync_type],
                'sync_date': [sync_date],
                'start_date': [start_date],
                'end_date': [end_date],
                'status': [status],
                'affected_rows': [affected_rows],
                'error_msg': [error_msg]
            })
            
            # 保存到数据库
            self.to_sql(log_df, 'data_sync_log', 'append')
            logger.info(f"已记录同步状态: {sync_type}, {status}, {affected_rows}行")
            return True
        except Exception as e:
            logger.error(f"记录同步状态失败: {e}")
            return False


if __name__ == '__main__':
    # 直接运行测试
    db = DatabaseManager()
    if db.table_exists('stock_basic'):
        print("stock_basic表存在")
    else:
        print("stock_basic表不存在")
    db.close() 