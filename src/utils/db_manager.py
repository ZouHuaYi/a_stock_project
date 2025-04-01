import pymysql
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime

from src.config.config import DB_CONFIG, TABLE_SCHEMAS
from src.utils.logger import logger

class DatabaseManager:
    def __init__(self):
        """初始化数据库管理器"""
        self.engine = None
        self.connection = None
        self.connect()
    
    def connect(self):
        """创建数据库连接"""
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
    
    def close(self):
        """关闭数据库连接"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")
    
    def drop_tables(self):
        """删除所有表"""
        try:
            with self.engine.connect() as conn:
                for table_name in TABLE_SCHEMAS.keys():
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                    logger.info(f"表 {table_name} 删除成功")
            return True
        except Exception as e:
            logger.error(f"删除表失败: {e}")
            return False
    
    def create_tables(self):
        """创建数据表"""
        try:
            with self.engine.connect() as conn:
                for table_name, schema in TABLE_SCHEMAS.items():
                    # 删除可能存在的旧表
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                    # 创建新表
                    conn.execute(text(schema))
                    logger.info(f"表 {table_name} 创建成功")
            return True
        except Exception as e:
            logger.error(f"创建数据表失败: {e}")
            return False
    
    def save_data(self, df: pd.DataFrame, table: str, if_exists: str = 'append'):
        """
        保存数据到MySQL
        
        参数:
            df (DataFrame): 要保存的数据
            table (str): 表名
            if_exists (str): 如果表存在时的处理方式（'fail', 'replace', 'append'）
        """
        try:
            # 处理日期列
            date_columns = df.select_dtypes(include=['datetime64']).columns
            for col in date_columns:
                df[col] = pd.to_datetime(df[col])
            
            # 保存数据
            df.to_sql(table, self.engine, if_exists=if_exists, index=False)
            logger.info(f"成功保存{len(df)}条数据到表{table}")
            return True
        except Exception as e:
            logger.error(f"保存数据到MySQL失败: {e}")
            return False
    
    def get_data(self, table: str, columns: list = None, where: str = None):
        """获取数据"""
        try:
            with self.engine.connect() as conn:
                query = text(f"SELECT {', '.join(columns)} FROM {table}")
                if where:
                    query = query.where(text(where))
                return pd.read_sql(query, conn)
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return None
        
    def get_all_data(self, table: str):
        """获取所有数据"""
        try:
            with self.engine.connect() as conn:
                query = text(f"SELECT * FROM {table}")
                return pd.read_sql(query, conn)
        except Exception as e:
            logger.error(f"获取所有数据失败: {e}")
            return None
        
    def execute_sql(self, sql: str):
        """执行SQL语句"""
        try:
            with self.engine.connect() as conn:
                return conn.execute(text(sql))
        except Exception as e:
            logger.error(f"执行SQL语句失败: {e}")
            return None

    def log_sync_status(self, sync_type, start_date, end_date, status, affected_rows=0, error_msg=None):
        """记录数据同步状态"""
        try:
            sync_date = datetime.now().date()
            
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
            self.save_data(log_df, 'data_sync_log', 'append')
            logger.info(f"已记录同步状态: {sync_type}, {status}, {affected_rows}行")
        except Exception as e:
            logger.error(f"记录同步状态失败: {e}")

# 创建全局数据库管理器实例
db_manager = DatabaseManager() 