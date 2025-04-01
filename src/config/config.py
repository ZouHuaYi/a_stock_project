import os
from datetime import datetime, timedelta

# 运行环境的根目录
BASE_DIR = os.getcwd()

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'stock_data_news',
    'connection_string': 'mysql+pymysql://root:123456@localhost:3306/stock_data_news'
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': os.path.join(BASE_DIR, 'logs', '')
}

# 数据获取配置
DATA_FETCH_CONFIG = {
    'batch_size': 50,  # 每批处理的股票数量
    'sleep_time': 5,   # 批次间隔时间(秒)
    'initial_days': 365 * 3,  # 初始数据获取天数
    'test_stock_limit': 100,  # 测试模式股票数量限制
}

# 数据表配置
TABLE_SCHEMAS = {
    'stock_basic': """
        CREATE TABLE IF NOT EXISTS stock_basic (
            id INT AUTO_INCREMENT PRIMARY KEY,
            stock_code VARCHAR(20) NOT NULL COMMENT '股票代码',
            stock_name VARCHAR(100) NOT NULL COMMENT '股票名称',
            exchange VARCHAR(10) NOT NULL COMMENT '交易所',
            industry VARCHAR(100) NOT NULL COMMENT '行业',
            list_date DATE NOT NULL COMMENT '上市日期',
            market_cap DECIMAL(20,2) NOT NULL COMMENT '总市值',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
            UNIQUE KEY uk_stock_code (stock_code)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票基本信息表'
    """,
    
    'stock_daily': """
        CREATE TABLE IF NOT EXISTS stock_daily (
            id INT AUTO_INCREMENT PRIMARY KEY,
            stock_code VARCHAR(20) NOT NULL COMMENT '股票代码',
            trade_date DATE NOT NULL COMMENT '交易日期',
            open DECIMAL(10,2) NOT NULL COMMENT '开盘价',
            high DECIMAL(10,2) NOT NULL COMMENT '最高价',
            low DECIMAL(10,2) NOT NULL COMMENT '最低价',
            close DECIMAL(10,2) NOT NULL COMMENT '收盘价',
            volume DECIMAL(20,2) NOT NULL COMMENT '成交量',
            amount DECIMAL(20,2) NOT NULL COMMENT '成交额',
            amplitude DECIMAL(10,2) COMMENT '振幅',
            change_percent DECIMAL(10,2) COMMENT '涨跌幅',
            change_amount DECIMAL(10,2) COMMENT '涨跌额',
            turnover_rate DECIMAL(10,2) COMMENT '换手率',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
            UNIQUE KEY uk_stock_date (stock_code, trade_date),
            INDEX idx_trade_date (trade_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票日线数据表'
    """,
    
    'stock_daily_indicator': """
        CREATE TABLE IF NOT EXISTS stock_daily_indicator (
            id INT AUTO_INCREMENT PRIMARY KEY,
            stock_code VARCHAR(20) NOT NULL COMMENT '股票代码',
            trade_date DATE NOT NULL COMMENT '交易日期',
            ma5 DECIMAL(10,2) COMMENT '5日均线',
            ma10 DECIMAL(10,2) COMMENT '10日均线',
            ma20 DECIMAL(10,2) COMMENT '20日均线',
            vol_ma5 DECIMAL(20,2) COMMENT '5日成交量均线',
            vol_ma10 DECIMAL(20,2) COMMENT '10日成交量均线',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
            UNIQUE KEY uk_stock_date (stock_code, trade_date),
            INDEX idx_trade_date (trade_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票技术指标表'
    """,
    
    'data_sync_log': """
        CREATE TABLE IF NOT EXISTS data_sync_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            sync_type VARCHAR(20) NOT NULL COMMENT '同步类型',
            start_time TIMESTAMP NOT NULL COMMENT '开始时间',
            end_time TIMESTAMP NOT NULL COMMENT '结束时间',
            status VARCHAR(20) NOT NULL COMMENT '状态',
            records_count INT NOT NULL DEFAULT 0 COMMENT '记录数',
            error_message TEXT COMMENT '错误信息',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
            INDEX idx_sync_type (sync_type),
            INDEX idx_status (status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='数据同步日志表'
    """
} 