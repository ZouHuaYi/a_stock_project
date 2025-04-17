# -*- coding: utf-8 -*-
"""默认配置文件"""

import os
from datetime import datetime

# 基础配置
BASE_CONFIG = {
    'app_name': 'A股选股与分析工具',
    'app_version': '1.0.0',
    'app_date': '2023-05-01',
    'debug': True
}

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_CONFIG = {
    'log_dir': os.path.join(BASE_DIR, 'logs'),
    'data_dir': os.path.join(BASE_DIR, 'data'),
    'output_dir': os.path.join(BASE_DIR, 'output'),
    'analyzer_path': os.path.join(BASE_DIR, 'output', 'analyzer'),
    'selector_path': os.path.join(BASE_DIR, 'output', 'selector'),
    'temp_path': os.path.join(BASE_DIR, 'output', 'temp')
}

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'stock_data_news',
    'charset': 'utf8mb4',
    'connection_string': 'mysql+pymysql://root:123456@localhost:3306/stock_data_news?charset=utf8mb4'
}

# API配置
API_CONFIG = {
    'akshare': {
        'timeout': 30,
        'retry': 3
    },
    'gemini': {
        'api_key': "xxxxx",
        'timeout': 60
    },
    'openai': {
        'api_key': "sk-xxxx",
        'model': "deepseek/deepseek-chat-v3-0324:free",
        'base_url': "https://openrouter.ai/api/v1",
        'timeout': 60
    },
    'tavily': {
        'api_key': "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",
        'timeout': 30
    }
}

# 数据同步配置
DATA_CONFIG = {
    'batch_size': 20,  # 批处理大小
    'sleep_time': 2,  # 批次之间休眠时间（秒）
    'initial_days': 365,  # 初始同步天数
    'test_stock_limit': 0,  # 测试模式限制股票数量，0为不限制
    'retry_count': 3,  # 重试次数
    'retry_interval': 5  # 重试间隔（秒）
}

# 选股器配置
SELECTOR_CONFIG = {
    'default_days': 120,  # 默认回溯天数
    'default_threshold': 2,  # 默认选股分数阈值
    'default_limit': 50,  # 默认结果数量限制
    'batch_size': 20,  # 批处理股票数量
    'sleep_time': 1  # 批次间隔（秒）
}

# 分析器配置
ANALYZER_CONFIG = {
    'default_days': 365,  # 默认回溯天数
    'chart_style': 'seaborn-v0_8-darkgrid',  # 图表风格
    'chart_figsize': (16, 9),  # 图表尺寸
    'chart_dpi': 100,  # 图表DPI
    'font_family': 'SimHei'  # 字体
}

# 表结构配置
TABLE_SCHEMAS = {
    'stock_basic': """
        CREATE TABLE IF NOT EXISTS stock_basic (
            id INT AUTO_INCREMENT PRIMARY KEY,
            stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
            stock_name VARCHAR(50) NOT NULL COMMENT '股票名称',
            industry VARCHAR(50) COMMENT '所属行业',
            exchange VARCHAR(10) COMMENT '交易所',
            market_cap DECIMAL(20, 2) COMMENT '总市值(亿元)',
            list_date DATE COMMENT '上市日期',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uidx_code (stock_code)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票基本信息表';
    """,
    
    'stock_daily': """
        CREATE TABLE IF NOT EXISTS stock_daily (
            id INT AUTO_INCREMENT PRIMARY KEY,
            stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
            trade_date DATE NOT NULL COMMENT '交易日期',
            open DECIMAL(10, 2) COMMENT '开盘价',
            high DECIMAL(10, 2) COMMENT '最高价',
            low DECIMAL(10, 2) COMMENT '最低价',
            close DECIMAL(10, 2) COMMENT '收盘价',
            volume DECIMAL(20, 0) COMMENT '成交量',
            amount DECIMAL(20, 2) COMMENT '成交额',
            change_percent DECIMAL(10, 2) COMMENT '涨跌幅(%)',
            change_amount DECIMAL(10, 2) COMMENT '涨跌额',
            turnover_rate DECIMAL(10, 2) COMMENT '换手率(%)',
            amplitude DECIMAL(10, 2) COMMENT '振幅(%)',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uidx_code_date (stock_code, trade_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票日线行情表';
    """,
    
    'data_sync_log': """
        CREATE TABLE IF NOT EXISTS data_sync_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            sync_type VARCHAR(50) NOT NULL COMMENT '同步类型',
            start_time DATETIME NOT NULL COMMENT '开始时间',
            end_time DATETIME NOT NULL COMMENT '结束时间',
            status VARCHAR(20) NOT NULL COMMENT '状态',
            records_count INT NOT NULL DEFAULT 0 COMMENT '记录数',
            error_message TEXT COMMENT '错误信息',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='数据同步日志表';
    """,
    
    'stock_selection_result': """
        CREATE TABLE IF NOT EXISTS stock_selection_result (
            id INT AUTO_INCREMENT PRIMARY KEY,
            selection_date DATE NOT NULL COMMENT '选股日期',
            selector_type VARCHAR(50) NOT NULL COMMENT '选股器类型',
            stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
            stock_name VARCHAR(50) NOT NULL COMMENT '股票名称',
            score DECIMAL(10, 2) COMMENT '选股分数',
            current_price DECIMAL(10, 2) COMMENT '当前价格',
            positive_signals TEXT COMMENT '积极信号',
            warnings TEXT COMMENT '警告信号',
            additional_info TEXT COMMENT '附加信息',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='选股结果表';
    """,
    
    'stock_analysis_report': """
        CREATE TABLE IF NOT EXISTS stock_analysis_report (
            id INT AUTO_INCREMENT PRIMARY KEY,
            analysis_date DATE NOT NULL COMMENT '分析日期',
            analyzer_type VARCHAR(50) NOT NULL COMMENT '分析器类型',
            stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
            stock_name VARCHAR(50) NOT NULL COMMENT '股票名称',
            analysis_content TEXT COMMENT '分析内容',
            chart_path VARCHAR(255) COMMENT '图表路径',
            additional_info TEXT COMMENT '附加信息',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票分析报告表';
    """
}

# 日志配置
LOG_CONFIG = {
    'filename': datetime.now().strftime('stock_%Y%m%d.log'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'level': 'INFO',
    'console_level': 'INFO',
    'file_level': 'DEBUG',
    'max_bytes': 10485760,  # 10MB
    'backup_count': 5  # 保留5个备份文件
} 