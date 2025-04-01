import logging
import os
from src.config.config import LOG_CONFIG
from datetime import datetime

def setup_logger(name="stock_data_sync"):
    """设置日志配置"""
    # 确保日志目录存在
    log_dir = os.path.dirname(LOG_CONFIG['log_file'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(LOG_CONFIG['level'])
    
    # 创建文件处理器, 文件加上日期
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    if not os.path.exists(log_file):
        open(log_file, 'w').close()

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(LOG_CONFIG['format']))
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_CONFIG['format']))
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 创建全局logger实例
logger = setup_logger() 