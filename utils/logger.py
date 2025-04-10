# -*- coding: utf-8 -*-
"""日志模块"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys

# 导入配置
from config import LOG_CONFIG, PATH_CONFIG

# 确保日志目录存在
if not os.path.exists(PATH_CONFIG['log_dir']):
    os.makedirs(PATH_CONFIG['log_dir'])

# 全局日志格式
log_formatter = logging.Formatter(LOG_CONFIG['format'])

def setup_logger(name=None, level=None):
    """
    设置日志记录器
    
    参数:
        name (str, 可选): 记录器名称
        level (str, 可选): 日志级别
        
    返回:
        logging.Logger: 日志记录器
    """
    # 使用传入的名称或默认值
    logger_name = name or __name__
    
    # 使用传入的级别或配置中的级别
    log_level = level or LOG_CONFIG['level']
    
    # 创建记录器
    logger = logging.getLogger(logger_name)
    
    # 如果记录器已经有处理器，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    logger.setLevel(getattr(logging, log_level))
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(getattr(logging, LOG_CONFIG['console_level']))
    logger.addHandler(console_handler)
    
    # 创建文件处理器
    log_file = os.path.join(PATH_CONFIG['log_dir'], LOG_CONFIG['filename'])
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOG_CONFIG['max_bytes'],
        backupCount=LOG_CONFIG['backup_count'],
        encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(getattr(logging, LOG_CONFIG['file_level']))
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name=None):
    """
    获取日志记录器
    
    参数:
        name (str, 可选): 记录器名称
        
    返回:
        logging.Logger: 日志记录器
    """
    # 使用传入的名称或调用模块的名称
    logger_name = name or logging.getLogger().name
    
    # 返回已配置的记录器
    return setup_logger(logger_name) 