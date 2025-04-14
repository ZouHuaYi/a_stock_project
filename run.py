#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股选股与分析工具 - 主入口文件

使用方法:
    - 选股功能: python run.py select [volume|chan] [--options]
    - 分析功能: python run.py analyze [volprice|golden|deepseek] [股票代码] [--options]
      例如: python run.py analyze golden 000001  # 对000001进行黄金分割分析
           python run.py analyze volprice 600001  # 对600001进行量价分析
           python run.py analyze openai 300059  # 对300059进行AI深度分析
    - 更新数据: python run.py update [--basic] [--daily] [--full]
    
注意：股票代码必须是6位数字，如000001、600001等

详细选项请使用 python run.py -h 查看帮助
"""

import sys
import os
from datetime import datetime

# 确保路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置
from config import BASE_CONFIG, PATH_CONFIG
from utils.logger import setup_logger
from cli.command_line import main

def ensure_directories():
    """确保所有必要的目录都存在"""
    # 获取日志记录器
    logger = setup_logger()
    
    # 确保各个目录存在
    for key, dir_path in PATH_CONFIG.items():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"创建目录: {dir_path}")

if __name__ == "__main__":
    # 确保目录存在
    ensure_directories()
    
    # 执行主函数
    main()