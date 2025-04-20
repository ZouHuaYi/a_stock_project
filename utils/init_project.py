#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目初始化模块

用于创建项目所需的目录结构，确保所有必要的目录都存在。
"""

import os
import logging
from typing import List, Dict, Any

from config.default_config import PATH_CONFIG

logger = logging.getLogger(__name__)

def create_directories() -> List[str]:
    """
    创建项目所需的所有目录
    
    Returns:
        创建的目录列表
    """
    created_dirs = []
    
    # 遍历所有路径配置
    for key, path in PATH_CONFIG.items():
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
                created_dirs.append(path)
                logger.info(f"创建目录: {path}")
            except Exception as e:
                logger.error(f"创建目录 {path} 失败: {e}")
    
    return created_dirs

def check_directories() -> Dict[str, bool]:
    """
    检查项目所需的所有目录是否存在
    
    Returns:
        目录状态字典，键为目录路径，值为是否存在
    """
    dir_status = {}
    
    for key, path in PATH_CONFIG.items():
        dir_status[path] = os.path.exists(path)
        
        if not dir_status[path]:
            logger.warning(f"目录不存在: {path}")
        else:
            logger.debug(f"目录已存在: {path}")
    
    return dir_status

def init_project() -> bool:
    """
    初始化项目，创建必要的目录结构
    
    Returns:
        初始化是否成功
    """
    logger.info("开始初始化项目目录")
    
    # 创建所有必要的目录
    created_dirs = create_directories()
    
    # 再次检查所有目录
    dir_status = check_directories()
    
    # 检查是否所有目录都存在
    all_exists = all(dir_status.values())
    
    if all_exists:
        logger.info("项目目录初始化成功")
    else:
        missing_dirs = [path for path, exists in dir_status.items() if not exists]
        logger.error(f"项目目录初始化失败，以下目录未能创建: {missing_dirs}")
    
    return all_exists

if __name__ == "__main__":
    # 设置基本日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 初始化项目
    success = init_project()
    
    if success:
        print("项目目录初始化成功！")
    else:
        print("项目目录初始化失败，请检查日志获取详细信息。") 