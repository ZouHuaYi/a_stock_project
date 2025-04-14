# -*- coding: utf-8 -*-
"""配置模块初始化文件"""

from config.default_config import *

from .default_config import (
    BASE_CONFIG,
    PATH_CONFIG,
    DB_CONFIG,
    API_CONFIG,
    DATA_CONFIG,
    SELECTOR_CONFIG,
    ANALYZER_CONFIG
)

# 导出所有配置
__all__ = [
    'BASE_CONFIG',
    'PATH_CONFIG',
    'DB_CONFIG',
    'API_CONFIG',
    'DATA_CONFIG',
    'SELECTOR_CONFIG',
    'ANALYZER_CONFIG'
] 