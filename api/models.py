#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股选股与分析工具 - API模型定义

包含所有API使用的Pydantic模型定义
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

# 股票相关模型
class StockBase(BaseModel):
    """股票基本信息模型"""
    code: str
    name: str

class StockDetail(StockBase):
    """股票详细信息模型"""
    industry: Optional[str] = None
    price: Optional[float] = None
    change: Optional[float] = None
    volume: Optional[float] = None
    turnover: Optional[float] = None
    
# 分析相关模型
class AnalysisResult(BaseModel):
    """股票分析结果模型"""
    code: str
    method: str
    data: Dict[str, Any]
    
# 选股相关模型
class SelectionParams(BaseModel):
    """选股参数模型"""
    days: Optional[int] = Field(None, description="回溯数据天数")
    threshold: Optional[float] = Field(None, description="选股分数阈值")
    limit: Optional[int] = Field(None, description="限制结果数量")

class SelectionResult(BaseModel):
    """选股结果模型"""
    method: str
    stocks: List[StockBase]
    file_path: Optional[str] = None
    
# 数据更新相关模型
class UpdateParams(BaseModel):
    """数据更新参数模型"""
    full: bool = Field(False, description="执行全量更新")
    basic: bool = Field(False, description="执行股票基本信息更新")
    daily: bool = Field(False, description="执行股票日线数据更新")
    
class UpdateResult(BaseModel):
    """数据更新结果模型"""
    success: bool
    message: str

# 文件相关模型
class FileInfo(BaseModel):
    """文件信息模型"""
    name: str
    size: int
    created_time: str
    is_dir: bool 