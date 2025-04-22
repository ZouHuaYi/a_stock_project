#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股选股与分析工具 - 股票数据API路由

提供股票基本信息查询相关API路由
"""

import re
from typing import List, Optional
from fastapi import APIRouter, Query, Path, HTTPException

from utils.logger import setup_logger
from api.models import StockBase, StockDetail

# 创建路由
router = APIRouter(prefix="/api/stock", tags=["stocks"])

# 日志记录器
logger = setup_logger("api.stocks")

@router.get("/", response_model=List[StockBase])
async def get_stock_list(
    limit: int = Query(20, description="返回股票数量限制"),
    market: Optional[str] = Query(None, description="市场(sh:上海, sz:深圳, 不填则为全部)")
):
    """获取股票列表"""
    logger.info(f"获取股票列表, limit={limit}, market={market}")
    try:
        # 导入股票数据处理模块
        from data.stock_data import StockDataUpdater
        stock_data = StockDataUpdater()
        
        # 根据市场过滤股票列表
        if market:
            market = market.lower()
            if market == 'sh':
                stocks = stock_data.get_sh_stocks(limit=limit)
            elif market == 'sz':
                stocks = stock_data.get_sz_stocks(limit=limit)
            else:
                stocks = stock_data.get_all_stocks(limit=limit)
        else:
            stocks = stock_data.get_all_stocks(limit=limit)
        
        return stocks
    except Exception as e:
        logger.error(f"获取股票列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取股票列表失败: {str(e)}")

@router.get("/{code}", response_model=StockDetail)
async def get_stock_detail(
    code: str = Path(..., description="股票代码，6位数字", min_length=6, max_length=6)
):
    """获取单个股票详细信息"""
    logger.info(f"获取股票详情, code={code}")
    try:
        # 验证股票代码格式
        if not re.match(r'^\d{6}$', code):
            raise HTTPException(status_code=400, detail=f"股票代码格式无效: {code}，正确格式应为6位数字")
        
        # 导入股票数据处理模块
        from data.stock_data import StockDataUpdater
        stock_data = StockDataUpdater()
        
        # 获取股票详情
        stock_detail = stock_data.get_stock_detail(code)
        
        if not stock_detail:
            raise HTTPException(status_code=404, detail=f"股票 {code} 不存在")
            
        return stock_detail
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取股票详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取股票详情失败: {str(e)}") 