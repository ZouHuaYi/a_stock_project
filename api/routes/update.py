#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股选股与分析工具 - 数据更新API路由

提供股票数据更新相关API路由
"""

from fastapi import APIRouter, BackgroundTasks

from utils.logger import setup_logger
from api.models import UpdateParams, UpdateResult

# 创建路由
router = APIRouter(prefix="/api/update", tags=["update"])

# 日志记录器
logger = setup_logger("api.update")

@router.post("/", response_model=UpdateResult)
async def update_data(params: UpdateParams, background_tasks: BackgroundTasks):
    """更新股票数据"""
    logger.info(f"更新股票数据: full={params.full}, basic={params.basic}, daily={params.daily}")
    try:
        # 导入数据更新模块
        from data.stock_data import StockDataUpdater
        
        # 创建更新器
        updater = StockDataUpdater()
        
        # 根据参数执行不同的更新操作
        if params.full:
            # 在后台任务中执行全量更新
            background_tasks.add_task(updater.full_update)
            return {"success": True, "message": "全量更新任务已经开始在后台执行"}
        elif params.basic:
            # 在后台任务中执行基本信息更新
            background_tasks.add_task(updater.init_stock_basic)
            return {"success": True, "message": "基本信息更新任务已经开始在后台执行"}
        elif params.daily:
            # 在后台任务中执行日线数据更新
            background_tasks.add_task(updater.init_daily_data)
            return {"success": True, "message": "日线数据更新任务已经开始在后台执行"}
        else:
            # 如果没有指定具体更新类型，默认执行日线数据更新
            background_tasks.add_task(updater.init_daily_data)
            return {"success": True, "message": "日线数据更新任务已经开始在后台执行"}
    except Exception as e:
        logger.error(f"更新股票数据失败: {str(e)}")
        return {"success": False, "message": f"更新股票数据失败: {str(e)}"} 