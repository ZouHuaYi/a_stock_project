#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股选股与分析工具 - 选股API路由

提供选股操作相关API路由
"""

from datetime import datetime
from fastapi import APIRouter, Path, HTTPException

from utils.logger import setup_logger
from api.models import SelectionParams, SelectionResult
from selector import VolumeSelector, ChanSelector

# 创建路由
router = APIRouter(prefix="/api/select", tags=["select"])

# 日志记录器
logger = setup_logger("api.select")

@router.post("/{method}", response_model=SelectionResult)
async def select_stocks(
    method: str = Path(..., description="选股方法(volume: 成交量选股, chan: 缠论选股)"),
    params: SelectionParams = None
):
    """执行选股操作"""
    logger.info(f"执行选股, method={method}, params={params}")
    try:
        # 根据方法选择不同的选股器
        if method == "volume":
            selector = VolumeSelector(
                days=params.days if params else None,
                threshold=params.threshold if params else None,
                limit=params.limit if params else None
            )
        elif method == "chan":
            selector = ChanSelector(
                days=params.days if params else None,
                threshold=params.threshold if params else None,
                limit=params.limit if params else None
            )
        else:
            raise HTTPException(status_code=400, detail=f"不支持的选股方法: {method}")
        
        # 执行选股
        results = selector.run_screening()
        
        # 如果有结果，保存
        file_path = None
        if not results.empty:
            # 生成输出文件名
            output_file = f"{method}_selection_{datetime.now().strftime('%Y%m%d')}.csv"
            
            # 保存结果
            file_path = selector.save_results(results, output_file)
            
            # 转换结果格式
            stocks = []
            for _, row in results.iterrows():
                stocks.append({
                    "code": row.get('code', ''),
                    "name": row.get('name', '')
                })
        else:
            stocks = []
        
        return {
            "method": method,
            "stocks": stocks,
            "file_path": file_path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"执行选股失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"执行选股失败: {str(e)}") 