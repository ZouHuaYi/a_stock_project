#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股选股与分析工具 - 股票分析API路由

提供股票分析相关API路由
"""

import re
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Query, Path, HTTPException

from utils.logger import setup_logger
from api.models import AnalysisResult
from analyzer import VolPriceAnalyzer, GoldenCutAnalyzer, AiAnalyzer

# 创建路由
router = APIRouter(prefix="/api/analyze", tags=["analyze"])

# 日志记录器
logger = setup_logger("api.analyze")

@router.get("/{method}/{code}", response_model=AnalysisResult)
async def analyze_stock(
    method: str = Path(..., description="分析方法(golden: 黄金分割, volprice: 量价分析, openai: AI分析)"),
    code: str = Path(..., description="股票代码，6位数字", min_length=6, max_length=6),
    days: Optional[int] = Query(None, description="回溯数据天数"),
    end_date: Optional[str] = Query(None, description="结束日期，格式：YYYY-MM-DD"),
    save_chart: bool = Query(True, description="是否保存图表"),
    ai_type: Optional[str] = Query(None, description="AI类型，仅在method=openai时有效，可选值：openai或gemini")
):
    """对股票进行分析"""
    logger.info(f"分析股票, method={method}, code={code}, days={days}, end_date={end_date}")
    try:
        # 验证股票代码格式
        if not re.match(r'^\d{6}$', code):
            raise HTTPException(status_code=400, detail=f"股票代码格式无效: {code}，正确格式应为6位数字")
        
        # 生成输出文件名
        output_file = f"{code}_{method}_{datetime.now().strftime('%Y%m%d')}"
        
        # 根据方法选择不同的分析器
        if method == "golden":
            analyzer = GoldenCutAnalyzer(
                stock_code=code,
                end_date=end_date,
                days=days
            )
        elif method == "volprice":
            analyzer = VolPriceAnalyzer(
                stock_code=code,
                end_date=end_date,
                days=days
            )
        elif method == "openai":
            if not ai_type:
                ai_type = "openai"
            analyzer = AiAnalyzer(
                stock_code=code,
                end_date=end_date,
                days=days,
                ai_type=ai_type
            )
        else:
            raise HTTPException(status_code=400, detail=f"不支持的分析方法: {method}")
        
        # 执行分析
        result = analyzer.run_analysis(save_path=output_file)
        
        if not result:
            raise HTTPException(status_code=500, detail=f"分析失败，未返回结果")
        
        # 获取分析结果和图表路径
        chart_path = None
        if save_chart and hasattr(analyzer, 'chart_path') and analyzer.chart_path:
            chart_path = analyzer.chart_path
        
        return {
            "code": code,
            "method": method,
            "data": {
                "result": result,
                "chart_path": chart_path
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析股票失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析股票失败: {str(e)}") 