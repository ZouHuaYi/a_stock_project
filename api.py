#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股选股与分析工具 - API服务

使用方法:
    - 启动API服务: python api.py
    - 或: uvicorn api:app --reload

API 端点:
    - /api/stocks: 获取股票列表
    - /api/stock/{code}: 获取单个股票信息
    - /api/analyze/{method}/{code}: 对特定股票进行分析
    - /api/select/{method}: 执行选股操作
    - /api/files/list/{folder}: 列出特定文件夹下的文件
    - /api/files/download/{folder}/{filename}: 下载特定文件
    - /api/update: 更新股票数据
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, Query, Path, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import re

# 确保路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置和其他模块
from config import BASE_CONFIG, PATH_CONFIG
from utils.logger import setup_logger

# 导入分析器和选择器模块
from analyzer import VolPriceAnalyzer, GoldenCutAnalyzer, AiAnalyzer
from selector import VolumeSelector, ChanSelector

# 创建日志记录器
logger = setup_logger("api")

# 创建FastAPI应用
app = FastAPI(
    title="A股选股分析API",
    description="提供A股股票数据查询和分析功能的API",
    version="1.0.0"
)

# 定义数据模型
class StockBase(BaseModel):
    code: str
    name: str

class StockDetail(StockBase):
    industry: Optional[str] = None
    price: Optional[float] = None
    change: Optional[float] = None
    volume: Optional[float] = None
    turnover: Optional[float] = None
    
class AnalysisResult(BaseModel):
    code: str
    method: str
    data: Dict[str, Any]
    
class SelectionParams(BaseModel):
    days: Optional[int] = Field(None, description="回溯数据天数")
    threshold: Optional[float] = Field(None, description="选股分数阈值")
    limit: Optional[int] = Field(None, description="限制结果数量")

class SelectionResult(BaseModel):
    method: str
    stocks: List[StockBase]
    file_path: Optional[str] = None
    
class UpdateParams(BaseModel):
    full: bool = Field(False, description="执行全量更新")
    basic: bool = Field(False, description="执行股票基本信息更新")
    daily: bool = Field(False, description="执行股票日线数据更新")
    
class UpdateResult(BaseModel):
    success: bool
    message: str

class FileInfo(BaseModel):
    name: str
    size: int
    created_time: str
    is_dir: bool

# API端点
@app.get("/")
async def root():
    return {"message": "欢迎使用A股选股分析API"}

@app.get("/api/stocks", response_model=List[StockBase])
async def get_stock_list(
    limit: int = Query(20, description="返回股票数量限制"),
    market: Optional[str] = Query(None, description="市场(sh:上海, sz:深圳, 不填则为全部)")
):
    """获取股票列表"""
    logger.info(f"获取股票列表, limit={limit}, market={market}")
    try:
        # 导入股票数据处理模块
        from data.stock_data import StockData
        stock_data = StockData()
        
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

@app.get("/api/stock/{code}", response_model=StockDetail)
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
        from data.stock_data import StockData
        stock_data = StockData()
        
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

@app.get("/api/analyze/{method}/{code}", response_model=AnalysisResult)
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

@app.post("/api/select/{method}", response_model=SelectionResult)
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

@app.get("/api/files/list/{folder}", response_model=List[FileInfo])
async def list_files(
    folder: str = Path(..., description="文件夹名称(results, analysis, reports, charts, temp)"),
):
    """列出指定文件夹下的文件"""
    logger.info(f"列出文件夹内容: {folder}")
    try:
        valid_folders = ["results", "analysis", "reports", "charts", "temp"]
        if folder not in valid_folders:
            raise HTTPException(status_code=400, detail=f"不支持的文件夹: {folder}，可用文件夹: {', '.join(valid_folders)}")
        
        # 构建文件夹路径
        folder_path = os.path.join(PATH_CONFIG['datas_dir'], folder)
        
        if not os.path.exists(folder_path):
            return []
        
        # 获取文件列表
        file_list = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            item_stat = os.stat(item_path)
            is_dir = os.path.isdir(item_path)
            
            file_list.append({
                "name": item,
                "size": item_stat.st_size,
                "created_time": datetime.fromtimestamp(item_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                "is_dir": is_dir
            })
        
        return file_list
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"列出文件夹内容失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"列出文件夹内容失败: {str(e)}")

@app.get("/api/files/download/{folder}/{filename}")
async def download_file(
    folder: str = Path(..., description="文件夹名称(results, analysis, reports, charts, temp)"),
    filename: str = Path(..., description="文件名称")
):
    """下载指定文件"""
    logger.info(f"下载文件: {folder}/{filename}")
    try:
        valid_folders = ["results", "analysis", "reports", "charts", "temp"]
        if folder not in valid_folders:
            raise HTTPException(status_code=400, detail=f"不支持的文件夹: {folder}，可用文件夹: {', '.join(valid_folders)}")
        
        # 构建文件路径
        file_path = os.path.join(PATH_CONFIG['datas_dir'], folder, filename)
        
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            raise HTTPException(status_code=404, detail=f"文件不存在: {folder}/{filename}")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"下载文件失败: {str(e)}")

@app.post("/api/update", response_model=UpdateResult)
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

if __name__ == "__main__":
    # 启动API服务
    uvicorn.run("api:app", host="0.0.0.0", port=9108, reload=True) 