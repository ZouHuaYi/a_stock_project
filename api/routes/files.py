#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股选股与分析工具 - 文件操作API路由

提供文件列表查询和下载功能相关的API路由
"""

import os
from datetime import datetime
from fastapi import APIRouter, Path, HTTPException
from fastapi.responses import FileResponse

from utils.logger import setup_logger
from config import PATH_CONFIG
from api.models import FileInfo

# 创建路由
router = APIRouter(prefix="/api/files", tags=["files"])

# 日志记录器
logger = setup_logger("api.files")

@router.get("/list/{folder}", response_model=list[FileInfo])
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

@router.get("/download/{folder}/{filename}")
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