#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股选股与分析工具 - API服务入口

使用方法:
    - 启动API服务: python main.py
    - 或: uvicorn main:app --reload

API 端点:
    - /api/stock: 获取股票列表
    - /api/stock/{code}: 获取单个股票信息
    - /api/analyze/{method}/{code}: 对特定股票进行分析
    - /api/select/{method}: 执行选股操作
    - /api/files/list/{folder}: 列出特定文件夹下的文件
    - /api/files/download/{folder}/{filename}: 下载特定文件
    - /api/update: 更新股票数据
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 确保路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入路由模块
from api.routes import stocks, analyze, select, files, update

# 创建FastAPI应用
app = FastAPI(
    title="A股选股分析API",
    description="提供A股股票数据查询和分析功能的API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 添加根路由
@app.get("/")
async def root():
    return {"message": "欢迎使用A股选股分析API"}

# 注册路由
app.include_router(stocks.router)
app.include_router(analyze.router)
app.include_router(select.router)
app.include_router(files.router)
app.include_router(update.router)

if __name__ == "__main__":
    # 启动API服务
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True) 