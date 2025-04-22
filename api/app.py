#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股选股与分析工具 - API应用工厂

创建和配置FastAPI应用实例
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.logger import setup_logger
from api.routes import stocks, analyze, select, files, update

# 日志记录器
logger = setup_logger("api.app")

def create_app() -> FastAPI:
    """
    创建并配置FastAPI应用实例
    
    Returns:
        FastAPI: 配置好的FastAPI应用实例
    """
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
    
    logger.info("API应用已创建并配置完成")
    
    return app 