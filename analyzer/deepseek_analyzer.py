# -*- coding: utf-8 -*-
"""深度学习分析器模块，用于使用大语言模型进行股票综合分析"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import logging
import jieba
from collections import Counter
from wordcloud import WordCloud
import json
from typing import Union

from analyzer.base_analyzer import BaseAnalyzer
from utils.indicators import plot_stock_chart, calculate_technical_indicators
from utils.logger import get_logger
from utils.akshare_api import AkshareAPI
# 设置中文字体
font_path = fm.findfont(fm.FontProperties(family='SimHei'))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建日志记录器
logger = get_logger(__name__)

class DeepseekAnalyzer(BaseAnalyzer):
    """深度学习分析器类，用于使用大语言模型进行股票综合分析"""
    
    def __init__(self, stock_code: str, stock_name: str = None, end_date: Union[str, datetime] = None, 
                 days: int = 365, ai_type: str = "deepseek"):
        """
        初始化DeepSeek分析器
        
        参数:
            stock_code (str): 股票代码
            stock_name (str, 可选): 股票名称，如不提供则通过基类获取
            end_date (str 或 datetime, 可选): 结束日期，默认为当前日期
            days (int, 可选): 回溯天数，默认365天
            ai_type (str, 可选): AI模型类型，如 "deepseek", "gemini" 等
        """
        super().__init__(stock_code, stock_name, end_date, days)
        self.ai_type = ai_type
        
        # 初始化各类数据属性
        self.financial_data = None
        self.news_data = None
        self.indicators = {}
        self.tech_summary = {}
        self.financial_summary = {}
        self.news_summary = {}
        self.analysis_report = ""
        
        # 设置字体
        try:
            self.font_path = fm.findfont(fm.FontProperties(family='SimHei'))
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logger.warning(f"设置中文字体失败: {str(e)}，将使用系统默认字体")
            self.font_path = None
    
    def fetch_data(self) -> bool:
        """获取股票数据"""
        try:
            # 从AkShare获取股票日线数据
            self.daily_data = self.get_stock_daily_data()
            
            if self.daily_data.empty:
                logger.warning(f"未能获取到股票 {self.stock_code} 的数据")
                return False
            
            logger.info(f"成功获取 {self.stock_code} 的 {len(self.daily_data)} 条数据记录")
            return True
        except Exception as e:
            logger.error(f"获取股票数据时出错: {str(e)}")
            return False
    
    def fetch_financial_data(self) -> bool:
        """获取财务报表数据"""
        try:
            akshare = AkshareAPI()
            self.financial_data = akshare.get_financial_reports(
                stock_code=self.stock_code,
                start_year=str(datetime.now().year - 1)
            )
            
            if self.financial_data is not None and not (isinstance(self.financial_data, pd.DataFrame) and self.financial_data.empty):
                logger.info(f"成功获取 {self.stock_code} 的财务数据")
                return True
            logger.warning(f"未获取到 {self.stock_code} 的财务数据")
            return False
        except Exception as e:
            logger.error(f"获取财务数据失败: {e}")
            return False
    
    def fetch_news_sentiment(self, days=30) -> bool:
        """获取新闻舆情数据"""
        try:
            akshare = AkshareAPI()
            self.news_data = akshare.get_news_sentiment(
                stock_code=self.stock_code,
                stock_name=self.stock_name,
                days=days
            )
            
            if self.news_data is not None and not self.news_data.empty:
                logger.info(f"成功获取 {self.stock_code} 的新闻舆情数据")
                return True
            logger.warning(f"未获取到 {self.stock_code} 的新闻舆情数据")
            return False
        except Exception as e:
            logger.error(f"获取新闻舆情数据失败: {e}")
            return False
    
    def generate_technical_summary(self) -> dict:
        """生成技术分析摘要"""
        if self.daily_data is None or self.daily_data.empty:
            logger.warning("没有可用的技术指标数据，无法生成摘要")
            return None
        
        akshare = AkshareAPI()
        return akshare.generate_technical_summary(self.daily_data, self.indicators)
    
    def generate_financial_summary(self) -> str:
        """生成财务分析摘要"""
        if self.financial_data is None or self.financial_data.empty:
            return None
        
        akshare = AkshareAPI()
        return akshare.generate_financial_summary(
            stock_code=self.stock_code,
            stock_name=self.stock_name,
            financial_reports=self.financial_data
        )
    
    def generate_news_summary(self) -> dict:
        """生成新闻舆情摘要"""
        if self.news_data is None or self.news_data.empty:
            return None
        
        akshare = AkshareAPI()
        return akshare.generate_news_summary(self.news_data)
    
    def plot_analysis_charts(self, save_filename=None) -> bool:
        """绘制分析图表"""
        if self.daily_data is None or self.daily_data.empty:
            logger.warning("无数据可绘制。请先获取数据。")
            return False

        if save_filename is None:
            save_filename = f"{self.stock_code}_技术分析_{self.end_date.strftime('%Y%m%d')}.png"
        
        save_path = os.path.join(self.save_path, save_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            plot_days = min(60, len(self.daily_data))
            plot_df = self.daily_data.iloc[-plot_days:].copy()
            title = f'{self.stock_name}({self.stock_code}) 技术分析'
            
            fig, axes = plot_stock_chart(
                plot_df, 
                title=title, 
                save_path=save_path,
                plot_ma=True, 
                plot_volume=True, 
                plot_boll=True
            )
            
            if 'analysis_result' in self.__dict__ and isinstance(self.analysis_result, dict):
                self.analysis_result.update({'chart_path': save_path})
            
            logger.info(f"技术分析图表已保存至: {save_path}")
            return True
        except Exception as e:
            logger.error(f"绘制图表失败: {e}")
            return False
    
    def generate_word_cloud(self, save_filename=None) -> bool:
        """生成新闻词云图"""
        if self.news_data is None or self.news_data.empty:
            logger.warning("没有可用的新闻数据，无法生成词云")
            return False
        
        if save_filename is None:
            save_filename = f"{self.stock_code}_词云_{self.end_date.strftime('%Y%m%d')}.png"
        
        save_path = os.path.join(self.save_path, save_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            text = ' '.join(self.news_data['title'].tolist())
            words = ' '.join([word for word in jieba.cut(text) if len(word) > 1 and word not in ['股票', '公司']])
            
            wc = WordCloud(
                font_path=self.font_path,
                width=800,
                height=600,
                background_color='white',
                max_words=50
            ).generate(words)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'{self.stock_name} 新闻词云')
            
            plt.savefig(save_path)
            logger.info(f"词云图已保存至: {save_path}")
            plt.close()
            return True
        except Exception as e:
            logger.error(f"生成词云图失败: {e}")
            return False
    
    def analyze_with_ai(self, additional_context=None) -> str:
        """使用AI模型进行综合分析"""
        try:
            technical_summary = self.generate_technical_summary()
            financial_summary = self.generate_financial_summary()
            news_summary = self.generate_news_summary()
            
            if not technical_summary:
                logger.warning("缺少技术分析数据，无法进行AI分析")
                return "缺少技术分析数据，无法生成AI分析报告"
            
            logger.info("正在生成分析报告...")
            
            # 生成分析报告
            report = self._generate_analysis_report(
                technical_summary,
                financial_summary,
                news_summary,
                additional_context
            )
            
            # 存储报告
            self.analysis_report = report.strip()
            
            logger.info(f"成功生成 {self.stock_code} 的AI分析报告")
            return self.analysis_report
            
        except Exception as e:
            error_msg = f"AI分析出错: {e}"
            logger.error(error_msg)
            return error_msg
    
    def _generate_analysis_report(self, tech_summary, fin_summary, news_summary, additional_context) -> str:
        """生成分析报告"""
        report_parts = []
        
        # 添加标题
        report_parts.append(f"{self.stock_name}({self.stock_code})综合分析报告\n")
        
        # 技术分析部分
        tech_analysis = f"""
        技术面分析：
        目前股价为{tech_summary.get('最新收盘价', '未知')}元，
        短期趋势{tech_summary.get('短期趋势', '未知')}，中期趋势{tech_summary.get('中期趋势', '未知')}。
        MACD指标显示{tech_summary.get('MACD信号', '未知')}，RSI指标为{tech_summary.get('RSI状态', '未知')}状态。
        """
        report_parts.append(tech_analysis)
        
        # 财务分析部分
        finance_analysis = f"""
        基本面分析：
        {fin_summary if isinstance(fin_summary, str) else '无财务数据'}
        """
        report_parts.append(finance_analysis)
        
        # 新闻舆情分析部分
        if news_summary:
            sentiment = news_summary.get('舆情倾向', '未知')
            news_count = news_summary.get('新闻数量', 0)
            news_analysis = f"""
            舆情分析：
            近期共有{news_count}条相关新闻，整体舆情{sentiment}。
            """
            
            if '近期热点话题' in news_summary:
                hot_topics = '、'.join(news_summary['近期热点话题'][:3])
                news_analysis += f"热点话题包括：{hot_topics}。"
            report_parts.append(news_analysis)
        else:
            report_parts.append("舆情分析：无相关新闻数据。")
        
        # 综合建议
        if '短期趋势' in tech_summary:
            if tech_summary['短期趋势'] == '上涨' and (not news_summary or news_summary.get('舆情倾向') != '消极'):
                recommendation = "综合建议：技术面向好，可考虑逢低买入，注意控制仓位。"
            elif tech_summary['短期趋势'] == '下跌' or (news_summary and news_summary.get('舆情倾向') == '消极'):
                recommendation = "综合建议：存在下行风险，建议观望或减仓。"
            else:
                recommendation = "综合建议：市场震荡，建议持币观望。"
        else:
            recommendation = "综合建议：数据不足，无法给出明确建议。"
        report_parts.append(recommendation)
        
        # 风险提示
        risk_warning = """
        风险提示：
        股市有风险，入市需谨慎。本分析仅供参考，不构成投资建议。
        """
        report_parts.append(risk_warning)
        
        # 添加额外上下文
        if additional_context:
            report_parts.append(f"\n额外分析：\n{additional_context}")
        
        return "\n".join(report_parts)
    
    def save_analysis_report(self, filename=None) -> str:
        """保存分析报告到文件"""
        try:
            if filename is None:
                filename = f"{self.stock_code}_分析报告_{self.end_date.strftime('%Y%m%d')}.txt"
            
            save_path = os.path.join(self.save_path, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f"{'='*50}\n")
                f.write(f"{self.stock_name}({self.stock_code}) AI分析报告\n")
                f.write(f"分析日期: {self.end_date.strftime('%Y-%m-%d')}\n")
                f.write(f"{'='*50}\n\n")
                
                f.write("【技术分析】\n")
                f.write(f"{self.tech_summary}\n\n")
                
                if self.financial_summary:
                    f.write("【财务分析】\n")
                    f.write(f"{self.financial_summary}\n\n")
                
                if self.news_summary:
                    f.write("【新闻舆情】\n")
                    f.write(f"{self.news_summary}\n\n")
                
                f.write("【AI综合分析】\n")
                f.write(f"{self.analysis_report}\n\n")
                
                f.write("【技术指标数据】\n")
                for name, value in self.indicators.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{name}: {value:.4f}\n")
                    else:
                        f.write(f"{name}: {value}\n")
            
            logger.info(f"分析报告已保存到 {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"保存分析报告时出错: {str(e)}")
            return ""
    
    def run_analysis(self, save_path=None, additional_context=None) -> dict:
        """运行完整的分析流程"""
        try:
            # 获取数据
            if not self.fetch_data():
                return {'status': 'error', 'message': '获取股票数据失败'}
            
            # 获取财务数据
            self.fetch_financial_data()
            
            # 获取新闻舆情
            self.fetch_news_sentiment()
            
            # 生成技术分析摘要
            self.tech_summary = self.generate_technical_summary()
            
            # 绘制分析图表
            chart_path = ""
            if self.plot_analysis_charts():
                chart_path = os.path.join(self.save_path, f"{self.stock_code}_技术分析_{self.end_date.strftime('%Y%m%d')}.png")
            
            # 生成词云图
            wordcloud_path = ""
            if self.news_data is not None and not self.news_data.empty:
                if self.generate_word_cloud():
                    wordcloud_path = os.path.join(self.save_path, f"{self.stock_code}_词云_{self.end_date.strftime('%Y%m%d')}.png")
            
            # 使用AI进行分析
            analysis_report = self.analyze_with_ai(additional_context)
            
            # 保存分析报告
            report_path = self.save_analysis_report()
            
            # 整合结果
            result = {
                'status': 'success',
                'stock_code': self.stock_code,
                'stock_name': self.stock_name,
                'date': self.end_date.strftime('%Y-%m-%d'),
                'technical_summary': self.tech_summary,
                'financial_summary': self.financial_summary,
                'news_summary': self.news_summary,
                'chart_path': chart_path,
                'wordcloud_path': wordcloud_path,
                'report_path': report_path,
                'analysis_report': analysis_report
            }
            
            # 保存分析结果
            self.analysis_result = result
            
            logger.info(f"{self.stock_code} ({self.stock_name}) 分析完成")
            return result
            
        except Exception as e:
            error_msg = f"运行分析流程时出错: {str(e)}"
            logger.error(error_msg)
            return {'status': 'error', 'message': error_msg} 