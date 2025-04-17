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

class AiAnalyzer(BaseAnalyzer):
    """深度学习分析器类，用于使用大语言模型进行股票综合分析"""
    
    def __init__(self, stock_code: str, stock_name: str = None, end_date: Union[str, datetime] = None, 
                 days: int = 365, ai_type: str = "openai"):
        """
        初始化AI分析器
        
        参数:
            stock_code (str): 股票代码
            stock_name (str, 可选): 股票名称，如不提供则通过基类获取
            end_date (str 或 datetime, 可选): 结束日期，默认为当前日期
            days (int, 可选): 回溯天数，默认365天
        """
        super().__init__(stock_code, stock_name, end_date, days)
        
        # 初始化各类数据属性
        self.financial_data = None # 财务数据
        self.news_data = None # 新闻数据
        self.tech_summary = {} # 技术分析摘要
        self.financial_summary = {} # 财务分析摘要
        self.news_summary = {} # 新闻舆情摘要
        self.analysis_report = "" # AI分析报告
        self.ai_type = ai_type
        
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
            return self.prepare_data()
        except Exception as e:
            logger.error(f"获取股票数据时出错: {str(e)}")
            return False
    
    def fetch_financial_data(self) -> bool:
        """获取财务报表数据"""
        try:
            akshare = AkshareAPI()
            self.financial_data = akshare.get_financial_reports(
                stock_code=self.stock_code,
                start_year=str(datetime.now().year - 5)
            )
            if self.financial_data is not None and not (isinstance(self.financial_data, pd.DataFrame) and self.financial_data.empty):
                logger.info(f"成功获取 {self.stock_code} 的财务数据")
                # 添加调试日志
                logger.debug(f"财务数据类型: {type(self.financial_data)}")
                if isinstance(self.financial_data, dict):
                    for key, value in self.financial_data.items():
                        logger.debug(f"{key} 数据类型: {type(value)}, 是否为空: {value.empty if isinstance(value, pd.DataFrame) else '非DataFrame'}")
                return True
            logger.warning(f"未获取到 {self.stock_code} 的财务数据")
            return False
        except Exception as e:
            logger.error(f"获取财务数据失败: {e}")
            return False
    
    def fetch_news_sentiment(self, days=30) -> bool:
        """获取新闻舆情数据"""
        try:
            logger.info(f"正在使用Tavily搜索{self.stock_name}的相关新闻...")
            
            # 创建查询参数
            query = f"{self.stock_code} {self.stock_name} 股票"
            payload = {
                "query": query,
                "search_depth": "basic",  
                "max_results": 10, 
                "include_domains": ["eastmoney.com", "sina.com.cn", "10jqka.com.cn"]
            }
            
            # 获取数据
            response = self.tavily_api.search_base_news(payload)
            if response:
                # 处理Tavily返回的新闻数据
                self.news_data = self.tavily_api.process_tavily_response(response)
                if not self.news_data.empty:
                    logger.info(f"成功获取 {self.stock_code} 的新闻舆情数据")
                    return True
                else:
                    logger.warning(f"未获取到 {self.stock_code} 的新闻舆情数据")
                    return False
            else:
                logger.warning(f"Tavily API返回空数据")
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
        summary = akshare.generate_technical_summary(self.daily_data, self.indicators)
        return summary

    def generate_financial_summary(self) -> dict:
        """生成财务分析摘要"""
        if self.financial_data is None:
            return None
        
        akshare = AkshareAPI()
        summary = akshare.generate_financial_summary(
            stock_code=self.stock_code,
            stock_name=self.stock_name,
            financial_reports=self.financial_data
        )
        
        # 存储摘要
        self.financial_summary = summary
        return summary
    
    def generate_news_summary(self) -> dict:
        """生成新闻舆情摘要"""
        if self.news_data is None or self.news_data.empty:
            return None
            
        # 情感分析
        avg_sentiment = self.news_data['sentiment'].mean()
        sentiment_status = "积极" if avg_sentiment > 0.2 else "消极" if avg_sentiment < -0.2 else "中性"
        
        # 词频分析
        text = ' '.join(self.news_data['title'].tolist())
        words = [word for word in jieba.cut(text) if len(word) > 1 and word not in ['股票', '公司']]
        word_freq = Counter(words).most_common(5)
        
        summary = {
            '新闻数量': len(self.news_data),
            '平均情感得分': round(avg_sentiment, 2),
            '舆情倾向': sentiment_status,
            '近期热点话题': [word[0] for word in word_freq],
            '最新新闻标题': self.news_data.iloc[0]['title']
        }
        self.news_summary = summary
        return summary
    
    def plot_analysis_charts(self, save_filename=None) -> str:
        """绘制分析图表"""
        if self.daily_data is None or self.daily_data.empty:
            logger.warning("无数据可绘制。请先获取数据。")
            return False

        if save_filename is None:
            save_filename = f"{self.stock_code}_技术分析_{self.end_date.strftime('%Y%m%d')}.png"
        
        save_path = os.path.join(self.save_path, save_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            plot_df = self.daily_data.copy()
            title = f'{self.stock_name}({self.stock_code}) 技术分析'
            
            plot_bool = plot_stock_chart(
                df=plot_df, 
                indicators=self.indicators,
                title=title, 
                save_path=save_path,
                plot_ma=True, 
                plot_volume=True, 
                plot_macd=True,
                plot_kdj=True,
                plot_rsi=True,
                plot_boll=True
            )
            
            if plot_bool:
                self.analysis_result.update({'chart_path': save_path})
            
            logger.info(f"技术分析图表已保存至: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"绘制图表失败: {e}")
            return ''
    
    def generate_word_cloud(self, save_filename=None) -> str:
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
            return save_path
        except Exception as e:
            logger.error(f"生成词云图失败: {e}")
            return ''
    
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

            prompt = f"""
            你是一位专业的中国A股市场分析师，请对{self.stock_name}({self.stock_code}), 请根据以下数据生成一份关于{self.stock_name}({self.stock_code})的AI分析报告：

            分析参考：{report}

            请分析：1.技术面评估 2.基本面简评 3.舆情分析 4.投资建议 5.风险提示
            尽量简洁，总字数控制在1500字以内。
            """
            # 使用AI模型进行分析
            if self.ai_type == "openai":
                analysis_report = self.llm_api.generate_openai_response(prompt)
            elif self.ai_type == "gemini":
                analysis_report = self.llm_api.generate_gemini_response(prompt)
            else:
                raise ValueError(f"不支持的AI模型类型: {self.ai_type}")
            
            # 存储报告
            self.analysis_report = analysis_report.strip()
            
            logger.info(f"成功生成 {self.stock_code} 的AI分析报告")
            return self.analysis_report
            
        except Exception as e:
            error_msg = f"AI分析出错: {e}"
            logger.error(error_msg)
            return error_msg
    
    def _generate_analysis_report(self, tech_summary, fin_summary, news_summary, additional_context) -> str:
        """生成分析报告"""
        if not isinstance(tech_summary, dict):
            logger.error("技术分析摘要格式错误")
            return "无法生成分析报告：技术分析数据格式错误"

        report_parts = []
        # 添加标题
        report_parts.append(f"{self.stock_name}({self.stock_code})综合分析报告\n")
        # 技术分析部分
        tech_analysis = f"""
        技术面分析：
        目前股价为: {tech_summary.get('当前价格', '未知')}元，
        技术信号: {tech_summary.get('技术信号', '未知')}，
        整体研判: {tech_summary.get('整体研判', '未知')}。
        """
        report_parts.append(tech_analysis)
        
        # 财务分析部分
        if isinstance(fin_summary, dict) and len(fin_summary) > 3:  # 排除只包含基本信息的摘要
            # 提取关键财务指标
            finance_analysis = "\n        基本面分析：\n"
            
            if '报告期' in fin_summary:
                finance_analysis += f"        最新报告期：{fin_summary['报告期']}\n"
                
            if '盈利能力' in fin_summary:
                finance_analysis += "        盈利能力：\n"
                for k, v in fin_summary['盈利能力'].items():
                    finance_analysis += f"        - {k}: {v}\n"
                    
            if '偿债能力' in fin_summary:
                finance_analysis += "        偿债能力：\n"
                for k, v in fin_summary['偿债能力'].items():
                    finance_analysis += f"        - {k}: {v}\n"
                    
            if '营运能力' in fin_summary:
                finance_analysis += "        营运能力：\n"
                for k, v in fin_summary['营运能力'].items():
                    finance_analysis += f"        - {k}: {v}\n"
                    
            if '每股指标' in fin_summary:
                finance_analysis += "        每股指标：\n"
                for k, v in fin_summary['每股指标'].items():
                    finance_analysis += f"        - {k}: {v}\n"
                    
            if '成长能力' in fin_summary:
                finance_analysis += "        成长能力：\n"
                for k, v in fin_summary['成长能力'].items():
                    finance_analysis += f"        - {k}: {v}\n"
                    
            if '财务评价' in fin_summary:
                finance_analysis += "        财务评价：\n"
                for item in fin_summary['财务评价']:
                    finance_analysis += f"        - {item}\n"
        elif isinstance(fin_summary, str):
            finance_analysis = f"\n        基本面分析：\n        {fin_summary}\n"
        else:
            finance_analysis = "\n        基本面分析：\n        无有效财务数据\n"
            
        report_parts.append(finance_analysis)
        
        # 新闻舆情分析部分
        if isinstance(news_summary, dict):
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
        
        report_parts.append("\n 技术指标数据：\n")
        # 技术指标数据
        for name, value in self.indicators.items():
            if isinstance(value, pd.Series):
                value = value.iloc[-5:]
            elif isinstance(value, pd.DataFrame):
                value = value.iloc[-5:]
            else:
                value = value
            report_parts.append(f"{name}: {value}")
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
                return {'status': 'error', 'message': '数据处理失败'}
           
            # 获取财务数据
            self.fetch_financial_data()
           
            # 获取新闻舆情
            self.fetch_news_sentiment()
            
            # 生成技术分析摘要
            self.tech_summary = self.generate_technical_summary()
            
            # 绘制分析图表
            chart_path = self.plot_analysis_charts()
           
            # 生成词云图
            wordcloud_path = self.generate_word_cloud()
           
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
            self.save_analysis_result()
            
            logger.info(f"{self.stock_code} ({self.stock_name}) 分析完成")
            return result
            
        except Exception as e:
            error_msg = f"运行分析流程时出错: {str(e)}"
            logger.error(error_msg)
            return {'status': 'error', 'message': error_msg} 