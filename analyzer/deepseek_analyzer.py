# -*- coding: utf-8 -*-
"""深度学习分析器模块，用于使用大语言模型进行股票综合分析"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
import logging
import jieba
from collections import Counter
from wordcloud import WordCloud

from analyzer.base_analyzer import BaseAnalyzer
from utils.indicators import plot_stock_chart, calculate_technical_indicators
from utils.logger import get_logger
from utils.akshare_api import get_financial_reports
# 设置中文字体
font_path = fm.findfont(fm.FontProperties(family='SimHei'))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建日志记录器
logger = get_logger(__name__)

class DeepseekAnalyzer(BaseAnalyzer):
    """深度学习分析器类，用于使用大语言模型进行股票综合分析"""
    
    def __init__(self, stock_code, stock_name=None, end_date=None, days=365, ai_type="gemini", save_path="./datas/analysis"):
        """
        初始化深度学习分析器
        
        参数:
            stock_code (str): 股票代码
            stock_name (str, 可选): 股票名称，如不提供则通过基类获取
            end_date (str 或 datetime, 可选): 结束日期，默认为当前日期
            days (int, 可选): 回溯天数，默认365天
            ai_type (str, 可选): 分析模型类型，默认"deepseek"
            save_path (str, 可选): 图片和报告保存路径，默认当前目录
        """
        super().__init__(stock_code, stock_name, end_date, days)
        self.ai_type = ai_type
        self.save_path = save_path
        self.news_data = None
        self.financial_data = None
        self.analysis_report = ""
        
        # 创建保存目录(如果不存在)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 尝试导入LLM和搜索API
        try:
            from src.utils.llm_api import LLMAPI
            from src.utils.tavily_api import TavilyAPI
            self.llm = LLMAPI()
            self.tavily_api = TavilyAPI()
            self.api_available = True
        except ImportError:
            logger.warning("未找到LLM或Tavily API模块，将无法进行AI分析和网络搜索")
            self.api_available = False
    
    def fetch_data(self) -> bool:
        """
        获取股票数据
        
        返回:
            bool: 是否成功获取数据
        """
        logger.info(f"正在获取 {self.stock_name} ({self.stock_code}) 的日线数据...")
        
        try:
            # 从数据库获取股票日线数据
            self.daily_data = self.get_stock_daily_data()
            
            if self.daily_data.empty:
                logger.warning(f"未能从数据库获取到股票 {self.stock_code} 的数据")
                return False
            else:
                logger.info(f"成功获取 {self.stock_code} 的 {len(self.daily_data)} 条数据记录")
                return True
            
        except Exception as e:
            logger.error(f"获取股票数据时出错: {str(e)}")
            return False
    
    def fetch_financial_data(self) -> bool:
        """获取财务报表数据"""
        if not self.api_available:
            logger.warning("API模块不可用，无法获取财务数据")
            return False
            
        try:
            # 使用get_stock_finance_indicator方法获取财务数据
            self.financial_data = get_financial_reports(
                limit=4,  # 获取最近4个季度的数据
                report_type="季度报告"  # 可选:"季度报告", "半年报", "年度报告"
            )
            
            if self.financial_data is not None and not self.financial_data.empty:
                logger.info(f"成功获取 {self.stock_code} 的财务数据")
                return True
            else:
                logger.warning(f"未获取到 {self.stock_code} 的财务数据")
                return False
        except Exception as e:
            logger.error(f"获取财务数据失败: {e}")
            return False
    
    def fetch_news_sentiment(self, days=30) -> bool:
        """使用Tavily获取新闻舆情数据"""
        if not self.api_available:
            logger.warning("API模块不可用，无法获取新闻舆情数据")
            return False
            
        try:    
            # 确保股票名称已获取
            if not self.stock_name:
                self.get_stock_name()
                
            if not self.stock_name:
                raise ValueError("未能获取股票名称")
            
            logger.info(f"正在搜索{self.stock_name}的相关新闻...")
            
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
                return True   
            else:
                logger.warning("未获取到新闻舆情数据")
                return False
        except Exception as e:
            logger.error(f"获取新闻舆情失败: {e}")
            return False
    
    def generate_technical_summary(self) -> dict:
        """生成技术分析摘要"""
        if self.daily_data is None or self.daily_data.empty:
            logger.warning("没有可用的技术指标数据，无法生成摘要")
            return None
        
        # 获取最近数据
        recent_data = self.daily_data.iloc[-1]
        prev_data = self.daily_data.iloc[-2] if len(self.daily_data) > 1 else None
        
        # 计算当日变化
        price_change = 0
        price_change_pct = 0
        if prev_data is not None:
            price_change = recent_data['close'] - prev_data['close']
            price_change_pct = price_change / prev_data['close'] * 100
        
        # 判断趋势
        short_trend = "上涨" if recent_data['close'] > recent_data['MA5'] else "下跌"
        medium_trend = "上涨" if recent_data['close'] > recent_data['MA20'] else "下跌"
        long_trend = "上涨" if recent_data['close'] > recent_data['MA60'] else "下跌"
        
        # MACD信号
        macd_signal = "看多" if recent_data['MACD_DIF'] > recent_data['MACD_DEA'] else "看空"
        
        # RSI状态
        rsi_value = recent_data['RSI_14']
        if rsi_value > 70:
            rsi_status = "超买"
        elif rsi_value < 30:
            rsi_status = "超卖"
        else:
            rsi_status = "中性"
        
        # 生成摘要
        summary = {
            '股票代码': self.stock_code,
            '股票名称': self.stock_name,
            '最新收盘价': round(recent_data['close'], 2),
            '价格变化': f"{round(price_change, 2)}({round(price_change_pct, 2)}%)",
            '短期趋势': short_trend,
            '中期趋势': medium_trend,
            '长期趋势': long_trend,
            'MACD信号': macd_signal,
            'RSI状态': f"{rsi_status}({round(rsi_value, 2)})"
        }
        
        return summary
    
    def generate_financial_summary(self) -> str:
        """生成财务分析摘要"""
        if self.financial_data is None or self.financial_data.empty:
            return None
        
        try:
            # 获取最新财务数据
            latest_data = self.financial_data.iloc[0]
            
            # 计算关键指标
            eps = latest_data.get('基本每股收益(元)', 0)
            roe = latest_data.get('净资产收益率(%)', 0)
            debt_ratio = latest_data.get('资产负债率(%)', 0)
            gross_margin = latest_data.get('毛利率(%)', 0)
            net_margin = latest_data.get('净利率(%)', 0)
            revenue_growth = latest_data.get('营业收入同比增长(%)', 0)
            profit_growth = latest_data.get('净利润同比增长(%)', 0)
            
            # 获取报告期
            report_date = latest_data.get('报告期', '未知')
            
            # 生成财务摘要文本
            summary = (
                f"报告期：{report_date}\n"
                f"每股收益：{eps}元\n"
                f"净资产收益率：{roe}%\n"
                f"资产负债率：{debt_ratio}%\n"
                f"毛利率：{gross_margin}%\n"
                f"净利率：{net_margin}%\n"
                f"营收同比增长：{revenue_growth}%\n"
                f"净利润同比增长：{profit_growth}%"
            )
            
            return summary
        except Exception as e:
            logger.error(f"生成财务摘要时出错: {e}")
            return "财务数据处理出错"
    
    def generate_news_summary(self) -> dict:
        """生成新闻舆情摘要"""
        if self.news_data is None or self.news_data.empty:
            return None
            
        try:
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
            
            return summary
        except Exception as e:
            logger.error(f"生成新闻摘要时出错: {e}")
            return None
    
    def plot_analysis_charts(self, save_filename=None) -> bool:
        """绘制分析图表"""
        if self.daily_data is None or self.daily_data.empty:
            logger.warning("无数据可绘制。请先获取数据。")
            return False

        if save_filename is None:
            save_filename = f"{self.stock_code}_技术分析_{self.end_date.strftime('%Y%m%d')}.png"
        save_path = os.path.join(self.save_path, self.stock_code, save_filename)
        
        # 使用通用绘图函数
        try:
            # 选择最近60个交易日的数据
            plot_days = min(60, len(self.daily_data))
            plot_df = self.daily_data.iloc[-plot_days:].copy()
            
            title = f'{self.stock_name}({self.stock_code}) 技术分析'
            
            # 调用通用绘图函数
            fig, axes = plot_stock_chart(
                plot_df, 
                title=title, 
                save_path=save_path,
                plot_ma=True, 
                plot_volume=True, 
                plot_boll=True
            )
            
            # 更新分析结果中的图表路径
            if 'analysis_result' in self.__dict__ and isinstance(self.analysis_result, dict):
                self.analysis_result.update({
                    'chart_path': save_path
                })
            
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
        save_path = os.path.join(self.save_path, self.stock_code, save_filename)
        
        try:
            text = ' '.join(self.news_data['title'].tolist())
            words = ' '.join([word for word in jieba.cut(text) if len(word) > 1 and word not in ['股票', '公司']])
            
            wc = WordCloud(
                font_path=font_path,
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
        """使用AI进行综合分析"""
        if not self.api_available:
            logger.warning("AI API不可用，无法进行AI分析")
            return "AI分析模块未加载，无法生成AI分析报告"
        
        try:
            technical_summary = self.generate_technical_summary()
            financial_summary = self.generate_financial_summary()
            news_summary = self.generate_news_summary()
            
            if not technical_summary:
                logger.warning("缺少技术分析数据，无法进行AI分析")
                return "缺少技术分析数据，无法生成AI分析报告"
            
            logger.info("正在使用AI生成分析报告...")
            
            # 准备提示词
            prompt = f"""
            你是一位专业的中国A股市场分析师，请对{technical_summary['股票名称']}({technical_summary['股票代码']})进行简要分析：
            
            1. 股票信息：{technical_summary['股票名称']}({technical_summary['股票代码']})，最新价：{technical_summary['最新收盘价']}元
            2. 技术面：短期趋势{technical_summary['短期趋势']}，中期趋势{technical_summary['中期趋势']}，
               MACD信号：{technical_summary['MACD信号']}，RSI：{technical_summary['RSI状态']}
            3. 财务：{financial_summary if financial_summary else "无财务数据"}
            4. 新闻舆情：{self._format_news_summary(news_summary) if news_summary else "无新闻数据"}
            {additional_context or ''}
            
            请分析：1.技术面评估 2.基本面简评 3.舆情分析 4.投资建议 5.风险提示
            尽量简洁，总字数控制在1500字以内。
            """
            
            # 根据指定模型类型调用不同的API
            if self.ai_type == "gemini":
                logger.info(f"使用Gemini模型生成分析报告")
                self.analysis_report = self.llm.generate_gemini_response(prompt)
            else:  # 默认使用deepseek
                logger.info(f"使用Deepseek模型生成分析报告")
                self.analysis_report = self.llm.generate_deepseek_response(prompt)
            
            return self.analysis_report
            
        except Exception as e:
            logger.error(f"AI分析出错: {e}")
            return f"AI分析过程中出错: {e}"
    
    def _format_news_summary(self, summary) -> str:
        """格式化新闻摘要"""
        if not summary:
            return "无新闻数据"
        
        formatted = (
            f"近期新闻{summary['新闻数量']}条，"
            f"平均情感得分:{summary['平均情感得分']}({summary['舆情倾向']})，"
            f"热点话题:{','.join(summary['近期热点话题'][:3]) if '近期热点话题' in summary else '无'}，"
            f"最新标题:{summary['最新新闻标题'] if '最新新闻标题' in summary else '无'}"
        )
        return formatted
    
    def save_analysis_report(self, file_name=None) -> str:
        """保存完整分析报告到文件"""
        if file_name is None:
            file_name = f"{self.stock_code}_分析报告_{self.end_date.strftime('%Y%m%d')}.txt"
        
        file_path = os.path.join(self.save_path, self.stock_code, file_name)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"{'='*40}\n")
                f.write(f"{self.stock_name}({self.stock_code}) 综合分析报告\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*40}\n\n")
                
                # 技术分析摘要
                tech_summary = self.generate_technical_summary()
                if tech_summary:
                    f.write("【技术分析摘要】\n")
                    for key, value in tech_summary.items():
                        if key not in ['股票代码', '股票名称']:
                            f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                # 财务分析摘要
                if self.financial_data is not None:
                    fin_summary = self.generate_financial_summary()
                    f.write("【财务分析摘要】\n")
                    if fin_summary:
                        f.write(fin_summary)
                    else:
                        f.write("无财务数据")
                    f.write("\n\n")
                
                # 新闻舆情摘要
                if self.news_data is not None:
                    news_summary = self.generate_news_summary()
                    f.write("【新闻舆情摘要】\n")
                    if news_summary:
                        for key, value in news_summary.items():
                            f.write(f"{key}: {value}\n")
                    else:
                        f.write("无新闻舆情数据")
                    f.write("\n\n")
                
                # AI分析报告
                f.write("【AI综合分析】\n")
                f.write(self.analysis_report if self.analysis_report else "未生成AI分析报告")
            
            logger.info(f"完整分析报告已保存至 {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"保存分析报告失败: {e}")
            return None
    
    def run_analysis(self, additional_context=None) -> dict:
        """
        运行完整分析流程
        
        参数:
            additional_context (str, 可选): 额外的分析上下文信息
            
        返回:
            dict: 分析结果
        """
        # 初始化分析结果
        self.analysis_result = {
            'status': 'error',
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'date': self.end_date.strftime('%Y-%m-%d'),
            'message': '分析未完成'
        }
        
        # 获取数据
        if not self.fetch_data():
            self.analysis_result.update({
                'message': '获取数据失败'
            })
            return self.analysis_result
        
        # 准备技术指标
        if not self.prepare_data():
            self.analysis_result.update({
                'message': '准备数据失败'
            })
            return self.analysis_result
        
        # 获取财务数据(可选)
        fin_success = self.fetch_financial_data()
        
        # 获取新闻舆情(可选)
        news_success = self.fetch_news_sentiment()
        
        # 创建数据目录
        data_dir = os.path.join(self.save_path, self.stock_code)
        os.makedirs(data_dir, exist_ok=True)
        
        # 绘制分析图表
        chart_success = self.plot_analysis_charts(save_filename=f"{self.stock_code}_技术分析.png")
        
        # 生成新闻词云(如果有新闻数据)
        if self.news_data is not None and not self.news_data.empty:
            self.generate_word_cloud(save_filename=f"{self.stock_code}_词云.png")
        
        # 使用AI进行分析
        ai_analysis = self.analyze_with_ai(additional_context)
        
        # 保存完整报告
        report_path = self.save_analysis_report(f"{self.stock_code}_分析报告.txt")
        
        # 构建分析结果
        self.analysis_result.update({
            'status': 'success',
            'technical_summary': self.generate_technical_summary(),
            'financial_summary': self.generate_financial_summary() if fin_success else None,
            'news_summary': self.generate_news_summary() if news_success else None,
            'ai_analysis': ai_analysis,
            'report_path': report_path,
            'description': f"{self.stock_name}({self.stock_code})已完成AI综合分析"
        })
        
        # 保存分析结果到数据库
        self.save_analysis_result()
        
        logger.info("分析流程完成，结果已保存")
        return self.analysis_result 