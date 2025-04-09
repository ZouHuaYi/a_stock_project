import os
import matplotlib.pyplot as plt
from datetime import datetime
import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from src.utils.llm_api import LLMAPI
from src.utils.tavily_api import TavilyAPI
from src.utils.akshare_api import AkshareAPI

# 设置中文字体
font_path = fm.findfont(fm.FontProperties(family='SimHei'))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AStockAnalyzer:
    def __init__(self, stock_code, period="1y", ai_type="deepseek"):
        """
        初始化A股分析器
        :param stock_code: 股票代码，如 '600519' (贵州茅台)
        :param period: 数据周期，如 '1y' (1年)
        :param ai_type: 分析模型，如 'gemini' (Gemini-1.5 Pro) 或 'deepseek' (DeepSeek-V3)
        """
        self.stock_code = stock_code
        self.period = period
        self.stock_name = None
        self.data = None
        self.indicators = {}
        self.news_data = None
        self.financial_reports = None
        self.analysis_report = ""
        self.ai_type = ai_type
        self.llm = LLMAPI()
        self.tavily_api = TavilyAPI()
        self.akshare = AkshareAPI()

    def fetch_stock_data(self):
        """从AKShare获取A股历史数据"""
        try:
            # 获取股票名称
            self.stock_name = self.akshare.get_stock_name(self.stock_code)
            
            # 计算年数
            years = int(self.period[:-1]) if self.period[:-1].isdigit() else 1
            
            # 获取历史数据
            self.data = self.akshare.get_stock_history(
                stock_code=self.stock_code, 
                period="daily", 
                years=years, 
                adjust="qfq"  # 前复权
            )
            
            print(f"成功获取 {self.stock_name}({self.stock_code}) 的历史数据")
            return True
        except Exception as e:
            print(f"获取数据失败: {e}")
            return False
    
    def fetch_financial_reports(self):
        """获取财务报表数据"""
        try:
            self.financial_reports = self.akshare.get_financial_reports(
                stock_code=self.stock_code, 
                start_year="2023"
            )
              
            print(f"成功获取 {self.stock_code} 的财务报表数据")
            return True
        except Exception as e:
            print(f"获取财务报表失败: {e}")
            return False
    
    def fetch_news_sentiment(self, days=30):
        """使用Tavily获取新闻舆情数据"""
        try:    
            # 确保股票名称已获取
            if not self.stock_name:
                raise ValueError("未获取股票名称")
            
            print(f"正在使用Tavily搜索{self.stock_name}的相关新闻...")
            
            # 创建查询参数 - 对于免费API，降低复杂度
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
                # 处理Tavily返回的新闻数据，主要提取标题、内容、来源、发布时间、情感得分
                self.news_data = self.tavily_api.process_tavily_response(response)
                return True   
            else:
                return False
        except Exception as e:
            print(f"获取新闻舆情失败: {e}")
            return False
    
    def calculate_technical_indicators(self):
        """计算各种技术指标"""
        if self.data is None or self.data.empty:
            raise ValueError("没有可用的股票数据，请先获取数据")
        
        # 使用AkshareAPI计算技术指标    
        self.indicators = self.akshare.calculate_technical_indicators(self.data)
        
        print("技术指标计算完成")
    
    def generate_technical_summary(self):
        """生成技术分析摘要"""
        if not self.indicators:
            raise ValueError("没有可用的技术指标，请先计算指标")
        
        # 将股票代码和名称添加到数据中，以便AkshareAPI使用
        data_with_info = self.data.copy()
        data_with_info['stock_code'] = self.stock_code
        data_with_info['stock_name'] = self.stock_name
        
        # 使用AkshareAPI生成技术分析摘要
        summary = self.akshare.generate_technical_summary(data_with_info, self.indicators)
        
        return summary
    
    def generate_financial_summary(self):
        """生成财务分析摘要"""
        if self.financial_reports is None or self.financial_reports.empty:
            return None

        # 使用AkshareAPI生成财务分析摘要
        return self.akshare.generate_financial_summary(
            stock_code=self.stock_code,
            stock_name=self.stock_name,
            financial_reports=self.financial_reports
        )
    
    def generate_news_summary(self):
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
        
        return summary
    
    def plot_analysis_charts(self, save_path=None):
        """绘制分析图表"""
        if not self.indicators:
            raise ValueError("没有可用的技术指标，请先计算指标")
            
        plt.figure(figsize=(15, 25))
        
        # 确保所有指标与数据索引对齐
        data_indices = self.data.index
        
        # 价格和移动平均线
        plt.subplot(5, 1, 1)
        plt.plot(data_indices, self.data['Close'], label='收盘价')
        plt.plot(data_indices, self.indicators['SMA_5'].reindex(data_indices), label='5日均线')
        plt.plot(data_indices, self.indicators['SMA_10'].reindex(data_indices), label='10日均线')
        plt.plot(data_indices, self.indicators['SMA_20'].reindex(data_indices), label='20日均线')
        plt.title(f'{self.stock_name}({self.stock_code}) 价格走势')
        plt.legend()
        
        # MACD
        plt.subplot(5, 1, 2)
        plt.plot(data_indices, self.indicators['MACD'].reindex(data_indices), label='MACD')
        plt.plot(data_indices, self.indicators['MACD_signal'].reindex(data_indices), label='信号线')
        plt.bar(data_indices, self.indicators['MACD_hist'].reindex(data_indices), label='MACD柱状图')
        plt.title('MACD指标')
        plt.legend()
        
        # KDJ
        plt.subplot(5, 1, 3)
        plt.plot(data_indices, self.indicators['KDJ_K'].reindex(data_indices), label='K值')
        plt.plot(data_indices, self.indicators['KDJ_D'].reindex(data_indices), label='D值')
        plt.plot(data_indices, self.indicators['KDJ_J'].reindex(data_indices), label='J值')
        plt.axhline(y=80, color='r', linestyle='--')
        plt.axhline(y=20, color='g', linestyle='--')
        plt.title('KDJ指标')
        plt.legend()
        
        # RSI
        plt.subplot(5, 1, 4)
        plt.plot(data_indices, self.indicators['RSI_6'].reindex(data_indices), label='6日RSI')
        plt.plot(data_indices, self.indicators['RSI_12'].reindex(data_indices), label='12日RSI')
        plt.axhline(y=70, color='r', linestyle='--')
        plt.axhline(y=30, color='g', linestyle='--')
        plt.title('相对强弱指数(RSI)')
        plt.legend()
        
        # 成交量
        plt.subplot(5, 1, 5)
        plt.bar(data_indices, self.data['Volume'], label='成交量')
        plt.title('成交量')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"图表已保存至 {save_path}")
        else:
            plt.show()
    
    def generate_word_cloud(self, save_path=None):
        """生成新闻词云图"""
        if self.news_data is None or self.news_data.empty:
            print("没有可用的新闻数据")
            return
            
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
        
        if save_path:
            plt.savefig(save_path)
            print(f"词云图已保存至 {save_path}")
        else:
            plt.show()
    
    def analyze_with_gemini(self, additional_context=None):
        """使用Google Gemini-1.5 Pro进行综合分析"""
        try:
            
            technical_summary = self.generate_technical_summary()
            financial_summary = self.generate_financial_summary()
            news_summary = self.generate_news_summary()
            
            print("正在使用 AI 生成分析报告...")
            
            # 准备提示词 - 精简以适应免费API的token限制
            prompt = f"""
            你是一位专业的中国A股市场分析师，请对{technical_summary['股票名称']}({technical_summary['股票代码']})进行简要分析：
            
            1. 股票信息：{technical_summary['股票名称']}({technical_summary['股票代码']})，最新价：{technical_summary['最新收盘价']}元
            2. 技术面：短期趋势{technical_summary['短期趋势']}，中期趋势{technical_summary['中期趋势']}，
               MACD信号：{technical_summary['MACD信号']}，RSI：{technical_summary['RSI状态']}
            3. 财务：{financial_summary if financial_summary else "无财务数据"}
            4. 新闻舆情：{self._format_news_summary_short(news_summary) if news_summary else "无新闻数据"}
            {additional_context or ''}
            
            请分析：1.技术面评估 2.基本面简评 3.舆情分析 4.投资建议 5.风险提示
            尽量简洁，总字数控制在1500字以内。
            """
            
            try:
                if self.ai_type == "gemini":
                    self.analysis_report = self.llm.generate_gemini_response(prompt)
                elif self.ai_type == "deepseek":
                    self.analysis_report = self.llm.generate_deepseek_response(prompt)
                return self.analysis_report
                
            except Exception as e:
                print(f"Gemini API调用错误: {e}")
            
        except Exception as e:
            print(f"调用AI分析API出错: {e}")
            return f"无法生成分析报告: {str(e)}"
   
    def _format_news_summary_short(self, summary):
        """格式化新闻摘要(简洁版)"""
        if not summary:
            return "无新闻数据"
        sentiment = f"情感倾向:{summary['舆情倾向']}({summary['平均情感得分']})"
        hot_topics = f"热点:{','.join(summary['近期热点话题'][:3])}" if summary.get('近期热点话题') else ""
        latest = f"最新:{summary['最新新闻标题'][:30]}..." if len(summary.get('最新新闻标题', '')) > 30 else summary.get('最新新闻标题', '')
        return f"{sentiment}；{hot_topics}；{latest}"

    def save_full_report(self, file_path):
        """保存完整分析报告"""
        if not self.analysis_report:
            raise ValueError("没有可用的分析报告，请先生成报告")
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*40}\n")
            f.write(f"{self.stock_name}({self.stock_code}) 综合分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*40}\n\n")
            
            # 技术分析摘要
            tech_summary = self.generate_technical_summary()
            f.write("【技术分析摘要】\n")
            for key, value in tech_summary.items():
                if key not in ['股票代码', '股票名称']:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # 财务分析摘要
            if self.financial_reports is not None:
                fin_summary = self.generate_financial_summary()
                f.write("【财务分析摘要】\n")
                if fin_summary:
                    f.write(fin_summary)
                else:
                    f.write("无财务数据")
                f.write("\n")
            
            # 新闻舆情摘要
            if self.news_data is not None:
                news_summary = self.generate_news_summary()
                f.write("【新闻舆情摘要】\n")
                for key, value in news_summary.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Gemini分析报告
            f.write("【AI综合分析】\n")
            f.write(self.analysis_report)
        
        print(f"完整分析报告已保存至 {file_path}")

    # 保留DeepSeek方法但简化实现，方便兼容
    def analyze_with_deepseek(self, additional_context=None):
        """兼容性方法，内部调用Gemini"""
        return self.analyze_with_gemini(additional_context)
    
    def _format_financial_summary(self, summary):
        """格式化财务摘要(完整版)"""
        if not summary:
            return "无财务数据"
        formatted = []
        for key, value in summary.items():
            if key == '报告期':
                formatted.append(f"- 报告期: {value}")
            elif '率' in key or '变化' in key:
                formatted.append(f"- {key}: {value}%")
            elif '元' in key:
                formatted.append(f"- {key}: {value}")
            else:
                formatted.append(f"- {key}: {value}")
        return '\n'.join(formatted)
    
    def _format_news_summary(self, summary):
        """格式化新闻摘要(完整版)"""
        if not summary:
            return "无新闻数据"
        return f"""
        - 近期新闻数量: {summary['新闻数量']}
        - 平均情感得分: {summary['平均情感得分']} ({summary['舆情倾向']})
        - 热点话题: {', '.join(summary['近期热点话题']) if '近期热点话题' in summary else '无'}
        - 最新新闻标题: "{summary['最新新闻标题'] if '最新新闻标题' in summary else '无'}"
        """

    def run_analysis(self, context:str=None):
        """运行分析流程"""
        if not self.fetch_stock_data():
            print("获取股票数据失败，请检查股票代码后重试")
            return
        
        # 获取财务报表
        self.fetch_financial_reports()
        
        # 获取新闻舆情  
        self.fetch_news_sentiment()
        
        # 计算技术指标
        self.calculate_technical_indicators()   

        # 创建数据目录
        data_dir = f"datas/{self.stock_code}"
        os.makedirs(data_dir, exist_ok=True)
        
        # 绘制分析图表
        self.plot_analysis_charts(save_path=f"{data_dir}/technical.png")
        
        # 生成新闻词云
        self.generate_word_cloud(save_path=f"{data_dir}/wordcloud.png")
        
        # 使用Gemini进行分析
        analysis = self.analyze_with_gemini(
            additional_context=context
        )

        if analysis:
            print("\n分析报告:")
            print(analysis)
            
        # 保存完整报告
        self.save_full_report(f"{data_dir}/analysis.txt") 

        print("分析流程完成，结果已保存")