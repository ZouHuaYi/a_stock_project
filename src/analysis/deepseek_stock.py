import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
from datetime import datetime, timedelta
import requests
import json
import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from src.config.config import DEEPSEEK_API_KEY

# 设置中文字体
font_path = fm.findfont(fm.FontProperties(family='SimHei'))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AStockAnalyzer:
    def __init__(self, stock_code, period="1y"):
        """
        初始化A股分析器
        :param stock_code: 股票代码，如 '600519' (贵州茅台)
        :param period: 数据周期，如 '1y' (1年)
        """
        self.stock_code = stock_code
        self.period = period
        self.stock_name = None
        self.data = None
        self.indicators = {}
        self.news_data = None
        self.financial_reports = None
        self.analysis_report = ""
        
    def fetch_stock_data(self):
        """从AKShare获取A股历史数据"""
        try:
            # 获取股票名称
            stock_info = ak.stock_individual_info_em(symbol=self.stock_code)
            self.stock_name = stock_info.loc[stock_info['item'] == '股票简称', 'value'].iloc[0]
            
            # 计算日期范围
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365*int(self.period[:-1]))).strftime('%Y%m%d')
            
            # 获取历史数据
            self.data = ak.stock_zh_a_hist(
                symbol=self.stock_code, 
                period="daily", 
                start_date=start_date, 
                end_date=end_date,
                adjust="hfq"  # 后复权
            )
            
            # 处理数据格式
            self.data['日期'] = pd.to_datetime(self.data['日期'])
            self.data.set_index('日期', inplace=True)
            self.data.rename(columns={
                '开盘': 'Open',
                '收盘': 'Close',
                '最高': 'High',
                '最低': 'Low',
                '成交量': 'Volume'
            }, inplace=True)
            
            print(f"成功获取 {self.stock_name}({self.stock_code}) 的历史数据")
            return True
        except Exception as e:
            print(f"获取数据失败: {e}")
            return False
    
    def fetch_financial_reports(self):
        """获取财务报表数据"""
        try:
            # 获取主要财务指标
            self.financial_reports = ak.stock_financial_report_sina(
                stock=self.stock_code, 
                symbol="主要财务指标"
            )
            
            # 按报告期排序
            self.financial_reports['报告日'] = pd.to_datetime(self.financial_reports['报告日'])
            self.financial_reports.sort_values('报告日', ascending=False, inplace=True)
            
            print(f"成功获取 {self.stock_name} 的财务报表数据")
            return True
        except Exception as e:
            print(f"获取财务报表失败: {e}")
            return False
    
    def fetch_news_sentiment(self, days=30):
        """获取新闻舆情数据"""
        try:
            # 这里使用模拟数据，实际应用中可接入财经新闻API
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 模拟新闻数据
            news_samples = [
                {"title": f"{self.stock_name}发布年度财报，净利润增长20%", "date": (end_date - timedelta(days=2)).strftime('%Y-%m-%d'), "sentiment": 0.8},
                {"title": f"行业政策利好，{self.stock_name}受益", "date": (end_date - timedelta(days=5)).strftime('%Y-%m-%d'), "sentiment": 0.7},
                {"title": f"{self.stock_name}大股东减持股份", "date": (end_date - timedelta(days=10)).strftime('%Y-%m-%d'), "sentiment": -0.5},
                {"title": f"分析师看好{self.stock_name}未来发展", "date": (end_date - timedelta(days=15)).strftime('%Y-%m-%d'), "sentiment": 0.6},
                {"title": f"{self.stock_name}产品通过国家认证", "date": (end_date - timedelta(days=20)).strftime('%Y-%m-%d'), "sentiment": 0.9},
            ]
            
            self.news_data = pd.DataFrame(news_samples)
            self.news_data['date'] = pd.to_datetime(self.news_data['date'])
            
            print(f"获取 {self.stock_name} 的新闻舆情数据 (模拟数据)")
            return True
        except Exception as e:
            print(f"获取新闻舆情失败: {e}")
            return False
    
    def calculate_technical_indicators(self):
        """计算各种技术指标"""
        if self.data is None or self.data.empty:
            raise ValueError("没有可用的股票数据，请先获取数据")
            
        df = self.data.copy()
        
        # 移动平均线
        self.indicators['SMA_5'] = df.ta.sma(length=5, close='Close')
        self.indicators['SMA_10'] = df.ta.sma(length=10, close='Close') 
        self.indicators['SMA_20'] = df.ta.sma(length=20, close='Close')
        self.indicators['SMA_60'] = df.ta.sma(length=60, close='Close')
        
        # 相对强弱指数(RSI)
        self.indicators['RSI_6'] = df.ta.rsi(length=6, close='Close')
        self.indicators['RSI_12'] = df.ta.rsi(length=12, close='Close')
        
        # MACD
        macd = df.ta.macd(fast=12, slow=26, signal=9, close='Close')
        self.indicators['MACD'] = macd['MACD_12_26_9']
        self.indicators['MACD_signal'] = macd['MACDs_12_26_9']
        self.indicators['MACD_hist'] = macd['MACDh_12_26_9']
        
        # KDJ指标
        stoch = df.ta.stoch(high='High', low='Low', close='Close', k=9, d=3, smooth_k=3)
        self.indicators['KDJ_K'] = stoch['STOCHk_9_3_3']
        self.indicators['KDJ_D'] = stoch['STOCHd_9_3_3']
        self.indicators['KDJ_J'] = 3 * stoch['STOCHk_9_3_3'] - 2 * stoch['STOCHd_9_3_3']
        
        # 布林带
        bbands = df.ta.bbands(length=20, close='Close')
        self.indicators['BB_upper'] = bbands['BBU_20_2.0']
        self.indicators['BB_middle'] = bbands['BBM_20_2.0']
        self.indicators['BB_lower'] = bbands['BBL_20_2.0']
        
        print("技术指标计算完成")
    
    def generate_technical_summary(self):
        """生成技术分析摘要"""
        if not self.indicators:
            raise ValueError("没有可用的技术指标，请先计算指标")
            
        last_close = self.data['Close'].iloc[-1]
        
        # 确保使用相同的索引
        last_index = self.data.index[-1]
        
        # 安全地获取指标值
        sma_5 = self.indicators['SMA_5'].loc[last_index] if last_index in self.indicators['SMA_5'].index else np.nan
        sma_10 = self.indicators['SMA_10'].loc[last_index] if last_index in self.indicators['SMA_10'].index else np.nan
        sma_20 = self.indicators['SMA_20'].loc[last_index] if last_index in self.indicators['SMA_20'].index else np.nan
        sma_60 = self.indicators['SMA_60'].loc[last_index] if last_index in self.indicators['SMA_60'].index else np.nan
        rsi_6 = self.indicators['RSI_6'].loc[last_index] if last_index in self.indicators['RSI_6'].index else np.nan
        rsi_12 = self.indicators['RSI_12'].loc[last_index] if last_index in self.indicators['RSI_12'].index else np.nan
        macd = self.indicators['MACD'].loc[last_index] if last_index in self.indicators['MACD'].index else np.nan
        macd_signal = self.indicators['MACD_signal'].loc[last_index] if last_index in self.indicators['MACD_signal'].index else np.nan
        kdj_k = self.indicators['KDJ_K'].loc[last_index] if last_index in self.indicators['KDJ_K'].index else np.nan
        kdj_d = self.indicators['KDJ_D'].loc[last_index] if last_index in self.indicators['KDJ_D'].index else np.nan
        kdj_j = self.indicators['KDJ_J'].loc[last_index] if last_index in self.indicators['KDJ_J'].index else np.nan
        
        summary = {
            '股票代码': self.stock_code,
            '股票名称': self.stock_name,
            '最新收盘价': round(last_close, 2),
            '5日均线': round(sma_5, 2) if not np.isnan(sma_5) else None,
            '10日均线': round(sma_10, 2) if not np.isnan(sma_10) else None,
            '20日均线': round(sma_20, 2) if not np.isnan(sma_20) else None,
            '60日均线': round(sma_60, 2) if not np.isnan(sma_60) else None,
            '6日RSI': round(rsi_6, 2) if not np.isnan(rsi_6) else None,
            '12日RSI': round(rsi_12, 2) if not np.isnan(rsi_12) else None,
            'MACD': round(macd, 4) if not np.isnan(macd) else None,
            'MACD信号线': round(macd_signal, 4) if not np.isnan(macd_signal) else None,
            'KDJ_K': round(kdj_k, 2) if not np.isnan(kdj_k) else None,
            'KDJ_D': round(kdj_d, 2) if not np.isnan(kdj_d) else None,
            'KDJ_J': round(kdj_j, 2) if not np.isnan(kdj_j) else None,
            '短期趋势': "上涨" if last_close > sma_5 > sma_10 else "下跌",
            '中期趋势': "上涨" if sma_10 > sma_20 > sma_60 else "下跌",
            'RSI状态': "超买" if rsi_6 > 80 or rsi_12 > 70 else "超卖" if rsi_6 < 20 or rsi_12 < 30 else "中性",
            'MACD信号': "金叉" if macd > macd_signal else "死叉",
            'KDJ信号': "超买" if kdj_j > 100 else "超卖" if kdj_j < 0 else "中性"
        }
        
        return summary
    
    def generate_financial_summary(self):
        """生成财务分析摘要"""
        if self.financial_reports is None or self.financial_reports.empty:
            return None
            
        latest_report = self.financial_reports.iloc[0]
        prev_report = self.financial_reports.iloc[1] if len(self.financial_reports) > 1 else None
        
        summary = {
            '报告期': latest_report['报告日'].strftime('%Y-%m-%d'),
            '每股收益(元)': latest_report['每股收益'],
            '每股净资产(元)': latest_report['每股净资产'],
            '净资产收益率(%)': latest_report['净资产收益率'],
            '净利润(亿元)': latest_report['净利润'],
            '同比增长(%)': latest_report['净利润同比增长率'],
            '资产负债率(%)': latest_report['资产负债率'],
            '毛利率(%)': latest_report['销售毛利率'],
            '现金流状况': "良好" if float(latest_report['经营现金流量净额'].replace('亿', '')) > 0 else "紧张"
        }
        
        if prev_report is not None:
            summary['每股收益变化'] = (float(latest_report['每股收益']) - float(prev_report['每股收益'])) / float(prev_report['每股收益']) * 100
            summary['净利润变化'] = (float(latest_report['净利润'].replace('亿', '')) - float(prev_report['净利润'].replace('亿', ''))) / float(prev_report['净利润'].replace('亿', '')) * 100
        
        return summary
    
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
    
    def analyze_with_deepseek(self, additional_context=None):
        """使用DeepSeek-Chat进行综合分析"""
        technical_summary = self.generate_technical_summary()
        financial_summary = self.generate_financial_summary()
        news_summary = self.generate_news_summary()
        
        # 准备提示词
        prompt = f"""
        你是一位专业的A股市场分析师，请根据以下数据对股票进行综合分析：
        
        1. 股票基本信息：
        - 股票代码：{technical_summary['股票代码']}
        - 股票名称：{technical_summary['股票名称']}
        - 最新收盘价：{technical_summary['最新收盘价']}元
        
        2. 技术分析摘要：
        - 短期
        3. 财务分析摘要：
        {self._format_financial_summary(financial_summary) if financial_summary else "无可用财务数据"}
        
        4. 新闻舆情分析：
        {self._format_news_summary(news_summary) if news_summary else "无可用新闻数据"}
        
        请提供以下分析内容：
        1. 综合技术面评估（趋势、关键指标信号）
        2. 基本面简评（盈利能力、财务健康状况）
        3. 舆情影响分析
        4. 综合投资建议（短期/中期）
        5. 潜在风险提示
        
        附加背景：{additional_context or '无'}
        """
    
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            } 
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一位专业的A股市场分析师，提供客观、专业的股票分析。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                result = response.json()
                self.analysis_report = result['choices'][0]['message']['content']
                return self.analysis_report
            else:
                print(f"DeepSeek API请求失败: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            print(f"调用DeepSeek API出错: {e}")
            return None

    def _format_financial_summary(self, summary):
        """格式化财务摘要"""
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
        """格式化新闻摘要"""
        return f"""
        - 近期新闻数量: {summary['新闻数量']}
        - 平均情感得分: {summary['平均情感得分']} ({summary['舆情倾向']})
        - 热点话题: {', '.join(summary['近期热点话题'])}
        - 最新新闻标题: "{summary['最新新闻标题']}"
        """

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
                for key, value in fin_summary.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # 新闻舆情摘要
            if self.news_data is not None:
                news_summary = self.generate_news_summary()
                f.write("【新闻舆情摘要】\n")
                for key, value in news_summary.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # DeepSeek分析报告
            f.write("【AI综合分析】\n")
            f.write(self.analysis_report)
        
        print(f"完整分析报告已保存至 {file_path}")

if __name__ == "__main__":
    analyzer = AStockAnalyzer("000001", "1y")
    # 获取数据
    if analyzer.fetch_stock_data():
        # 获取财务报表
        analyzer.fetch_financial_reports()
        
        # 获取新闻舆情 (模拟数据)
        analyzer.fetch_news_sentiment(days=60)
        
        # 计算技术指标
        analyzer.calculate_technical_indicators()
        
        # 绘制分析图表
        analyzer.plot_analysis_charts(save_path="technical_analysis.png")
        
        # 生成新闻词云
        analyzer.generate_word_cloud(save_path="news_wordcloud.png")
        
        # 使用DeepSeek进行分析
        analysis = analyzer.analyze_with_deepseek(
            additional_context="白酒行业龙头，具有较强品牌溢价能力"
        )
        
        if analysis:
            print("\nDeepSeek 分析报告:")
            print(analysis)
            
            # 保存完整报告
            analyzer.save_full_report("astock_analysis_report.txt")
        else:
            print("生成分析报告失败")
    else:
        print("获取股票数据失败，请检查股票代码后重试")



