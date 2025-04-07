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
import google.generativeai as genai
from src.config.config import GEMINI_API_KEY, TAVILY_API_KEY

# 设置中文字体
font_path = fm.findfont(fm.FontProperties(family='SimHei'))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置Gemini API
genai.configure(api_key=GEMINI_API_KEY)

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
                adjust="qfq"  # 前复权
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
            self.financial_reports = ak.stock_financial_analysis_indicator(symbol=f"{self.stock_code}", start_year="2023")
              
            print(f"成功获取 {self.stock_code} 的财务报表数据")
            return True
        except Exception as e:
            print(f"获取财务报表失败: {e}")
            return False
    
    def fetch_news_sentiment(self, days=30):
        """使用Tavily获取新闻舆情数据"""
        try:
            if not TAVILY_API_KEY or TAVILY_API_KEY == "your_tavily_api_key_here":
                print("未配置有效的Tavily API密钥，使用模拟数据")
                return self._use_mock_news_data(days)
                
            # 确保股票名称已获取
            if not self.stock_name:
                raise ValueError("股票名称未获取，请先调用fetch_stock_data")
            
            print(f"正在使用Tavily搜索{self.stock_name}的相关新闻...")
            
            # 创建查询参数 - 对于免费API，降低复杂度
            query = f"{self.stock_code} {self.stock_name} 股票"
            payload = {
                "query": query,
                "search_depth": "basic",  # 免费API使用basic深度
                "max_results": 10,  # 减少结果数量
                "include_domains": ["eastmoney.com", "sina.com.cn", "10jqka.com.cn"]
            }
            
            # 依次尝试不同的认证方式
            return (self._try_tavily_with_header_api_key(query, payload, days) or 
                    self._try_tavily_with_bearer_token(query, payload, days) or 
                    self._try_tavily_with_url_param(query, payload, days) or 
                    self._use_mock_news_data(days))
                
        except Exception as e:
            print(f"获取新闻舆情失败: {e}")
            return self._use_mock_news_data(days)
    
    def _try_tavily_with_header_api_key(self, query, payload, days):
        """使用Header X-Api-Key认证方式尝试调用Tavily API"""
        try:
            print("尝试使用X-Api-Key认证方式...")
            headers = {
                "Content-Type": "application/json",
                "X-Api-Key": TAVILY_API_KEY
            }
            
            response = requests.post(
                "https://api.tavily.com/search",
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                return self._process_tavily_response(response, days)
            else:
                print(f"X-Api-Key认证方式失败 (HTTP {response.status_code})")
                if response.status_code != 401:  # 仅在非认证错误时显示响应
                    print(f"响应内容: {response.text}")
                return False
        except Exception as e:
            print(f"X-Api-Key认证尝试出错: {e}")
            return False
    
    def _try_tavily_with_bearer_token(self, query, payload, days):
        """使用Bearer Token认证方式尝试调用Tavily API"""
        try:
            print("尝试使用Bearer Token认证方式...")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TAVILY_API_KEY}"
            }
            
            response = requests.post(
                "https://api.tavily.com/search",
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                return self._process_tavily_response(response, days)
            else:
                print(f"Bearer Token认证方式失败 (HTTP {response.status_code})")
                if response.status_code != 401:  # 仅在非认证错误时显示响应
                    print(f"响应内容: {response.text}")
                return False
        except Exception as e:
            print(f"Bearer Token认证尝试出错: {e}")
            return False
    
    def _try_tavily_with_url_param(self, query, payload, days):
        """使用URL参数认证方式尝试调用Tavily API"""
        try:
            print("尝试使用URL参数认证方式...")
            # 将payload参数转换为URL参数
            params = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": payload["search_depth"],
                "max_results": payload["max_results"],
            }
            if "include_domains" in payload:
                domains = ",".join(payload["include_domains"])
                params["include_domains"] = domains
            
            response = requests.get(
                "https://api.tavily.com/search",
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                return self._process_tavily_response(response, days)
            else:
                print(f"URL参数认证方式失败 (HTTP {response.status_code})")
                if response.status_code != 401:  # 仅在非认证错误时显示响应
                    print(f"响应内容: {response.text}")
                return False
        except Exception as e:
            print(f"URL参数认证尝试出错: {e}")
            return False
    
    def _process_tavily_response(self, response, days):
        """处理Tavily API的成功响应"""
        try:
            results = response.json()
            
            # 提取新闻数据
            news_samples = []
            for result in results.get("results", []):
                title = result.get("title", "")
                sentiment = 0.5  # 默认中性
                
                # 简单规则：基于关键词的情感判断
                pos_words = ["利好", "增长", "提升", "突破", "上涨", "盈利", "利润"]
                neg_words = ["下跌", "亏损", "减持", "风险", "警示", "下滑", "退市"]
                
                for word in pos_words:
                    if word in title:
                        sentiment += 0.1
                
                for word in neg_words:
                    if word in title:
                        sentiment -= 0.1
                
                # 将情感值限制在[-1, 1]范围内
                sentiment = max(-1, min(1, sentiment))
                
                publish_date = result.get("published_date", datetime.now().strftime("%Y-%m-%d"))
                
                news_samples.append({
                    "title": title,
                    "date": publish_date,
                    "sentiment": sentiment,
                    "url": result.get("url", "")
                })
            
            if not news_samples:
                print("未找到相关新闻，使用模拟数据")
                return False
            
            self.news_data = pd.DataFrame(news_samples)
            self.news_data['date'] = pd.to_datetime(self.news_data['date'])
            
            print(f"成功获取 {self.stock_name} 的新闻舆情数据 ({len(news_samples)}条)")
            return True
            
        except Exception as e:
            print(f"处理Tavily响应时出错: {e}")
            return False
    
    def _use_mock_news_data(self, days=30):
        """使用模拟数据作为备选"""
        try:
            print("使用模拟新闻数据...")
            # 使用模拟数据，实际应用中可接入财经新闻API
            end_date = datetime.now()
            
            # 确保stock_name存在
            stock_name = "股票"
            if hasattr(self, 'stock_name') and self.stock_name:
                stock_name = self.stock_name
            elif hasattr(self, 'stock_code'):
                stock_name = f"股票{self.stock_code}"
                
            # 模拟新闻数据 - 更丰富的模板
            templates = [
                {"title": f"{stock_name}发布年度财报，净利润增长", "sentiment": 0.8},
                {"title": f"行业政策利好，{stock_name}有望受益", "sentiment": 0.7},
                {"title": f"{stock_name}高管增持公司股份", "sentiment": 0.6},
                {"title": f"分析师看好{stock_name}未来发展前景", "sentiment": 0.5},
                {"title": f"{stock_name}新产品获市场认可", "sentiment": 0.9},
                {"title": f"{stock_name}大股东减持股份", "sentiment": -0.5},
                {"title": f"{stock_name}业绩低于市场预期", "sentiment": -0.6},
                {"title": f"监管部门关注{stock_name}相关问题", "sentiment": -0.4},
                {"title": f"{stock_name}面临行业竞争加剧", "sentiment": -0.3},
                {"title": f"{stock_name}宣布重大投资计划", "sentiment": 0.7}
            ]
            
            # 随机选择5-7条新闻
            import random
            num_news = random.randint(5, 7)
            selected_templates = random.sample(templates, min(num_news, len(templates)))
            
            news_samples = []
            for i, template in enumerate(selected_templates):
                days_ago = random.randint(1, min(days, 30))
                news_date = (end_date - timedelta(days=days_ago)).strftime('%Y-%m-%d')
                news_samples.append({
                    "title": template["title"],
                    "date": news_date,
                    "sentiment": template["sentiment"],
                    "url": "#"
                })
            
            # 按日期排序
            news_samples.sort(key=lambda x: x["date"], reverse=True)
            
            self.news_data = pd.DataFrame(news_samples)
            self.news_data['date'] = pd.to_datetime(self.news_data['date'])
            
            print(f"已生成 {len(news_samples)} 条模拟新闻数据")
            return True
        except Exception as e:
            print(f"生成模拟数据失败: {e}")
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

        print(f"成功获取到 {len(self.financial_reports)} 条财务指标记录。")
        # print("\n原始数据预览:")
        # print(financial_indicator_df.head()) # 显示前几行数据

        # --- 3. 数据处理与选择关键指标 ---
        # 数据通常按报告日期降序排列，第一行是最新的
        latest_data = self.financial_reports.iloc[0] # 获取最新一期的数据

        print(f"\n最新报告期: {latest_data.name}") # Series的name通常是日期索引

        # 提取关键指标 (根据实际列名选择，列名可能随akshare版本或数据源变化)
        # 注意：列名需要根据实际返回的 DataFrame 进行确认！这里用常见的指标举例
        # 你可以通过 print(financial_indicator_df.columns) 查看所有列名

        # 假设列名如下（你需要根据实际情况调整）
        key_metrics = {}
        possible_metrics = {
            'roe': '净资产收益率(%)', # 盈利能力
            'net_profit_margin': '销售净利率(%)', # 盈利能力
            'gross_profit_margin': '销售毛利率(%)', # 盈利能力
            'debt_to_asset_ratio': '资产负债率(%)', # 偿债能力
            'current_ratio': '流动比率', # 短期偿债能力
            'quick_ratio': '速动比率', # 短期偿债能力
            'total_asset_turnover': '总资产周转率(次)', # 营运能力
            'eps': '基本每股收益(元)', # 每股指标
            'net_profit_growth_rate': '净利润同比增长率(%)' # 成长能力
        }

        print("\n尝试提取关键指标...")
        for key, col_name in possible_metrics.items():
            if col_name in latest_data.index:
                key_metrics[key] = latest_data[col_name]
                print(f"  - 提取到 {col_name}: {key_metrics[key]}")
            else:
                key_metrics[key] = 'N/A' # 如果找不到该列，标记为 N/A
                print(f"  - 未找到指标: {col_name}")

        # --- 4. 生成财务分析摘要 ---
        print("\n--- 财务分析摘要 ---")
        summary = f"公司代码: {self.stock_code}\n"
        summary += f"公司名称: {self.stock_name}\n"
        summary += f"最新报告期: {latest_data.name}\n\n"
        summary += "**盈利能力:**\n"
        summary += f"- 净资产收益率 (ROE): {key_metrics.get('roe', 'N/A')}% (衡量股东权益回报水平)\n"
        summary += f"- 销售净利率: {key_metrics.get('net_profit_margin', 'N/A')}% (衡量销售收入的盈利能力)\n"
        summary += f"- 销售毛利率: {key_metrics.get('gross_profit_margin', 'N/A')}% (衡量主营业务的初始盈利空间)\n\n"

        summary += "**偿债能力:**\n"
        summary += f"- 资产负债率: {key_metrics.get('debt_to_asset_ratio', 'N/A')}% (衡量总资产中通过负债筹集的比例，过高可能风险较大)\n"
        summary += f"- 流动比率: {key_metrics.get('current_ratio', 'N/A')} (衡量短期偿债能力，通常认为 > 2 较好)\n"
        summary += f"- 速动比率: {key_metrics.get('quick_ratio', 'N/A')} (更严格的短期偿债能力指标，通常认为 > 1 较好)\n\n"

        summary += "**营运能力:**\n"
        summary += f"- 总资产周转率: {key_metrics.get('total_asset_turnover', 'N/A')} 次 (衡量资产运营效率，越高越好)\n\n"

        summary += "**每股指标与成长性:**\n"
        summary += f"- 基本每股收益 (EPS): {key_metrics.get('eps', 'N/A')} 元 (衡量普通股股东每股盈利)\n"
        summary += f"- 净利润同比增长率: {key_metrics.get('net_profit_growth_rate', 'N/A')}% (衡量公司盈利的增长速度)\n\n"

        summary += "**初步结论:**\n"
        # 这里可以加入一些基于数据的判断逻辑，例如：
        roe_value = pd.to_numeric(key_metrics.get('roe'), errors='coerce')
        debt_ratio_value = pd.to_numeric(key_metrics.get('debt_to_asset_ratio'), errors='coerce')

        if roe_value is not None and roe_value > 15:  # 假设 ROE > 15% 算优秀
            summary += "- 公司盈利能力较强 (ROE 较高)。\n"
        elif roe_value is not None:
            summary += "- 公司盈利能力一般或有待观察。\n"

        if debt_ratio_value is not None and debt_ratio_value < 50:  # 假设资产负债率 < 50% 算稳健
            summary += "- 财务结构相对稳健 (资产负债率较低)。\n"
        elif debt_ratio_value is not None:
            summary += "- 需要关注公司的债务水平 (资产负债率较高)。\n"

        if key_metrics.get('net_profit_growth_rate') != 'N/A':
            growth_rate = pd.to_numeric(key_metrics.get('net_profit_growth_rate'), errors='coerce')
            if growth_rate is not None and growth_rate > 10:  # 假设增长率 > 10% 算良好
                summary += "- 公司具有一定的成长性。\n"
            elif growth_rate is not None:
                summary += "- 公司成长性需要进一步分析。\n"

        summary += "\n*注意: 此摘要仅基于部分关键财务指标的最新数据，未进行深入的趋势分析、行业对比和定性分析，不构成任何投资建议。*"
        print(summary)
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
    
    def analyze_with_gemini(self, additional_context=None):
        """使用Google Gemini-2.5 Pro进行综合分析"""
        try:
            if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
                print("未配置有效的Gemini API密钥")
                return "无法生成分析报告：API密钥无效"
                
            technical_summary = self.generate_technical_summary()
            financial_summary = self.generate_financial_summary()
            news_summary = self.generate_news_summary()
            
            print("正在使用Google Gemini生成AI分析报告...")
            
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
            尽量简洁，总字数控制在1000字以内。
            """
            
            try:
                # 配置Gemini模型
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 2024,  # 减少token使用量
                }
                
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-pro",
                    generation_config=generation_config,
                )
                
                # 生成分析报告
                response = model.generate_content(prompt)
                self.analysis_report = response.text
                return self.analysis_report
                
            except Exception as e:
                print(f"Gemini API调用错误: {e}")
                # 尝试使用备用模型
                try:
                    print("尝试使用备用模型...")
                    fallback_model = genai.GenerativeModel("gemini-1.0-pro")
                    response = fallback_model.generate_content(prompt)
                    self.analysis_report = response.text
                    return self.analysis_report
                except Exception as e2:
                    print(f"备用模型调用失败: {e2}")
                    return f"无法生成分析报告: {str(e)}"
            
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
        
        # 绘制分析图表
        self.plot_analysis_charts(save_path=f"datas/technical_{self.stock_code}.png")
        
        # 生成新闻词云
        self.generate_word_cloud(save_path=f"datas/wordcloud_{self.stock_code}.png")
        
        # 使用Gemini进行分析
        analysis = self.analyze_with_gemini(
            additional_context=context
        )

        if analysis:
            print("\n分析报告:")
            print(analysis)
            
        # 保存完整报告
        self.save_full_report(f"datas/analysis_{self.stock_code}.txt") 

        print("分析流程完成，结果已保存")
        

if __name__ == "__main__":
    analyzer = AStockAnalyzer("000001", "1y")
    analyzer.run_analysis(context="白酒行业龙头，具有较强品牌溢价能力")


