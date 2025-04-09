from datetime import datetime
import pandas as pd
import requests
from src.config.config import TAVILY_API_KEY

class TavilyAPI:
    def __init__(self):
        self.api_key = TAVILY_API_KEY
        self.url = "https://api.tavily.com/search"

    def search_base_news(self, payload: dict) -> list:
        """免费搜索新闻"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(
            url=self.url, 
            headers=headers, 
            json=payload, 
            timeout=15
        )

        if response.status_code == 200:
            return response
        else:
            print(f"Bearer Token认证方式失败 (HTTP {response.status_code})")
            return False

    def process_tavily_response(self, response: requests.Response) -> list:
        """处理Tavily 舆论数据评分"""
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
                print("未找到相关新闻")
                return False
            
            print(f"获得新闻舆情数据 ({len(news_samples)}条)")
            news_data = pd.DataFrame(news_samples)
            news_data['date'] = pd.to_datetime(news_data['date'])
            news_data.sort_values(by='date', ascending=False, inplace=True)
            news_data.reset_index(drop=True, inplace=True)
            return news_data
        except Exception as e:
            print(f"处理Tavily API响应失败: {e}")
            return False
