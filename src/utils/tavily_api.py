import requests
from src.config.config import TAVILY_API_KEY

class TavilyAPI:
    def __init__(self):
        self.api_key = TAVILY_API_KEY

    def search_base_news(self, payload: dict) -> list:
        """免费搜索新闻"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TAVILY_API_KEY}"
        }
        url = "https://api.tavily.com/search"
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)

        if response.status_code == 200:
            return response
        else:
            print(f"Bearer Token认证方式失败 (HTTP {response.status_code})")
            if response.status_code != 401:
                print(f"响应内容: {response.text}")
                return False
