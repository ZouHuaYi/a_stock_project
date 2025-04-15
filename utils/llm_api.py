# -*- coding: utf-8 -*-
"""LLM API工具模块"""

import google.generativeai as genai
from openai import OpenAI
from utils.logger import get_logger
from config import API_CONFIG

# 创建日志记录器
logger = get_logger(__name__)

class LLMAPI:
    """与大型语言模型接口交互的工具类"""
    
    def __init__(self):
        """初始化LLM API工具"""
        self.GEMINI_API_KEY = API_CONFIG.get('gemini', {}).get('api_key', '')
        self.OPENAI_API_KEY = API_CONFIG.get('openai', {}).get('api_key', '')
        self.AI_MODEL = API_CONFIG.get('openai', {}).get('model', '')
        self.BASE_URL = API_CONFIG.get('openai', {}).get('base_url', '')

    def generate_gemini_response(self, prompt: str) -> str:
        """
        生成Gemini响应
        
        参数:
            prompt (str): 输入提示
            
        返回:
            str: 模型生成的文本响应
        """
        try:
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 1024 * 2,
            }
            genai.configure(api_key=self.GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config=generation_config,
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API 调用失败: {str(e)}")
            return ""

    def generate_openai_response(self, prompt: str) -> str:
        """
        生成OpenAI响应
        
        参数:
            prompt (str): 输入提示
            
        返回:
            str: 模型生成的文本响应
        """
        try:
            client = OpenAI(api_key=self.OPENAI_API_KEY, base_url=self.BASE_URL)
            response = client.chat.completions.create(
                model=self.AI_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024 * 4,
                temperature=1,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek API 调用失败: {str(e)}")
            return "" 