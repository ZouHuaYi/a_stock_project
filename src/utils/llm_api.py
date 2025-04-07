from src.config.config import DEEPSEEK_API_KEY, GEMINI_API_KEY
import google.generativeai as genai
from openai import OpenAI

class LLMAPI:
    def __init__(self):
        self.GEMINI_API_KEY = GEMINI_API_KEY
        self.DEEPSEEK_API_KEY = DEEPSEEK_API_KEY

    def generate_gemini_response(self, prompt: str) -> str:
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 1024 * 2,  # 减少token使用量
        }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
        )
        return model.generate_content(prompt)

    def generate_deepseek_response(self, prompt: str) -> str:
        client = OpenAI(api_key=self.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024 * 2,
            temperature=1,
        )
        return response.choices[0].message.content
