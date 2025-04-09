from src.analysis.deepseek_stock import AStockAnalyzer

if __name__ == "__main__":
    analyzer = AStockAnalyzer(stock_code="000905", period="1y", ai_type="gemini")
    analyzer.run_analysis()












