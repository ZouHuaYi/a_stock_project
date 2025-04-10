from src.analysis.vol_price import VolPriceAnalysis
from src.analysis.golden_cut import FibonacciAnalysis
from src.analysis.deepseek_stock import AStockAnalyzer

if __name__ == "__main__":
    # analyzer = AStockAnalyzer(stock_code="000905", period="1y", ai_type="gemini")
    # analyzer.run_analysis()
    # analyzer = FibonacciAnalysis(stock_code="300776", stock_name="帝尔激光")
    # analyzer.run_analysis()
    analyzer = VolPriceAnalysis(stock_code="300776", stock_name="帝尔激光")
    result = analyzer.run_analysis(save_chart=True)
    print(result)












