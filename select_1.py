from src.select_stock.chan import DailyChanSystem
from src.select_stock.liangneng import StockAnalyzer

if __name__ == "__main__":
    # 选股
    StockAnalyzer().run_screening()
    # DailyChanSystem().run_screening()
