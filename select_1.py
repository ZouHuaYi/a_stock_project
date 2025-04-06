from src.select_stock.chan import DailyChanSystem
from src.select_stock.liangneng import StockAnalyzer
from src.select_stock.chanlun import chanlun_select_stock

if __name__ == "__main__":
    # 选股
    # chanlun_select_stock()
    # StockAnalyzer().run_screening()
    DailyChanSystem().run_screening()
