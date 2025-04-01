from src.utils.stork_api import StockAPI
from src.utils.db_manager import DatabaseManager
import pandas as pd

def init_stock_basic():
    stock_api = StockAPI()
    stock_list = stock_api.get_stock_list()

    # 确保stock_list不为空
    if stock_list is None or stock_list.empty:
        print("获取股票列表失败")
        return
        
    db_manager = DatabaseManager()

    def get_stock_info_task(stock_code):
        """获取单个股票信息的任务函数"""
        print(f"正在获取股票信息: {stock_code}")
        stock_info = stock_api.get_stock_info(stock_code)
        stock_info_df = pd.DataFrame([stock_info])
        db_manager.save_data(stock_info_df, 'stock_basic', 'append')
        return stock_info
    
    for stock_code in stock_list['stock_code']:
        get_stock_info_task(stock_code)

    db_manager.close()


def init_stock_daily():
    stock_api = StockAPI()
    db_manager = DatabaseManager()
    # 查询数据库 stock_basic 表，获取所有股票代码
    stock_list = db_manager.get_all_data('stock_basic')
    # 遍历股票代码，获取每日数据
    for stock_code in stock_list['stock_code']:
        stock_daily = stock_api.get_daily_data(stock_code=stock_code, start_date='20200101', end_date='20250401')
        if stock_daily is not None:
            # 这里需要一个循环，把stock_daily的每一行数据保存到数据库中
            for index, row in stock_daily.iterrows():
                stock_daily_df = pd.DataFrame([row])    
                db_manager.save_data(stock_daily_df, 'stock_daily', 'append')

    db_manager.close()

if __name__ == "__main__":
    init_stock_daily()
