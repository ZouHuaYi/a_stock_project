from datetime import datetime, timedelta
from src.utils.stork_api import StockAPI
from src.utils.db_manager import DatabaseManager
import pandas as pd
from src.utils.task_runner import TaskRunner

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

def save_stock_daily_data(db_manager, stock_list, start_date, end_date):
    stock_api = StockAPI()
    # 获取今天日期
    for stock_code in stock_list['stock_code']:
        stock_daily = stock_api.get_daily_data(stock_code=stock_code, start_date=start_date, end_date=end_date)
        if stock_daily is not None:
            # 这里需要一个循环，把stock_daily的每一行数据保存到数据库中
            for index, row in stock_daily.iterrows():
                # stock_code 和 trade_date 在数据库中组成的唯一键存在uk_stock_date，则不保存
                if not db_manager.check_data_exists('stock_daily', stock_code, row['trade_date']):
                    stock_daily_df = pd.DataFrame([row])
                    db_manager.save_data(stock_daily_df, 'stock_daily', 'append')
    
def init_stock_daily():
    """初始化股票日线数据"""
    db_manager = DatabaseManager()
    stock_list = db_manager.get_all_data('stock_basic')
    # 获取今天日期
    start_date = '20200101'
    end_date = datetime.now().strftime('%Y%m%d')  
    save_stock_daily_data(db_manager, stock_list, start_date, end_date)
    db_manager.close()

def init_stock_daily_day():
    """更新每天的日线数据"""
    db_manager = DatabaseManager()
    stock_list = db_manager.get_all_data('stock_basic')
    
    # 获取今天日期
    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    end_date = datetime.now().strftime('%Y%m%d')
    # 获取今天日期
    save_stock_daily_data(db_manager, stock_list, start_date, end_date)
    db_manager.close()

def init_select_data_day(trade_date):
    """更新指定的日线数据"""
    db_manager = DatabaseManager()
    stock_list = db_manager.get_all_data('stock_basic')
    
    # 获取今天日期
    start_date = trade_date
    end_date = trade_date
    # 获取今天日期
    save_stock_daily_data(db_manager, stock_list, start_date, end_date)
    db_manager.close()

if __name__ == "__main__":
    # init_stock_basic()
    # init_stock_daily()
    init_select_data_day('20250403')
