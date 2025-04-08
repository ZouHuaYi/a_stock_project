from datetime import datetime, timedelta
from src.utils.stork_api import StockAPI
from src.utils.db_manager import DatabaseManager
import pandas as pd
from src.utils.task_runner import TaskRunner

def init_stock_basic():
    """更新股票的 industry market_cap"""
    stock_api = StockAPI()
    db_manager = DatabaseManager()
    # 获取股票列表
    stock_list = db_manager.get_all_data('stock_basic')

    # 确保stock_list不为空
    if stock_list is None or stock_list.empty:
        print("获取股票列表失败")
        return
    
    # 更新股票的 industry market_cap
    for index, row in stock_list.iterrows():
        stock_info = stock_api.get_stock_info(row['stock_code'])
        if stock_info is not None:
            # 使用 db_manager 更新
            sql = f"""
                UPDATE stock_basic SET market_cap = '{stock_info['market_cap']}', industry = '{stock_info['industry']}' WHERE id = {row['id']}
            """ 
            db_manager.execute_sql(sql)
            print(f"更新股票 {row['stock_code']} 的 market_cap 和 industry")

    db_manager.close()

def supplement_save_stock_daily_data(trade_date):
    """补充保存股票日线数据"""
    stock_api = StockAPI()
    db_manager = DatabaseManager()
    # 获取股票列表
    stock_list = db_manager.get_all_data('stock_basic')
    # 确保stock_list不为空
    if stock_list is None or stock_list.empty:
        print("获取股票列表失败")
        return
    # 获取今天日期
    for stock_code in stock_list['stock_code']:
        if not db_manager.check_data_exists('stock_daily', stock_code, trade_date):
            stock_daily = stock_api.get_daily_data(stock_code=stock_code, start_date=trade_date, end_date=trade_date)
            print(f"获取股票 {stock_code} 的日线数据")
            if stock_daily is not None:
                db_manager.save_data(stock_daily, 'stock_daily', 'append')
    db_manager.close()

def save_stock_daily_data(db_manager, stock_list, start_date, end_date):
    stock_api = StockAPI()
    batch_size = 1000  # 设置批量处理的大小
    all_data = []
    
    for stock_code in stock_list['stock_code']:
        stock_daily = stock_api.get_daily_data(stock_code=stock_code, start_date=start_date, end_date=end_date)
        if stock_daily is not None:
            # 检查每条数据是否存在，不存在则加入批量插入列表
            for index, row in stock_daily.iterrows():
                if not db_manager.check_data_exists('stock_daily', stock_code, row['trade_date']):
                    all_data.append(row.to_dict())
                
            # 当积累的数据达到批处理大小时执行批量插入
            if len(all_data) >= batch_size:
                batch_df = pd.DataFrame(all_data)
                db_manager.save_data(batch_df, 'stock_daily', 'append')
                all_data = []  # 清空列表准备下一批
    
    # 处理剩余的数据
    if all_data:
        batch_df = pd.DataFrame(all_data)
        db_manager.save_data(batch_df, 'stock_daily', 'append')

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
    
    # 使用多线程加速获取和处理数据
    task_runner = TaskRunner()
    batch_size = 50  # 每批处理的股票数量
    
    # 将股票列表分批
    stock_batches = [stock_list['stock_code'][i:i + batch_size] for i in range(0, len(stock_list), batch_size)]
    
    def process_batch(stock_codes_batch):
        local_db_manager = DatabaseManager()  # 每个线程使用独立的数据库连接
        local_stock_api = StockAPI()
        all_data = []
        
        for stock_code in stock_codes_batch:
            stock_daily = local_stock_api.get_daily_data(stock_code=stock_code, start_date=start_date, end_date=end_date)
            if stock_daily is not None:
                for index, row in stock_daily.iterrows():
                    if not local_db_manager.check_data_exists('stock_daily', stock_code, row['trade_date']):
                        all_data.append(row.to_dict())
        
        if all_data:
            batch_df = pd.DataFrame(all_data)
            local_db_manager.save_data(batch_df, 'stock_daily', 'append')
        
        local_db_manager.close()
    
    # 提交任务到线程池
    for batch in stock_batches:
        task_runner.submit_task(process_batch, batch)
    
    # 等待所有任务完成
    task_runner.wait_all_done()
    db_manager.close()

if __name__ == "__main__":
    # init_stock_basic()
    # init_stock_daily()
    # init_select_data_day('20190408')
    init_stock_basic()
