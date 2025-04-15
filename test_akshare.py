import akshare as ak
import pandas as pd

# 测试获取沪深300指数数据
print("尝试获取沪深300指数日线数据:")
try:
    df_daily = ak.stock_zh_index_daily_em(symbol='000300')
    print(f"获取到 {len(df_daily)} 条日线数据")
    print(df_daily.head())
except Exception as e:
    print(f"获取日线数据失败: {e}")

# 测试获取沪深300指数分钟数据
print("\n尝试获取沪深300指数分钟数据:")
try:
    # 注意：指数分钟数据可能需要使用不同的函数
    df_min = ak.stock_zh_index_min_em(symbol='000300', period='5')
    print(f"获取到 {len(df_min)} 条5分钟数据")
    print(df_min.head())
except Exception as e:
    print(f"获取分钟数据失败: {e}")

# 测试获取个股分钟数据作为对比
print("\n尝试获取平安银行（000001）分钟数据:")
try:
    df_stock_min = ak.stock_zh_a_hist_min_em(symbol='000001', period='5', adjust='qfq')
    print(f"获取到 {len(df_stock_min)} 条5分钟数据")
    print(df_stock_min.head())
except Exception as e:
    print(f"获取个股分钟数据失败: {e}") 