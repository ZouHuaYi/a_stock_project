import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.utils.db_manager import DatabaseManager

# --- 参数配置 ---
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
YEARLY_MA_PERIOD = 250 # 年线近似周期
LOOKBACK_PERIOD_DIV = 30 # 背驰观察期（例如最近30个交易日）
LOOKBACK_PERIOD_PATTERN = 120 # 下跌+盘整+下跌模式观察期（例如最近120个交易日）


def get_stock_data(stock_code, start_date, end_date):
    """从数据库获取指定股票和时间段的日线数据"""
    db_manager = DatabaseManager()

    query = f"""
    SELECT trade_date, open, high, low, close, volume
    FROM stock_daily
    WHERE stock_code = '{stock_code}'
    AND trade_date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY trade_date ASC
    """
    try:
        # 使用 Pandas 读取数据
        result = db_manager.execute_sql(query)
        # 确保将游标结果转换为 DataFrame
        df = pd.DataFrame(result.fetchall(), columns=['trade_date', 'open', 'high', 'low', 'close', 'volume'])
        if df is not None and not df.empty:
            # 确保所有数值列都是float类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
        return df
    except Exception as e:
        print(f"查询股票 {stock_code} 数据失败: {e}")
        return pd.DataFrame()
    finally:
        db_manager.close()

def get_all_stock_codes():
    """获取数据库中所有股票代码"""
    db_manager = DatabaseManager()
    query = "SELECT stock_code FROM stock_basic"
    try:
        result = db_manager.execute_sql(query)
        # 确保将游标结果转换为 DataFrame
        df = pd.DataFrame(result.fetchall(), columns=['stock_code'])
        return df['stock_code'].tolist()
    except Exception as e:
        print(f"获取所有股票代码失败: {e}")
        return []
    finally:
        db_manager.close()

# --- 指标计算 ---
def calculate_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """计算MACD指标"""
    if df.empty or len(df) < slow:
        return df
    df['close'] = df['close'].astype(float)  # 确保为 float 类型
    df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['DIF'] = df['EMA_fast'] - df['EMA_slow']
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = (df['DIF'] - df['DEA']) * 2
    return df

def calculate_ma(df, period):
    """计算移动平均线"""
    if df.empty or len(df) < period:
        return df
    df['close'] = df['close'].astype(float)  # 确保为 float 类型
    df[f'MA_{period}'] = df['close'].rolling(window=period).mean()
    return df

# --- 缠论选股思路实现 (简化) ---
def check_macd_bottom_divergence(df, lookback=LOOKBACK_PERIOD_DIV):
    """
    检查近期是否可能出现MACD底背驰 (简化版 - 第一类买点信号之一)
    思路：在最近lookback周期内，股价创近期新低，但MACD柱状线(或DIF)的低点抬高。
    参考：缠论中背驰是核心概念，常结合MACD辅助判断[^7]。
    """
    if df.empty or len(df) < lookback + MACD_SLOW: # 确保有足够数据计算MACD和观察
        return False

    # 确保所有数值列都是float类型
    for col in ['low', 'MACD_hist', 'DIF']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    recent_data = df.iloc[-lookback:]
    if len(recent_data) < 5: # 需要至少几个点来判断趋势
        return False

    # 找到最近一个交易日和之前的最低点
    current_low_price = float(recent_data['low'].iloc[-1])
    previous_low_idx = recent_data['low'].iloc[:-1].idxmin() # 倒数第二个最低点索引
    previous_low_price = float(recent_data.loc[previous_low_idx, 'low'])
    current_macd_hist = float(recent_data['MACD_hist'].iloc[-1])
    previous_low_macd_hist = float(recent_data.loc[previous_low_idx, 'MACD_hist'])

    # 找到最近N期内的价格最低点及其索引
    abs_low_idx = recent_data['low'].idxmin()
    abs_low_price = float(recent_data.loc[abs_low_idx, 'low'])
    abs_low_macd_hist = float(recent_data.loc[abs_low_idx, 'MACD_hist'])

    # 条件1: 近期价格整体是下跌或探底趋势 (用简单斜率近似)
    price_trend = np.polyfit(range(len(recent_data)), recent_data['low'].astype(float), 1)[0] < 0

    # 条件2: 当前价格接近或低于前低 (形态要求)
    price_lower_or_near = current_low_price <= previous_low_price * 1.01 # 允许小幅误差

    # 条件3: MACD低点抬高 (背驰核心) - 比较最近两个显著低点
    macd_higher_low = current_macd_hist > previous_low_macd_hist

    # 条件4: 当前MACD柱仍在0轴下方或附近 (底背驰通常发生在0轴下)
    macd_below_zero = current_macd_hist < 0.1 # 允许在0轴附近

    # 条件5: 确保绝对最低点的MACD也参与比较 (防止中间小反弹误判)
    # 如果当前不是绝对最低点，比较当前和绝对最低点
    macd_higher_than_abs_low = True
    if abs_low_idx != recent_data.index[-1]:
         macd_higher_than_abs_low = current_macd_hist > abs_low_macd_hist and current_low_price <= abs_low_price * 1.01

    # 综合判断 (这些条件可以组合优化)
    if price_trend and price_lower_or_near and macd_higher_low and macd_below_zero and macd_higher_than_abs_low:
        # 进一步检查：背驰段的 DIF 低点是否也抬高 (更严格)
        previous_low_dif = float(recent_data.loc[previous_low_idx, 'DIF'])
        current_dif = float(recent_data['DIF'].iloc[-1])
        dif_higher_low = current_dif > previous_low_dif
        if dif_higher_low:
           print(f"    - {df.index[-1].date()}: Potential MACD Bottom Divergence (Price Low: {current_low_price:.2f} vs {previous_low_price:.2f}, MACD Hist Low: {current_macd_hist:.2f} vs {previous_low_macd_hist:.2f})")
           return True

    return False


def check_down_consolidation_down_pattern(df, lookback=LOOKBACK_PERIOD_PATTERN):
    """
    检查是否符合 "下跌+盘整+下跌" 模式，并在第二次下跌末期寻找潜在第一类买点信号 (简化)
    参考：缠论16课中小资金买卖法[^9]。
    实现难点：准确识别"盘整"段。这里用波动率或价格区间简化判断。
    """
    if df.empty or len(df) < lookback:
        return False

    # 确保所有数值列都是float类型
    for col in ['high', 'low']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # 这是一个非常简化的逻辑，实际识别需要复杂的形态分析
    # 粗略思路：
    # 1. 找到最近 lookback 周期内的最高点 H 和最低点 L。
    # 2. 大致划分三个阶段：初期下跌（从 H 开始），中期（相对平稳或反弹），近期下跌（接近或创新低 L）。
    # 3. 检查近期下跌段是否出现类似 MACD 底背驰的信号。

    recent_data = df.iloc[-lookback:]
    if len(recent_data) < 30: # 需要足够数据划分阶段
        return False

    high_idx = recent_data['high'].idxmax()
    low_idx = recent_data['low'].idxmin()
    high_price = float(recent_data.loc[high_idx, 'high'])
    low_price = float(recent_data.loc[low_idx, 'low'])

    # 必须是近期创了新低
    if low_idx != recent_data.index[-1] and low_idx != recent_data.index[-2]:
         return False

    # 必须是从一个相对高点下来的
    if high_idx > low_idx or (high_price - low_price) / high_price < 0.15: # 跌幅至少15%
        return False

    # 尝试在第二次下跌的末尾（即近期）寻找底背驰信号
    # 使用之前的底背驰函数，但作用在整个 lookback 周期上可能更合适
    # 注意：这里的 lookback 周期需要足够长以包含整个模式
    # 并且，需要确认当前确实处于第二次下跌的末端区域

    # 简化：如果近期创了 lookback 周期内的新低，并且同时满足MACD底背驰条件
    is_recent_low = low_idx == recent_data.index[-1] or low_idx == recent_data.index[-2]

    if is_recent_low:
        # 调用MACD底背驰检查，观察窗口可设为第二次下跌的大致期间，或整个lookback周期
        if check_macd_bottom_divergence(df, lookback=lookback // 2): # 用后半段观察背驰
             print(f"    - {df.index[-1].date()}: Potential 'Down+Consolidation+Down' end with divergence signal.")
             return True

    return False


def check_ma_break_retest(df, ma_period=YEARLY_MA_PERIOD):
    """
    检查是否放量突破年线后缩量回调年线附近 (简化)
    参考：缠论第7课提到的牛市选股思路之一[^5]。
    """
    if df.empty or f'MA_{ma_period}' not in df.columns or len(df) < ma_period + 10: # 需要MA计算后有几天观察
        return False

    # 确保所有数值列都是float类型
    numeric_columns = ['close', 'low', 'high', f'MA_{ma_period}', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(float)

    data = df.iloc[-10:] # 观察最近10天
    if len(data) < 3: return False

    current_close = float(data['close'].iloc[-1])
    current_low = float(data['low'].iloc[-1])
    current_ma = float(data[f'MA_{ma_period}'].iloc[-1])
    prev_close = float(data['close'].iloc[-2])
    prev_ma = float(data[f'MA_{ma_period}'].iloc[-2])
    avg_volume_10 = float(data['volume'].iloc[-10:].mean())
    current_volume = float(data['volume'].iloc[-1])

    # 条件1: 当前收盘价在年线上方或非常接近年线
    close_above_ma = current_close >= current_ma * 0.99

    # 条件2: 前一日收盘价也在年线上方或刚突破
    prev_close_above_ma = prev_close >= prev_ma

    # 条件3: 近期有过突破年线的动作（例如10天内最低点低于年线，但现在高于）
    recent_low_below = float(data['low'].iloc[:-1].min()) < float(data[f'MA_{ma_period}'].iloc[:-1].min()) * 1.01

    # 条件4: 当前处于回调状态（价格比前几日低）且缩量
    is_pullback = current_close < float(data['high'].iloc[-5:-1].max()) # 比近几日高点低
    is_low_volume = current_volume < avg_volume_10 * 0.8 # 成交量低于10日均量的80%

    # 条件5: 当前最低价接近或触及年线但未有效跌破
    low_near_ma = current_low >= current_ma * 0.98 and current_low < current_ma * 1.03

    # 综合判断 (简化)
    if close_above_ma and prev_close_above_ma and recent_low_below and is_pullback and is_low_volume and low_near_ma:
        print(f"    - {df.index[-1].date()}: Potential MA({ma_period}) Break & Retest (Close: {current_close:.2f}, MA: {current_ma:.2f}, Volume: {current_volume:.0f} vs Avg: {avg_volume_10:.0f})")
        return True

    return False


# --- 主程序 ---
def chanlun_select_stock():
    print("开始执行缠论思路选股...")
    all_codes = get_all_stock_codes()
    if not all_codes:
        print("未能获取股票列表，程序退出。")
        exit()

    print(f"数据库中共有 {len(all_codes)} 只股票。")

    potential_buys = {
        "macd_divergence": [],
        "down_consol_down": [],
        "ma_retest": []
    }

    today = date.today()
    # 获取足够长的历史数据用于计算指标和模式识别
    start_date = today - timedelta(days=LOOKBACK_PERIOD_PATTERN + MACD_SLOW + 100)

    processed_count = 0
    for code in all_codes:
        df_daily = get_stock_data(code, start_date, today)

        if len(df_daily) < LOOKBACK_PERIOD_PATTERN:
            continue

        # 计算指标
        df_daily = calculate_macd(df_daily)
        df_daily = calculate_ma(df_daily, YEARLY_MA_PERIOD)

        if df_daily.empty:
            continue

        # 应用选股逻辑
        try:
            if check_macd_bottom_divergence(df_daily.copy()): # 使用copy避免函数内修改影响其他检查
                potential_buys["macd_divergence"].append(code)

            if check_down_consolidation_down_pattern(df_daily.copy()):
                 potential_buys["down_consol_down"].append(code)

            if check_ma_break_retest(df_daily.copy()):
                 potential_buys["ma_retest"].append(code)

        except Exception as e:
             print(f"处理股票 {code} 时发生错误: {e}") # 打印错误，继续处理下一只

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"已处理 {processed_count} / {len(all_codes)} 只股票...")

    print("\n--- 选股结果 ---")
    print(f"基于MACD底背驰 (可能的第一类买点信号):")
    print(potential_buys["macd_divergence"] if potential_buys["macd_divergence"] else "无")

    print(f"\n基于'下跌+盘整+下跌'模式末端背驰信号:")
    print(potential_buys["down_consol_down"] if potential_buys["down_consol_down"] else "无")

    print(f"\n基于突破年线后缩量回踩年线:")
    print(potential_buys["ma_retest"] if potential_buys["ma_retest"] else "无")

    print("\n选股完成。请注意：以上结果仅为基于简化缠论思路的初步筛选，不构成投资建议。请结合图形和缠论原文进行深入分析。")

if __name__ == "__main__":
    chanlun_select_stock()


