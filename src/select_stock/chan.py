import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from src.utils.db_manager import DatabaseManager

class DailyChanSystem:
    def __init__(self):
        # 初始化数据库管理器
        self.db_manager = DatabaseManager()
        
        # 日线专用参数（根据缠论日线操作标准设置）
        self.params = {
            'pivot_window': 5,       # 中枢识别窗口[^2]
            'macd_fast': 12,         # MACD快线周期[^7]
            'macd_slow': 26,         # MACD慢线周期[^7]
            'vol_ratio': 1.5,        # 量能放大阈值[^10]
            'min_vol': 1e6           # 最小成交量（股）
        }

    # ========== 数据获取模块 ==========
    def get_daily_data(self, stock_code, start_date, end_date):
        """获取日线数据（包含复权价格）"""
        sql = f"""
        SELECT trade_date as date, open, high, low, close, volume 
        FROM stock_daily 
        WHERE stock_code='{stock_code}' AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
        """
        result = self.db_manager.execute_sql(sql)
        if result:
            df = pd.DataFrame(result.fetchall(), columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            # 确保数值列为float类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            # 设置日期索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        return pd.DataFrame()

    # ========== 核心分析模块 ==========
    def analyze_stock(self, stock_code, days=250):
        """主分析函数（默认分析250个交易日）"""
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = self.get_daily_data(stock_code, start_date, end_date)
        if len(df) < 30:  # 至少需要30个交易日数据
            return None
        
        # 特征计算
        df = self._calculate_features(df)
        
        # 中枢识别（日线级别）[^2][^8]
        df['pivots'] = self._identify_pivots(df)
        
        # 买卖信号检测
        signals = self._generate_signals(df)
        
        return {
            'stock_code': stock_code,
            'last_close': df.iloc[-1]['close'],
            'signals': signals,
            'trend_strength': self._evaluate_trend(df)
        }

    def _calculate_features(self, df):
        """计算技术指标"""
        # MACD指标[^7]
        exp12 = df['close'].ewm(span=self.params['macd_fast'], adjust=False).mean()
        exp26 = df['close'].ewm(span=self.params['macd_slow'], adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 均线系统（5/13/21/34/55日）[^2]
        for ma in [5, 13, 21, 34, 55]:
            df[f'ma{ma}'] = df['close'].rolling(ma).mean()
            
        # 量能指标
        df['vol_ma5'] = df['volume'].rolling(5).mean()
        return df

    def _identify_pivots(self, df):
        """识别日线中枢[^8]"""
        pivots = pd.Series(0, index=df.index)
        n = self.params['pivot_window']
        
        # 找局部高低点
        highs = argrelextrema(df['high'].values, np.greater_equal, order=n)[0]
        lows = argrelextrema(df['low'].values, np.less_equal, order=n)[0]
        
        # 标记中枢区域（简化版三线段重合）
        for i in range(2, len(df)-2):
            if (i in highs and i-1 in lows and i+1 in lows) or \
               (i in lows and i-1 in highs and i+1 in highs):
                pivot_range = df.iloc[i-1:i+2]
                if (pivot_range['high'].max() - pivot_range['low'].min()) < df['close'].mean() * 0.1:  # 振幅小于10%
                    pivots.iloc[i] = 1
        return pivots

    # ========== 信号生成模块 ==========
    def _generate_signals(self, df):
        """生成买卖信号[^7][^10]"""
        signals = []
        
        # 第一类买点：趋势背驰[^7]
        for i in range(2, len(df)-1):
            if df['pivots'].iloc[i] and \
               df['macd'].iloc[i] < 0 and df['signal'].iloc[i] < 0 and \
               df['macd'].iloc[i] > df['macd'].iloc[i-1] and \
               df['volume'].iloc[i] > df['vol_ma5'].iloc[i] * self.params['vol_ratio']:
                signals.append({
                    'date': df.index[i].strftime('%Y-%m-%d'),
                    'type': 'buy1',
                    'price': df['close'].iloc[i],
                    'reason': '日线底背驰+量能放大'
                })
        
        # 第二类买点：回调不破前低[^2]
        for i in range(3, len(df)):
            if df['close'].iloc[i] > df['ma21'].iloc[i] and \
               df['close'].iloc[i-1] < df['ma21'].iloc[i-1] and \
               df['low'].iloc[i] > df['low'].iloc[i-2]:
                signals.append({
                    'date': df.index[i].strftime('%Y-%m-%d'),
                    'type': 'buy2',
                    'price': df['close'].iloc[i],
                    'reason': '21日均线回踩不破前低'
                })
        
        return signals[-3:] if signals else []  # 返回最近3个信号

    # ========== 趋势评估模块 ==========
    def _evaluate_trend(self, df):
        """评估趋势强度（基于均线系统）[^2]"""
        last = df.iloc[-1]
        ma_rank = sum(last['close'] > last[[f'ma{ma}' for ma in [5,13,21,34,55]]])
        
        if last['close'] > last['ma55']:
            return '强势' if ma_rank >=4 else '震荡偏强'
        elif last['close'] > last['ma21']:
            return '弱势反弹' if ma_rank >=3 else '下跌中继'
        else:
            return '极弱'

    # ========== 执行入口 ==========
    def screen_stocks(self, industry=None, min_signal=1):
        """执行选股流程"""
        print("开始执行缠论日线选股分析...")
        
        if industry:
            sql = f"SELECT stock_code, stock_name FROM stock_basic WHERE industry='{industry}'"
        else:
            sql = "SELECT stock_code, stock_name FROM stock_basic"
            
        result = self.db_manager.execute_sql(sql)
        if not result:
            print("未能获取股票列表，程序退出。")
            return []
            
        stocks = pd.DataFrame(result.fetchall(), columns=['stock_code', 'stock_name'])
        print(f"数据库中共有 {len(stocks)} 只股票。")
        
        results = []
        processed_count = 0
        
        for _, row in stocks.iterrows():
            code = row['stock_code']
            name = row['stock_name']
            
            analysis = self.analyze_stock(code)
            if analysis and len(analysis['signals']) >= min_signal:
                analysis['stock_name'] = name
                results.append(analysis)
                
                # 打印选股信号
                latest_signal = analysis['signals'][-1] if analysis['signals'] else None
                if latest_signal:
                    print(f"    - {code} {name}: {latest_signal['type']} | 价格:{latest_signal['price']:.2f} | 趋势:{analysis['trend_strength']}")
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} / {len(stocks)} 只股票...")
                
        # 关闭数据库连接
        self.db_manager.close()
        
        # 按趋势强度排序
        return sorted(results, key=lambda x: (
            x['trend_strength'] in ['强势', '震荡偏强'],
            len(x['signals']),
            x['last_close']
        ), reverse=True)

    def run_screening(self, industry=None, min_signal=1):
        """执行选股流程"""
        print("开始执行缠论日线选股分析...")
        
        # 获取所有股票代码
        results = self.screen_stocks(industry, min_signal)
        if results:
            print("\n--- 缠论日线选股结果 ---")
            
        # 创建美观的表格输出
        table_data = []
        # results 中 latest_signal 中的最大值
        max_signal_date = max(results, key=lambda x: x['signals'][-1]['date'])['signals'][-1]['date']
        for stock in results:
            latest_signal = stock['signals'][-1] if stock['signals'] else None
            # 输出是程序运行的日期  是最近5个交易日范围
            if latest_signal:
                # 只保留最新的信号日期
                if latest_signal['date'] == max_signal_date:
                    table_data.append({
                        '股票代码': stock['stock_code'],
                        '股票名称': stock['stock_name'],
                        '信号类型': latest_signal['type'],
                        '信号价格': f"{latest_signal['price']:.2f}",
                        '信号日期': latest_signal['date'],
                        '趋势强度': stock['trend_strength'],
                        '当前价格': f"{stock['last_close']:.2f}"
                    })
        
        # 转换为DataFrame并打印
        if table_data:
            print(f"共选出 {len(table_data)} 只符合条件的股票:")
            df = pd.DataFrame(table_data)
            print(df.to_string(index=False))
        else:
            print("没有找到符合条件的股票")
        
# ========== 使用示例 ==========
if __name__ == "__main__":
    system = DailyChanSystem()
    
    # 示例1：全市场选股
    system.run_screening()
    
   
