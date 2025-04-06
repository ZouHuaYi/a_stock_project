import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.utils.db_manager import DatabaseManager

class StockAnalyzer:
    def __init__(self):
        """初始化股票分析器"""
        self.db_manager = DatabaseManager()

    def fetch_stock_basic(self) -> pd.DataFrame:
        """获取股票基本信息"""
        sql = "SELECT stock_code, stock_name FROM stock_basic"
        result = self.db_manager.execute_sql(sql)
        if result:
            df = pd.DataFrame(result.fetchall(), columns=['stock_code', 'stock_name'])
            return df
        return pd.DataFrame()

    def fetch_daily_data(self, stock_code: str, days: int = 120) -> pd.DataFrame:
        """获取股票日线数据"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        sql = f"""
            SELECT trade_date, open, high, low, close, volume, amount 
            FROM stock_daily 
            WHERE stock_code = '{stock_code}' AND trade_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY trade_date
        """
        result = self.db_manager.execute_sql(sql)
        if result:
            df = pd.DataFrame(result.fetchall(), columns=['trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
            # 确保数值列为float类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            # 设置日期索引
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            return df
        return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 均线系统
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma21'] = df['close'].rolling(window=21).mean()
        df['ma65'] = df['close'].rolling(window=65).mean()
        
        # 量能指标
        df['vol_ma5'] = df['volume'].rolling(window=5).mean()
        df['vol_ma21'] = df['volume'].rolling(window=21).mean()
        
        # 黄金分割目标位计算 [^1]
        low_idx = df['close'].idxmin()
        high_idx = df.loc[low_idx:, 'close'].idxmax()
        base_low = df.at[low_idx, 'close']
        base_high = df.at[high_idx, 'close']
        
        df['target_1.618'] = (base_high - base_low) * 1.618 + base_low
        df['target_2.618'] = (base_high - base_low) * 2.618 + base_low
        
        # OBV能量潮 [^1]
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # VR指标 [^3]
        up_volume = df[df['close'] > df['close'].shift(1)]['volume']
        down_volume = df[df['close'] < df['close'].shift(1)]['volume']
        up_sum = up_volume.rolling(window=21).sum()
        down_sum = down_volume.rolling(window=21).sum()
        df['vr'] = (up_sum + 0.5 * (df['volume'].rolling(window=21).sum() - up_sum - down_sum)) / \
                   (down_sum + 0.5 * (df['volume'].rolling(window=21).sum() - up_sum - down_sum)) * 100
        
        return df

    def analyze_volume_pattern(self, df: pd.DataFrame) -> List[str]:
        """分析量价形态"""
        signals = []
        last_5 = df.iloc[-5:]
        
        # 顺势量判断 [^6]
        up_days = last_5[last_5['close'] > last_5['close'].shift(1)]
        down_days = last_5[last_5['close'] < last_5['close'].shift(1)]
        
        if len(up_days) > 0 and len(down_days) > 0:
            up_vol_avg = up_days['volume'].mean()
            down_vol_avg = down_days['volume'].mean()
            
            if up_vol_avg > 1.8 * down_vol_avg:
                signals.append("健康顺势量")
            elif down_vol_avg > up_vol_avg:
                signals.append("警惕逆势量")
        
        # 量能潮分析 [^1]
        if df['obv'].iloc[-1] > df['obv'].iloc[-21] and df['close'].iloc[-1] > df['close'].iloc[-21]:
            signals.append("OBV与价格同步上升")
        
        # 凹洞量识别 [^2]
        if (df['volume'].iloc[-1] < df['vol_ma21'].iloc[-1] * 0.6 and 
            df['close'].iloc[-1] > df['close'].iloc[-2]):
            signals.append("凹洞量止跌信号")
            
        return signals

    def detect_main_force_action(self, df: pd.DataFrame) -> List[str]:
        """识别主力行为"""
        signals = []
        
        # 破底型进货识别 [^3]
        if (df['close'].iloc[-1] > df['close'].iloc[-5] and 
            df['volume'].iloc[-5] > df['vol_ma21'].iloc[-5] * 2):
            signals.append("破底型进货嫌疑")
        
        # 盘跌型出货识别 [^6]
        if (df['close'].iloc[-3] < df['close'].iloc[-6] and 
            df['close'].iloc[-1] < df['close'].iloc[-3] and
            df['volume'].iloc[-1] > df['vol_ma21'].iloc[-1]):
            signals.append("盘跌型出货嫌疑")
        
        # 头部量识别 [^4]
        if (df['close'].iloc[-1] > df['target_2.618'].iloc[-1] * 0.95 and
            df['volume'].iloc[-1] > df['vol_ma21'].iloc[-1] * 2):
            signals.append("目标位满足+异常大量")
            
        return signals

    def generate_trading_signals(self, df: pd.DataFrame) -> Dict:
        """生成交易信号"""
        signals = {
            'volume_pattern': self.analyze_volume_pattern(df),
            'main_force': self.detect_main_force_action(df),
            'technical': []
        }
        
        # 技术指标信号
        if df['ma5'].iloc[-1] > df['ma21'].iloc[-1] and df['ma5'].iloc[-2] <= df['ma21'].iloc[-2]:
            signals['technical'].append("5日均线上穿21日均线")
        
        if df['vr'].iloc[-1] < 40:
            signals['technical'].append(f"VR指标低位({df['vr'].iloc[-1]:.1f})")
            
        if df['close'].iloc[-1] > df['target_1.618'].iloc[-1]:
            signals['technical'].append(f"突破1.618目标位({df['target_1.618'].iloc[-1]:.2f})")
        
        return signals

    def evaluate_stock(self, stock_code: str) -> Optional[Dict]:
        """评估单只股票"""
        try:
            df = self.fetch_daily_data(stock_code)
            if len(df) < 65:  # 数据不足
                return None
                
            df = self.calculate_technical_indicators(df)
            signals = self.generate_trading_signals(df)
            
            # 综合评估
            score = 0
            positive_signals = []
            warning_signals = []
            
            for signal in signals['volume_pattern']:
                if signal in ["健康顺势量", "OBV与价格同步上升", "凹洞量止跌信号"]:
                    score += 1
                    positive_signals.append(signal)
                else:
                    score -= 1
                    warning_signals.append(signal)
                    
            for signal in signals['main_force']:
                if "进货" in signal:
                    score += 2
                    positive_signals.append(signal)
                else:
                    score -= 2
                    warning_signals.append(signal)
                    
            for signal in signals['technical']:
                if "上穿" in signal or "低位" in signal:
                    score += 1
                    positive_signals.append(signal)
                elif "目标位" in signal:
                    score -= 1
                    warning_signals.append(signal)
            
            return {
                'stock_code': stock_code,
                'current_price': df['close'].iloc[-1],
                'ma21': df['ma21'].iloc[-1],
                'target_1.618': df['target_1.618'].iloc[-1],
                'vr': df['vr'].iloc[-1],
                'score': score,
                'positive_signals': " | ".join(positive_signals),
                'warnings': " | ".join(warning_signals) if warning_signals else "无"
            }
            
        except Exception as e:
            print(f"分析{stock_code}时出错: {str(e)}")
            return None

    def run_screening(self):
        """执行选股流程"""
        print("开始执行量能选股分析...")
        stocks = self.fetch_stock_basic()
        if stocks.empty:
            print("未能获取股票列表，程序退出。")
            return
            
        print(f"数据库中共有 {len(stocks)} 只股票。")
        results = []
        
        processed_count = 0
        for _, row in stocks.iterrows():
            stock_code = row['stock_code']
            stock_name = row['stock_name']
            
            result = self.evaluate_stock(stock_code)
            if result and result['score'] > 2:  # 只保留评分>2的股票
                result['stock_name'] = stock_name
                results.append(result)
                
                # 打印选股信号
                print(f"    - {stock_code} {stock_name}: 评分 {result['score']} | {result['positive_signals']}")
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} / {len(stocks)} 只股票...")
        
        if results:
            # 按评分排序
            results.sort(key=lambda x: x['score'], reverse=True)
            df = pd.DataFrame(results)
            
            print("\n--- 量能选股结果 ---")
            print(f"共选出 {len(results)} 只符合条件的股票:")
            print(df[['stock_code', 'stock_name', 'current_price', 'score', 
                     'positive_signals', 'warnings']].to_string(index=False))
        else:
            print("没有找到符合条件的股票")
            
        # 关闭数据库连接
        self.db_manager.close()

if __name__ == '__main__':
    analyzer = StockAnalyzer()
    analyzer.run_screening()
