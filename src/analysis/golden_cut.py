import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import matplotlib.dates as mdates  # 用于标注
from matplotlib.patches import Rectangle
import os


class FibonacciAnalysis:
    """斐波那契回调分析类，用于计算和可视化股票的斐波那契回调水平"""
    
    def __init__(self, stock_code, stock_name=None, end_date=None, days=365, save_path="./datas"):
        """
        初始化斐波那契分析器
        
        参数:
            stock_code (str): 股票代码
            stock_name (str, 可选): 股票名称，如不提供则使用股票代码
            end_date (str 或 datetime, 可选): 结束日期，默认为当前日期
            days (int, 可选): 回溯天数，默认365天
            save_path (str, 可选): 图片保存路径，默认当前目录
        """
        self.stock_code = stock_code
        self.stock_name = stock_name if stock_name else stock_code
        # 处理end_date参数，支持字符串或datetime对象
        if end_date:
            if isinstance(end_date, str):
                # 尝试将字符串转换为datetime对象
                try:
                    self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
                except ValueError:
                    print(f"警告: 日期格式错误 '{end_date}'，使用当前日期代替")
                    self.end_date = datetime.now()
            else:
                # 假设是datetime对象
                self.end_date = end_date
        else:
            self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=days)
        self.end_date_str = self.end_date.strftime('%Y%m%d')
        self.start_date_str = self.start_date.strftime('%Y%m%d')
        self.stock_hist_df = None
        self.fib_levels = {}
        self.plot_annotations = []
        self.save_path = save_path
        # 创建保存目录(如果不存在)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
    def fetch_data(self):
        """从AkShare获取股票历史数据"""
        print(f"正在获取 {self.stock_name} ({self.stock_code}) 从 {self.start_date_str} 到 {self.end_date_str} 的日线数据...")
        
        try:
            self.stock_hist_df = ak.stock_zh_a_hist(
                symbol=self.stock_code, 
                period="daily",
                start_date=self.start_date_str, 
                end_date=self.end_date_str,
                adjust="qfq"  # 使用前复权数据
            )
            
            if self.stock_hist_df.empty:
                print(f"未能获取到股票 {self.stock_code} 的数据，请检查代码或网络。")
                return False
                
            print("数据获取成功！")
            return True
            
        except Exception as e:
            print(f"获取股票数据时出错: {e}")
            return False
            
    def prepare_data(self):
        """准备数据，适配matplotlib格式"""
        if self.stock_hist_df is None:
            return False
            
        # 转换日期
        self.stock_hist_df['日期'] = pd.to_datetime(self.stock_hist_df['日期'])
        self.stock_hist_df.set_index('日期', inplace=True)
        
        # 确保数据类型正确
        for col in ['开盘', '最高', '最低', '收盘', '成交量']:
            if col in self.stock_hist_df.columns:
                self.stock_hist_df[col] = self.stock_hist_df[col].astype(float)
        
        # 按日期排序
        self.stock_hist_df.sort_index(inplace=True)
        return True
        
    def calculate_fibonacci_levels(self):
        """识别主要波段并计算斐波那契回调位"""
        if self.stock_hist_df is None:
            return False
            
        # 找到整个时间段内的最低点和最高点
        swing_low_date = self.stock_hist_df['最低'].idxmin()
        swing_high_date = self.stock_hist_df['最高'].idxmax()
        
        # 确保找到的是一个有效的波段（低点在高点之前，构成一个上升浪）
        if swing_low_date < swing_high_date:
            swing_low_price = self.stock_hist_df.loc[swing_low_date, '最低']
            swing_high_price = self.stock_hist_df.loc[swing_high_date, '最高']
            price_diff = swing_high_price - swing_low_price
            
            if price_diff > 0:  # 确保价格有差异
                print(f"\n识别到主要上升波段:")
                print(f"  起点 (Low): {swing_low_date.strftime('%Y-%m-%d')} @ {swing_low_price:.2f}")
                print(f"  终点 (High): {swing_high_date.strftime('%Y-%m-%d')} @ {swing_high_price:.2f}")
                
                # 计算斐波那契回调水平
                self.fib_levels = {
                    'Fib 0.0% (High)': swing_high_price,
                    'Fib 38.2%': swing_high_price - price_diff * 0.382,
                    'Fib 50.0%': swing_high_price - price_diff * 0.5,
                    'Fib 61.8%': swing_high_price - price_diff * 0.618,
                    'Fib 100% (Low)': swing_low_price,
                    'Fib 161.8%': swing_high_price + price_diff * 0.618,
                    'Fib 200%': swing_high_price + price_diff * 1,
                    'Fib 261.8%': swing_high_price + price_diff * 1.618
                }
                
                # 准备标注信息
                self.plot_annotations = [
                    {
                        'xy': (swing_low_date, swing_low_price), 
                        'text': f'Swing Low\n{swing_low_price:.2f}', 
                        'xytext': (-60, -30)
                    },
                    {
                        'xy': (swing_high_date, swing_high_price), 
                        'text': f'Swing High\n{swing_high_price:.2f}', 
                        'xytext': (10, 20)
                    }
                ]
                
                print("\n计算得到的斐波那契回调位:")
                for label, price in self.fib_levels.items():
                    print(f"  {label}: {price:.2f}")
                    
                return True
            else:
                print("\n未能识别到有效的价格波动范围来计算斐波那契水平。")
        else:
            print(f"\n注意：时段内最高点 ({swing_high_date.strftime('%Y-%m-%d')}) 出现在最低点 ({swing_low_date.strftime('%Y-%m-%d')}) 之前。")
            print("这可能是一个下降趋势。如需分析反弹，请调整斐波那契计算逻辑。")
            print("当前将不绘制斐波那契线。")
            
        return False
        
    def plot_chart(self, save_filename=None):
        """使用matplotlib绘制K线图和斐波那契水平并保存为PNG文件
        
        参数:
            save_filename (str, 可选): 保存的文件名，默认为股票代码_日期.png
        """
        if self.stock_hist_df is None:
            print("无数据可绘制。请先获取数据。")
            return False

        if save_filename is None:
            save_filename = f"{self.stock_code}_{self.end_date.strftime('%Y%m%d')}.png"
        save_path = os.path.join(self.save_path, save_filename)
        
        print("\n正在生成图表...")
        
        # 创建一个包含两个子图的Figure(主图和成交量图)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        fig.suptitle(f'{self.stock_name} ({self.stock_code}) 日线图与斐波那契回调', fontsize=16)
        
        # 绘制K线图
        dates = self.stock_hist_df.index
        opens = self.stock_hist_df['开盘']
        highs = self.stock_hist_df['最高']
        lows = self.stock_hist_df['最低']
        closes = self.stock_hist_df['收盘']
        volumes = self.stock_hist_df['成交量']
        
        # 设置x轴为日期格式
        date_ticks = np.linspace(0, len(dates) - 1, min(10, len(dates)))
        date_labels = [dates[int(idx)].strftime('%Y-%m-%d') for idx in date_ticks]
        
        # 绘制K线
        width = 0.6  # K线宽度
        offset = width / 2.0
        
        # K线绘制逻辑
        for i in range(len(dates)):
            # 价格上涨用红色，下跌用绿色(中国市场风格)
            if closes[i] >= opens[i]:
                color = 'red'
                body_height = closes[i] - opens[i]
                body_bottom = opens[i]
            else:
                color = 'green'
                body_height = opens[i] - closes[i]
                body_bottom = closes[i]
            
            # 绘制影线
            ax1.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1)
            
            # 绘制实体
            if body_height == 0:  # 开盘=收盘的情况
                body_height = 0.001  # 赋予一个极小值，以便能够显示
            rect = Rectangle((i - offset, body_bottom), width, body_height, 
                             facecolor=color, edgecolor=color)
            ax1.add_patch(rect)
        
        # 绘制移动平均线
        ma5 = self.stock_hist_df['收盘'].rolling(window=5).mean()
        ma20 = self.stock_hist_df['收盘'].rolling(window=20).mean()
        ma60 = self.stock_hist_df['收盘'].rolling(window=60).mean()
        
        x = np.arange(len(dates))
        ax1.plot(x, ma5, 'blue', linewidth=1, label='MA5')
        ax1.plot(x, ma20, 'orange', linewidth=1, label='MA20')
        ax1.plot(x, ma60, 'purple', linewidth=1, label='MA60')
        
        # 绘制斐波那契回调线
        if self.fib_levels:
            fib_colors = {
                'Fib 38.2%': 'orange',
                'Fib 50.0%': 'yellowgreen',
                'Fib 61.8%': 'green',
                'Fib 100% (Low)': 'lightblue',
                'Fib 161.8%': 'red',
                'Fib 200%': 'blue',
                'Fib 261.8%': 'purple'
            }
            
            for level, price in self.fib_levels.items():
                if level in fib_colors:
                    ax1.axhline(y=price, color=fib_colors[level], linestyle='--', linewidth=1)
                    # 添加标签
                    ax1.text(len(dates) - 1, price, f"{level} ({price:.2f})", 
                            color=fib_colors[level], verticalalignment='center')
        
        # 绘制成交量
        for i in range(len(dates)):
            # 成交量颜色和K线一致，上涨为红，下跌为绿
            if closes[i] >= opens[i]:
                color = 'red'
            else:
                color = 'green'
            ax2.bar(i, volumes[i], width=width, color=color, alpha=0.7)
        
        # 添加网格线
        ax1.grid(True, linestyle=':', alpha=0.3)
        ax2.grid(True, linestyle=':', alpha=0.3)
        
        # 设置轴标签
        ax1.set_ylabel('价格 (前复权)')
        ax2.set_ylabel('成交量')
        
        # 设置x轴刻度和标签
        plt.xticks(date_ticks, date_labels, rotation=45)
        plt.tight_layout()
        
        # 添加波段起止点标注
        if self.plot_annotations:
            # 找到日期对应的索引位置
            for ann in self.plot_annotations:
                date = ann['xy'][0]
                price = ann['xy'][1]
                date_idx = self.stock_hist_df.index.get_loc(date)
                ax1.annotate(
                    ann['text'],
                    xy=(date_idx, price),
                    xytext=(date_idx + ann['xytext'][0]/10, price + ann['xytext'][1]/10),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2'),
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
                )
        
        # 添加图例
        ax1.legend(loc='upper left')
        
        # 保存图表
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
        plt.close(fig)  # 关闭图表释放内存
        return True
        
    def run_analysis(self, save_filename=None):
        """运行完整的分析流程"""
        if not self.fetch_data():
            return False
            
        if not self.prepare_data():
            return False
            
        self.calculate_fibonacci_levels()
        self.plot_chart(save_filename)
        
        print("\n分析完成。")
        return True


# 示例用法
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = FibonacciAnalysis(stock_code="000001", stock_name="平安银行", save_path="./datas")
    
    # 运行分析
    analyzer.run_analysis()