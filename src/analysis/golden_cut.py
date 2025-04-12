import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import matplotlib.dates as mdates  # 用于标注
from matplotlib.patches import Rectangle
import os

# 导入公共工具函数
from utils.indicators import calculate_fibonacci_levels, plot_stock_chart


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
        self.swing_low_date = None
        self.swing_high_date = None
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
        
        # 重命名列以与其他函数兼容
        name_map = {
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume'
        }
        self.stock_hist_df = self.stock_hist_df.rename(columns=name_map)
        
        return True
        
    def calculate_fibonacci_levels(self):
        """识别主要波段并计算斐波那契回调位"""
        if self.stock_hist_df is None:
            return False
        
        # 使用共享函数计算斐波那契水平，使用波段模式
        self.fib_levels = calculate_fibonacci_levels(self.stock_hist_df, use_swing=True)
        
        if not self.fib_levels:
            print("\n未能计算有效的斐波那契回调水平")
            return False
        
        # 找到波段的起点和终点供绘图标记使用
        # 这里简单实现，实际项目中可能需要更复杂的波段识别逻辑
        try:
            self.swing_low_date = self.stock_hist_df['low'].idxmin()
            high_df = self.stock_hist_df.loc[self.swing_low_date:]
            self.swing_high_date = high_df['high'].idxmax()
            
            # 准备标注信息
            self.plot_annotations = [
                {
                    'xy': (self.swing_low_date, self.stock_hist_df.loc[self.swing_low_date, 'low']), 
                    'text': f'波段低点\n{self.stock_hist_df.loc[self.swing_low_date, "low"]:.2f}', 
                    'xytext': (-60, -30)
                },
                {
                    'xy': (self.swing_high_date, self.stock_hist_df.loc[self.swing_high_date, 'high']), 
                    'text': f'波段高点\n{self.stock_hist_df.loc[self.swing_high_date, "high"]:.2f}', 
                    'xytext': (10, 20)
                }
            ]
            
            print("\n计算得到的斐波那契回调位:")
            for label, price in self.fib_levels.items():
                print(f"  {label}: {price:.2f}")
                
            return True
        except Exception as e:
            print(f"计算波段端点时出错: {e}")
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
        
        try:
            # 使用通用绘图函数
            title = f'{self.stock_name} ({self.stock_code}) 日线图与斐波那契回调'
            
            # 调用共享绘图函数
            fig, axes = plot_stock_chart(
                self.stock_hist_df,
                title=title,
                save_path=save_path,
                plot_ma=True,
                plot_volume=True,
                plot_fib=self.fib_levels
            )
            
            # 添加波段起止点标注（如果有的话）
            if fig and len(axes) > 0 and self.plot_annotations:
                ax1 = axes[0]
                for ann in self.plot_annotations:
                    date = ann['xy'][0]
                    price = ann['xy'][1]
                    ax1.annotate(
                        ann['text'],
                        xy=(date, price),
                        xytext=(ann['xytext'][0]/10, ann['xytext'][1]/10),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2'),
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
                    )
                
                # 保存更新后的图表
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            print(f"图表已保存到: {save_path}")
            return True
            
        except Exception as e:
            print(f"绘制图表时出错: {e}")
            return False
        
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