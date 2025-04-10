import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import os
from datetime import datetime, timedelta

class VolPriceAnalysis:
    """量价关系分析类，用于识别股票的洗盘、拉升等特征"""
    
    def __init__(self, stock_code, stock_name=None, end_date=None, days=60, save_path="./datas"):
        """
        初始化量价关系分析器
        
        参数:
            stock_code (str): 股票代码
            stock_name (str, 可选): 股票名称，如不提供则使用股票代码
            end_date (str 或 datetime, 可选): 结束日期，默认为当前日期
            days (int, 可选): 回溯天数，默认60天
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
            
            # 打印列名，用于调试
            print(f"获取的数据列名: {self.stock_hist_df.columns.tolist()}")
                
            print("数据获取成功！")
            return True
            
        except Exception as e:
            print(f"获取股票数据时出错: {e}")
            return False
            
    def prepare_data(self):
        """准备数据，计算各种指标"""
        if self.stock_hist_df is None:
            return False
        
        # 打印原始列名
        print(f"原始列名: {self.stock_hist_df.columns.tolist()}")
        
        # 标准化列名 - 适配不同版本akshare可能返回的不同列名
        column_mappings = {
            # 可能的日期列名
            '日期': 'date', 'date': 'date', '时间': 'date', 
            # 可能的价格列名
            '开盘': 'open', '开盘价': 'open', 'open': 'open',
            '收盘': 'close', '收盘价': 'close', 'close': 'close',
            '最高': 'high', '最高价': 'high', 'high': 'high',
            '最低': 'low', '最低价': 'low', 'low': 'low',
            # 可能的成交量列名
            '成交量': 'volume', '成交额': 'amount', 'volume': 'volume', 'vol': 'volume'
        }
        
        # 检查并重命名列
        renamed_columns = {}
        for old_col, new_col in column_mappings.items():
            if old_col in self.stock_hist_df.columns:
                renamed_columns[old_col] = new_col
        
        # 重命名列
        if renamed_columns:
            self.stock_hist_df = self.stock_hist_df.rename(columns=renamed_columns)
            print(f"列已重命名: {renamed_columns}")
        
        # 检查必要的列是否存在
        required_columns = ['open', 'close', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in self.stock_hist_df.columns]
        
        if missing_columns:
            print(f"错误: 缺少必要的列: {missing_columns}")
            print(f"现有列: {self.stock_hist_df.columns.tolist()}")
            return False
        
        # 确保日期列存在或创建日期列
        if '日期' in self.stock_hist_df.columns:
            self.stock_hist_df = self.stock_hist_df.rename(columns={'日期': 'date'})
        
        # 转换日期并设置为索引
        if 'date' in self.stock_hist_df.columns:
            self.stock_hist_df['date'] = pd.to_datetime(self.stock_hist_df['date'])
            self.stock_hist_df.set_index('date', inplace=True)
        else:
            # 如果没有日期列，尝试使用当前索引或创建日期索引
            if not isinstance(self.stock_hist_df.index, pd.DatetimeIndex):
                print("警告: 无法找到日期列，使用自动递增索引")
                # 创建日期范围作为索引
                end_date = self.end_date
                start_date = end_date - timedelta(days=len(self.stock_hist_df) - 1)
                date_range = pd.date_range(start=start_date, end=end_date, periods=len(self.stock_hist_df))
                self.stock_hist_df.index = date_range
        
        # 打印最终的列和索引类型
        print(f"处理后列名: {self.stock_hist_df.columns.tolist()}")
        print(f"索引类型: {type(self.stock_hist_df.index)}")
        
        # 确保数据类型正确
        for col in ['open', 'close', 'high', 'low', 'volume']:
            if col in self.stock_hist_df.columns:
                self.stock_hist_df[col] = self.stock_hist_df[col].astype(float)
        
        # 按日期排序
        self.stock_hist_df.sort_index(inplace=True)
        
        # 计算移动平均线
        self.stock_hist_df['MA5'] = self.stock_hist_df['close'].rolling(window=5).mean()
        self.stock_hist_df['MA10'] = self.stock_hist_df['close'].rolling(window=10).mean()
        self.stock_hist_df['MA20'] = self.stock_hist_df['close'].rolling(window=20).mean()
        self.stock_hist_df['MA30'] = self.stock_hist_df['close'].rolling(window=30).mean()
        
        # 计算成交量移动平均
        self.stock_hist_df['VOL_MA5'] = self.stock_hist_df['volume'].rolling(window=5).mean()
        self.stock_hist_df['VOL_MA10'] = self.stock_hist_df['volume'].rolling(window=10).mean()
        
        # 计算量比（当日成交量/5日平均成交量）
        self.stock_hist_df['volume_ratio'] = self.stock_hist_df['volume'] / self.stock_hist_df['VOL_MA5']
        
        # 计算涨跌幅
        self.stock_hist_df['price_change_pct'] = self.stock_hist_df['close'].pct_change() * 100
        
        return True
    
    def detect_wash_patterns(self, window=15):
        """
        检测洗盘特征
        
        参数:
            window (int): 检查的时间窗口，默认15天
            
        返回:
            list: 洗盘特征列表，包含特征类型和位置
        """
        if self.stock_hist_df is None or len(self.stock_hist_df) < window:
            print("数据不足，无法检测洗盘特征")
            return []
        
        # 复制最后window天的数据进行分析
        recent_data = self.stock_hist_df.iloc[-window:].copy()
        patterns = []
        
        # 检测特征1：成交量显著萎缩
        vol_ratio = recent_data['volume'].mean() / self.stock_hist_df['volume'].iloc[-(window*2):-window].mean()
        if vol_ratio < 0.7:  # 成交量萎缩到之前的70%以下
            patterns.append({
                'type': '成交量萎缩',
                'ratio': vol_ratio,
                'description': f'成交量萎缩至前期的{vol_ratio:.2f}倍'
            })
        
        # 检测特征2：价格在下跌过程中企稳（震荡但守住关键位）
        price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['low'].min()
        price_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        # 价格波动但整体趋势较小
        if price_range > 0.05 and abs(price_trend) < 0.03:
            patterns.append({
                'type': '横盘震荡',
                'range': price_range,
                'trend': price_trend,
                'description': f'价格振幅{price_range:.2%}，整体趋势{price_trend:.2%}'
            })
        
        # 检测特征3：出现较长的下影线（洗盘后的拉升信号）
        recent_shadows = []
        for i in range(min(5, len(recent_data))):
            row = recent_data.iloc[-(i+1)]
            lower_shadow = (row['close'] - row['low']) / row['close'] if row['close'] > row['open'] else (row['open'] - row['low']) / row['open']
            if lower_shadow > 0.02:  # 下影线超过2%
                recent_shadows.append({
                    'date': recent_data.index[-(i+1)],
                    'shadow_ratio': lower_shadow,
                    'description': f'{recent_data.index[-(i+1)].strftime("%Y-%m-%d")}出现{lower_shadow:.2%}的下影线'
                })
        
        if recent_shadows:
            patterns.append({
                'type': '下影线信号',
                'shadows': recent_shadows,
                'description': f'近期出现{len(recent_shadows)}次明显下影线'
            })
        
        # 检测特征4：量价关系 - 缩量下跌或盘整
        vol_decrease = (recent_data['volume'].iloc[-1] < recent_data['VOL_MA5'].iloc[-1])
        price_steady = abs(recent_data['price_change_pct'].iloc[-5:].mean()) < 1.0  # 近5天平均涨跌幅小于1%
        
        if vol_decrease and price_steady:
            patterns.append({
                'type': '缩量企稳',
                'vol_ratio': recent_data['volume'].iloc[-1] / recent_data['VOL_MA5'].iloc[-1],
                'description': f'成交量低于5日均量，价格趋于稳定'
            })
        
        return patterns
    
    def analyze_vol_price(self):
        """
        分析股票的量价关系，检测洗盘特征
        
        返回:
            dict: 分析结果
        """
        if not self.prepare_data():
            return {'status': 'error', 'message': '数据准备失败'}
        
        # 获取最近数据
        last_data = self.stock_hist_df.iloc[-1]
        patterns = self.detect_wash_patterns()
        
        # 判断是否可能处于洗盘阶段
        is_washing = False
        wash_confidence = 0
        
        # 根据检测到的特征判断洗盘可能性
        if patterns:
            wash_features = [p['type'] for p in patterns]
            if '成交量萎缩' in wash_features:
                wash_confidence += 40
            if '横盘震荡' in wash_features:
                wash_confidence += 30
            if '下影线信号' in wash_features:
                wash_confidence += 20
            if '缩量企稳' in wash_features:
                wash_confidence += 10
            
            is_washing = wash_confidence >= 50
        
        # 形成分析结论
        analysis_result = {
            'status': 'success',
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'date': self.end_date.strftime('%Y-%m-%d'),
            'is_washing': is_washing,
            'wash_confidence': wash_confidence,
            'patterns': patterns,
            'latest_price': last_data['close'],
            'latest_volume': last_data['volume'],
            'volume_ratio': last_data['volume_ratio'] if 'volume_ratio' in last_data else None,
            'ma5': last_data['MA5'],
            'ma20': last_data['MA20'],
            'description': self.generate_analysis_description(is_washing, wash_confidence, patterns)
        }
        
        return analysis_result
    
    def generate_analysis_description(self, is_washing, confidence, patterns):
        """
        生成分析描述
        
        参数:
            is_washing (bool): 是否可能在洗盘
            confidence (int): 洗盘可能性置信度
            patterns (list): 检测到的特征
            
        返回:
            str: 分析描述
        """
        if not patterns:
            return f"{self.stock_name}({self.stock_code})目前未检测到明显的洗盘特征。"
        
        if is_washing:
            desc = f"{self.stock_name}({self.stock_code})可能正处于洗盘阶段（置信度：{confidence}%），表现出以下特征：\n"
        else:
            desc = f"{self.stock_name}({self.stock_code})出现了一些洗盘相关特征，但尚不足以确认（置信度：{confidence}%）：\n"
        
        for pattern in patterns:
            desc += f"- {pattern['description']}\n"
        
        if confidence >= 70:
            desc += "\n综合判断：很可能是主力洗盘行为，可关注成交量企稳放大的突破信号。"
        elif confidence >= 50:
            desc += "\n综合判断：有一定的洗盘迹象，需继续观察量价变化来确认。"
        else:
            desc += "\n综合判断：洗盘特征不明显，可能是正常的调整或盘整。"
            
        return desc
    
    def plot_vol_price_chart(self, save_filename=None):
        """
        绘制量价关系图，突出显示洗盘特征
        
        参数:
            save_filename (str, 可选): 保存的文件名，默认为股票代码_洗盘分析_日期.png
        """
        if self.stock_hist_df is None or self.stock_hist_df.empty:
            print("无数据可绘制。请先获取数据。")
            return False

        if save_filename is None:
            save_filename = f"{self.stock_code}_洗盘分析_{self.end_date.strftime('%Y%m%d')}.png"
        save_path = os.path.join(self.save_path, save_filename)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 尝试再次进行量价分析
        try:
            analysis_result = self.analyze_vol_price()
            if analysis_result['status'] != 'success':
                print(f"分析失败，无法绘制图表: {analysis_result.get('message', '未知错误')}")
                return False
            patterns = analysis_result.get('patterns', [])
        except Exception as e:
            print(f"分析过程出错: {e}")
            # 使用空模式列表继续绘图
            patterns = []
            analysis_result = {
                'status': 'success',
                'is_washing': False,
                'wash_confidence': 0,
                'description': f"{self.stock_name}({self.stock_code})分析出错，无法显示结果。"
            }
        
        # 创建一个包含两个子图的Figure(主图和成交量图)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        title = f'{self.stock_name}({self.stock_code}) 量价关系分析'
        if analysis_result.get('is_washing', False):
            title += f" - 洗盘可能性: {analysis_result.get('wash_confidence', 0)}%"
        fig.suptitle(title, fontsize=16)
        
        # 绘制价格和均线
        ax1.plot(self.stock_hist_df.index, self.stock_hist_df['close'], label='收盘价', linewidth=2)
        ax1.plot(self.stock_hist_df.index, self.stock_hist_df['MA5'], label='MA5', linewidth=1)
        ax1.plot(self.stock_hist_df.index, self.stock_hist_df['MA10'], label='MA10', linewidth=1)
        ax1.plot(self.stock_hist_df.index, self.stock_hist_df['MA20'], label='MA20', linewidth=1)
        ax1.plot(self.stock_hist_df.index, self.stock_hist_df['MA30'], label='MA30', linewidth=1)
        
        # 绘制K线图 - 使用更安全的方式处理日期
        for i, (idx, row) in enumerate(self.stock_hist_df.iterrows()):
            # 确定K线颜色（涨为红，跌为绿）
            color = 'red' if row['close'] >= row['open'] else 'green'
            # 绘制影线
            ax1.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1)
            # 绘制实体
            body_bottom = row['open'] if row['close'] >= row['open'] else row['close']
            body_height = abs(row['close'] - row['open'])
            # 使用更安全的方式处理日期转换
            try:
                if isinstance(idx, pd.Timestamp):
                    x_pos = idx
                else:
                    x_pos = pd.Timestamp(idx)
                rect = Rectangle((x_pos, body_bottom), timedelta(days=0.6), body_height, 
                             facecolor=color, edgecolor=color, alpha=0.7)
                ax1.add_patch(rect)
            except Exception as e:
                print(f"绘制K线实体时出错: {e}, 索引: {idx}, 类型: {type(idx)}")
                # 继续处理下一个K线
                continue
        
        # 标记洗盘特征
        if patterns:
            for pattern in patterns:
                if pattern['type'] == '下影线信号':
                    for shadow in pattern.get('shadows', []):
                        date = shadow.get('date')
                        if date and date in self.stock_hist_df.index:
                            ax1.annotate('下影线', xy=(date, self.stock_hist_df.loc[date, 'low']),
                                     xytext=(date, self.stock_hist_df.loc[date, 'low'] * 0.95),
                                     arrowprops=dict(facecolor='blue', shrink=0.05),
                                     horizontalalignment='center', verticalalignment='top')
        
        # 设置主图标题和标签
        ax1.set_title('价格走势', fontsize=14)
        ax1.set_ylabel('价格', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制成交量
        last_n_days = min(30, len(self.stock_hist_df))  # 只显示最近30天的数据以便更清晰地观察
        if len(self.stock_hist_df) > last_n_days:
            plot_df = self.stock_hist_df.iloc[-last_n_days:]
        else:
            plot_df = self.stock_hist_df
            
        for idx, row in plot_df.iterrows():
            # 成交量颜色和K线一致
            color = 'red' if row['close'] >= row['open'] else 'green'
            ax2.bar(idx, row['volume'], color=color, alpha=0.7, width=0.8)
        
        # 绘制成交量均线
        ax2.plot(plot_df.index, plot_df['VOL_MA5'], color='blue', linewidth=1, label='VOL_MA5')
        ax2.plot(plot_df.index, plot_df['VOL_MA10'], color='orange', linewidth=1, label='VOL_MA10')
        
        # 标记成交量萎缩
        if patterns and any(p['type'] == '成交量萎缩' for p in patterns):
            # 使用更安全的方式获取数据点
            try:
                mid_point = len(plot_df) // 2
                ax2.annotate('成交量萎缩区域', 
                            xy=(plot_df.index[mid_point], plot_df['volume'].mean()),
                            xytext=(plot_df.index[mid_point], plot_df['volume'].max() * 0.8),
                            arrowprops=dict(facecolor='purple', shrink=0.05),
                            horizontalalignment='center', verticalalignment='top')
            except Exception as e:
                print(f"标记成交量萎缩区域时出错: {e}")
        
        # 设置成交量图标题和标签
        ax2.set_title('成交量分析', fontsize=14)
        ax2.set_ylabel('成交量', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 调整x轴日期格式
        plt.xticks(rotation=45)
        
        # 如果索引是DatetimeIndex，使用日期格式化器
        if isinstance(self.stock_hist_df.index, pd.DatetimeIndex):
            date_format = mdates.DateFormatter('%Y-%m-%d')
            ax2.xaxis.set_major_formatter(date_format)
            # 设置适当的日期定位器
            ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
        
        # 添加分析结论
        desc = analysis_result.get('description', f"无法获取{self.stock_name}({self.stock_code})的分析结论")
        plt.figtext(0.5, 0.01, desc, ha='center', fontsize=12, 
                  bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # 为底部文字留出空间
        
        try:
            # 保存图表
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"量价关系分析图已保存至: {save_path}")
            plt.close(fig)  # 关闭图形，释放内存
            return True
        except Exception as e:
            print(f"保存图表失败: {e}")
            return False
    
    def run_analysis(self, save_chart=True):
        """
        运行完整分析流程
        
        参数:
            save_chart (bool, 可选): 是否保存分析图表，默认为True
            
        返回:
            dict: 分析结果
        """
        if not self.fetch_data():
            return {'status': 'error', 'message': '获取数据失败'}
        
        # 先尝试准备数据
        if not self.prepare_data():
            return {'status': 'error', 'message': '数据准备失败', 'patterns': []}
        
        # 分析量价关系
        analysis_result = self.analyze_vol_price()
        
        # 如果分析成功且需要保存图表，则绘制图表
        if analysis_result['status'] == 'success' and save_chart:
            try:
                self.plot_vol_price_chart()
            except Exception as e:
                print(f"绘制图表失败: {e}")
                analysis_result['chart_error'] = str(e)
        
        return analysis_result


# 使用示例
if __name__ == "__main__":
    analyzer = VolPriceAnalysis(stock_code="000001", stock_name="平安银行")
    result = analyzer.run_analysis(save_chart=True)
    print(result['description'])
