# chan_analyzer.py
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator # 用于更好的坐标轴刻度
import matplotlib.dates as mdates # 用于日期格式化
from matplotlib.patches import Rectangle
from typing import List, Dict, Optional, Tuple
import logging
import os # 用于创建目录和保存图片

# -- Matplotlib 全局设置 --
# 设置中文显示 - 确保系统中已安装 SimHei 字体，或更换为其他可用中文字体
# 例如: 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei'
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except Exception as e:
    logging.warning(f"设置中文字体失败: {e}. 图表标签可能显示不正确。请确保安装了SimHei字体或更换为其他可用中文字体。")

# -- 日志设置 --
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -- Pandas 设置 --
pd.options.mode.chained_assignment = None # 默认为 'warn'

# -- 缠论绘图颜色定义 --
chan_colors = {
    'kline_up': '#FF3333',       # 阳线(红)
    'kline_down': '#33AA33',     # 阴线(绿)
    'volume_up': '#FF6666',      # 成交量-阳
    'volume_down': '#66CC66',    # 成交量-阴
    'fx_top': '#D62728',         # 顶分型(红)
    'fx_bottom': '#2CA02C',      # 底分型(绿)
    'bi_up': '#FF4500',          # 向上笔(橙红)
    'bi_down': '#32CD32',        # 向下笔(亮绿)
    'xd_up': '#FF0000',          # 向上线段(红) - 可设置更粗/不同样式
    'xd_down': '#008000',        # 向下线段(绿) - 可设置更粗/不同样式
    'zs_bg': (0.7, 0.9, 0.9, 0.35),  # 中枢背景(粉蓝, 带透明度)
    'zs_border': '#4682B4',      # 中枢边框(钢蓝)
    'zs_text': '#191970',        # 中枢文字(午夜蓝)
    'grid': '#DCDCDC',           # 网格线(淡灰)
    'text': '#333333',           # 普通文字(深灰)
    'background': '#F5F5F5',     # 背景(白烟)
    'title': '#000080'           # 标题(海军蓝)
}

class ChanAnalyzer:
    """
    对股票/指数数据进行简化的缠论分析，跨越多个时间周期。
    重点是识别和绘制 K线、分型、笔和中枢（中心）。
    【注意】本实现中的缠论元素识别（分型、笔、线段、中枢）是高度简化的，
           仅用于演示目的，并未严格遵循所有缠论规则（如包含关系处理、特征序列、严格的笔/段定义等）。
    """

    def __init__(self,
                 symbol: str = "sz000001",  # 默认为平安银行
                 periods: List[str] = ['30min', '5min', '1min'], # 关注请求的分钟级别 + 基础
                 end_date: Optional[str] = None,
                 data_len_min: int = 800): # *最低*频率（如1分钟）的K线数量
        """
        初始化缠论分析器。

        参数:
            symbol (str): 股票代码 (必须包含市场前缀，如 sz000001 或 sh600519)。
                          注意：当前版本仅支持个股，不支持指数。
            periods (List[str]): 要分析的时间周期列表 (例如, '30min', '5min', '1min')。
                                 如果需要其他分钟图表，则应包含 '1min'。
            end_date (Optional[str]): 历史数据的结束日期 (YYYYMMDD)。默认为最新日期。
            data_len_min (int): *1分钟*时间周期的大约数据点数量。
        """
        # 检查股票代码格式
        if not symbol.startswith(('sh', 'sz')):
            raise ValueError("股票代码必须以'sh'或'sz'开头，如 'sh600519'或'sz000001'")
        
        self.symbol = symbol
        _periods = list(periods)
        if any(p in ['5min', '15min', '30min', '60min'] for p in _periods) and '1min' not in _periods:
             _periods.append('1min')
             logging.warning("已将 '1min' 添加到周期列表，因为需要它来重采样。")
            
        self.periods_all = sorted(list(set(_periods)), key=lambda p: {'1min':1, '5min':5, '15min':15, '30min':30, '60min':60, 'daily':1440}.get(p, 9999))
        self.periods_requested = sorted(list(set(periods)), key=lambda p: {'1min':1, '5min':5, '15min':15, '30min':30, '60min':60, 'daily':1440}.get(p, 9999))
        self.end_date = end_date if end_date else pd.Timestamp.now().strftime('%Y%m%d')
        self.data_len_min = data_len_min

        self.data: Dict[str, pd.DataFrame] = {}
        self.fenxing: Dict[str, pd.DataFrame] = {p: pd.DataFrame() for p in self.periods_all}
        self.bi: Dict[str, List[Dict]] = {p: [] for p in self.periods_all}
        self.xianduan: Dict[str, List[Dict]] = {p: [] for p in self.periods_all}
        self.zhongshu: Dict[str, List[Dict]] = {p: [] for p in self.periods_all}

        # 创建保存图片的目录
        self.plot_save_dir = f"./{self.symbol}_chanlun_plots"
        os.makedirs(self.plot_save_dir, exist_ok=True)
        logging.info(f"图表将保存在目录: {self.plot_save_dir}")

        logging.info(f"为 {self.symbol} 初始化缠论分析器...")
        self._load_all_data()
        if not self.data or not any(p in self.data for p in self.periods_requested):
             logging.error(f"未能加载任何请求的周期 ({self.periods_requested}) 的数据。分析无法进行。")
        else:
            loaded_periods = sorted([p for p in self.periods_requested if p in self.data])
            logging.info(f"数据成功加载，用于分析的周期: {loaded_periods}")
            self.periods_requested = loaded_periods


    def _fetch_minute_data(self, symbol_to_fetch: str, period: str = '1min') -> Optional[pd.DataFrame]:
        """使用 AKShare 获取分钟数据。"""
        logging.info(f"正在获取 {symbol_to_fetch} 的 {period} 数据...")
        ak_period = period.replace('min', '')
        if ak_period not in ['1', '5', '15', '30', '60']:
            logging.error(f"AKShare 不支持的分钟周期: {period}")
            return None
        
        try:
            # 提取正确的股票代码格式
            # 检查是否有前缀如"sh"或"sz"
            if symbol_to_fetch.startswith(('sh', 'sz')):
                market = 'sh' if symbol_to_fetch.startswith('sh') else 'sz'
                symbol_code = symbol_to_fetch[2:]  # 移除前缀
            else:
                # 如果没有前缀，默认为上证
                market = 'sh' if symbol_to_fetch.startswith('0') else 'sz'
                symbol_code = symbol_to_fetch
            
            logging.info(f"使用代码 {symbol_code} 和市场 {market} 获取数据")
            
            # 使用akshare获取数据
            df = ak.stock_zh_a_hist_min_em(symbol=symbol_code, period=ak_period, 
                                          end_date=self.end_date, adjust='qfq')

            if df is None or df.empty:
                logging.warning(f"未能获取到 {symbol_to_fetch} 的 {period} 数据。")
                return None

            df['datetime'] = pd.to_datetime(df['时间'])
            df.set_index('datetime', inplace=True)
            df = df[['开盘', '最高', '最低', '收盘', '成交量']].astype(float)
            df.rename(columns={'开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}, inplace=True)
            df.sort_index(inplace=True)
            required_len = self.data_len_min if period == '1min' else max(150, self.data_len_min // int(ak_period))
            df = df.iloc[-required_len:]

            if df.empty:
                logging.warning(f"限制长度后，{period} 的数据帧为空。")
                return None
                
            logging.info(f"成功获取并处理了 {len(df)} 行 {period} 数据。")
            return df
        except Exception as e:
            logging.error(f"获取 {symbol_to_fetch} 的 {period} 数据时出错: {e}")
            return None

    def _resample_data(self, base_df: pd.DataFrame, target_freq_pd: str) -> Optional[pd.DataFrame]:
        """将高频数据重采样至低频。"""
        logging.info(f"正在将数据重采样到目标频率 {target_freq_pd}...")
        if base_df is None or base_df.empty or not isinstance(base_df.index, pd.DatetimeIndex):
             logging.error("基础 DataFrame 无效，无法重采样。")
             return None
        try:
            ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            resampled_df = base_df.resample(target_freq_pd, label='right', closed='right').agg(ohlc_dict)
            resampled_df.dropna(subset=['open', 'high', 'low', 'close'], how='all', inplace=True)
            resampled_df['volume'] = resampled_df['volume'].fillna(0)
            if resampled_df.empty:
                logging.warning(f"重采样到 {target_freq_pd} 结果为空 DataFrame。")
                return None
            logging.info(f"重采样成功，为 {target_freq_pd} 生成了 {len(resampled_df)} 条K线。")
            return resampled_df
        except Exception as e:
             logging.error(f"将数据重采样到 {target_freq_pd} 时出错: {e}")
             return None

    def _load_all_data(self):
        """加载所有需要周期的数据。"""
        base_minute_df = None
        
        # 直接使用完整的原始代码
        symbol_code = self.symbol

        if any('min' in p for p in self.periods_all):
             logging.info("尝试加载1分钟基础数据...")
             base_minute_df = self._fetch_minute_data(symbol_code, period='1min')
             if base_minute_df is not None and '1min' in self.periods_all:
                 self.data['1min'] = base_minute_df
             elif base_minute_df is None:
                 logging.warning("未能加载1分钟数据。")

        target_freq_map = {'5min':'5min', '15min':'15min', '30min':'30min', '60min':'60min'}
        for period in self.periods_all:
            if period in self.data: continue
            if period in target_freq_map:
                target_freq_pd = target_freq_map[period]
                if base_minute_df is not None:
                    resampled_df = self._resample_data(base_minute_df, target_freq_pd)
                    if resampled_df is not None: self.data[period] = resampled_df
                    else: logging.warning(f"无法通过重采样生成 {period} 数据。")
                else:
                     logging.warning(f"1分钟基础数据缺失。尝试直接获取 {period} 数据。")
                     direct_fetch = self._fetch_minute_data(symbol_code, period=period)
                     if direct_fetch is not None: self.data[period] = direct_fetch
                     else: logging.error(f"获取或重采样 {period} 数据失败。")
            elif period == 'daily':
                 logging.warning("日线数据处理暂未实现。")
            else:
                logging.warning(f"周期 '{period}' 当前不支持。")

        original_all = self.periods_all[:]
        self.periods_all = [p for p in original_all if p in self.data and not self.data[p].empty]
        if len(self.periods_all) < len(original_all):
            missing = set(original_all) - set(self.periods_all)
            logging.warning(f"以下周期的数据未能成功加载: {missing}")


    def _detect_fenxing(self, df: pd.DataFrame) -> pd.DataFrame:
        """简化版分型检测 (忽略包含关系)。"""
        logging.debug(f"运行简化版分型检测...")
        fx = pd.DataFrame(index=df.index)
        fx['fx_high'] = np.nan
        fx['fx_low'] = np.nan
        fx['fx_type'] = ''

        for i in range(1, len(df) - 1):
            k_prev, k_curr, k_next = df.iloc[i-1], df.iloc[i], df.iloc[i+1]
            
            # 放宽分型检测条件，只要当前K线的高点比邻近K线高即可
            is_potential_top = (k_curr['high'] > k_prev['high'] and k_curr['high'] > k_next['high'])
            
            # 放宽分型检测条件，只要当前K线的低点比邻近K线低即可
            is_potential_bottom = (k_curr['low'] < k_prev['low'] and k_curr['low'] < k_next['low'])
            
            if is_potential_top:
                fx.loc[df.index[i], 'fx_high'] = k_curr['high']
                fx.loc[df.index[i], 'fx_type'] = 'top'
            elif is_potential_bottom:
                fx.loc[df.index[i], 'fx_low'] = k_curr['low']
                fx.loc[df.index[i], 'fx_type'] = 'bottom'

        # 简化版过滤：处理连续同类型分型 (保留更好的) 和 确保类型交替
        fx_points_list = []
        for dt, row in fx.iterrows():
            if pd.notna(row['fx_type']) and row['fx_type'] != '':
                k_index = df.index.get_loc(dt) # 获取K线索引
                fx_points_list.append({'datetime': dt, 'type': row['fx_type'],
                                     'price': row['fx_high'] if row['fx_type'] == 'top' else row['fx_low'],
                                     'k_index': k_index})

        if not fx_points_list: return fx # 没有检测到分型点

        # 排序确保按时间（和索引）顺序处理
        fx_points_list.sort(key=lambda x: x['k_index'])

        processed_fx = []
        if fx_points_list:
             processed_fx.append(fx_points_list[0]) # 保留第一个

        for i in range(1, len(fx_points_list)):
            current_fx = fx_points_list[i]
            last_processed_fx = processed_fx[-1]

            # 确保分型之间至少间隔1根K线 
            if current_fx['k_index'] - last_processed_fx['k_index'] > 1:
                # 类型不同，直接添加
                if current_fx['type'] != last_processed_fx['type']:
                    processed_fx.append(current_fx)
                else:
                    # 类型相同，保留 "更好" 的那个 (更高的顶，更低的底)
                    if current_fx['type'] == 'top' and current_fx['price'] > last_processed_fx['price']:
                        processed_fx[-1] = current_fx # 替换
                    elif current_fx['type'] == 'bottom' and current_fx['price'] < last_processed_fx['price']:
                        processed_fx[-1] = current_fx # 替换

        # 从处理后的列表重新构建DataFrame
        final_fx = pd.DataFrame(index=df.index)
        final_fx['fx_high'] = np.nan
        final_fx['fx_low'] = np.nan
        final_fx['fx_type'] = ''
        for p in processed_fx:
            if p['type'] == 'top':
                 final_fx.loc[p['datetime'], 'fx_high'] = p['price']
            else:
                 final_fx.loc[p['datetime'], 'fx_low'] = p['price']
            final_fx.loc[p['datetime'], 'fx_type'] = p['type']

        logging.debug(f"简化版分型检测完成。找到 {len(processed_fx)} 个过滤后的分型点。")
        return final_fx


    def _detect_bi(self, df_with_fx: pd.DataFrame, level: str) -> List[Dict]:
        """简化版笔检测 (基于过滤后的分型)。【注意：高度简化】"""
        logging.debug(f"运行简化版笔检测，周期 {level}...")
        bi_list = []
        fx_points_df = df_with_fx[df_with_fx['fx_type'].isin(['top', 'bottom'])].reset_index()
        fx_points = fx_points_df.to_dict('records') # 转换为字典列表

        if len(fx_points) < 2:
            logging.debug("分型点不足，无法形成笔。")
            return []

        # 确保分型点的类型交替 - 这一步保证了首尾有效的分型对
        filtered_fx_points = []
        if len(fx_points) > 0:
            filtered_fx_points.append(fx_points[0])
            current_type = fx_points[0]['fx_type']
            
            for i in range(1, len(fx_points)):
                if fx_points[i]['fx_type'] != current_type:
                    filtered_fx_points.append(fx_points[i])
                    current_type = fx_points[i]['fx_type']

        # 特殊处理：如果交替筛选后分型点太少，则放宽条件使用原始分型点
        if len(filtered_fx_points) < 2:
            logging.debug("交替筛选后分型点不足，使用原始分型点形成笔。")
            filtered_fx_points = fx_points
            
        # 形成笔
        for i in range(len(filtered_fx_points) - 1):
            start_fx = filtered_fx_points[i]
            end_fx = filtered_fx_points[i+1]

            # 确保类型交替
            if start_fx['fx_type'] == end_fx['fx_type']: 
                continue

            start_price = start_fx['fx_high'] if start_fx['fx_type'] == 'top' else start_fx['fx_low']
            end_price = end_fx['fx_high'] if end_fx['fx_type'] == 'top' else end_fx['fx_low']

            # 创建笔，放宽有效性检查
            bi_type = 'up' if start_fx['fx_type'] == 'bottom' else 'down'
            
            # 获取笔区间内的K线，用于计算最高/最低价
            try:
                bi_kline_range = self.data[level].loc[start_fx['datetime']:end_fx['datetime']]
                if bi_kline_range.empty: continue # 如果切片为空则跳过
                bi_high = bi_kline_range['high'].max()
                bi_low = bi_kline_range['low'].min()
            except KeyError:
                logging.warning(f"无法找到笔的时间范围 [{start_fx['datetime']} to {end_fx['datetime']}]，跳过此笔。")
                continue

            # 确保笔中至少有3根K线
            k_count = len(bi_kline_range)
            if k_count < 3:
                continue

            bi = {
                'start_dt': start_fx['datetime'],
                'end_dt': end_fx['datetime'],
                'start_price': start_price,
                'end_price': end_price,
                'high': bi_high, # 笔区间内的最高价
                'low': bi_low,   # 笔区间内的最低价
                'type': bi_type,
                'level': level,
                'k_count_approx': k_count # 近似的K线计数
            }
            bi_list.append(bi)

        logging.debug(f"简化版笔检测完成。找到 {len(bi_list)} 笔。")
        return bi_list


    def _detect_xianduan(self, bi_list: List[Dict], level: str) -> List[Dict]:
        """简化版线段检测 (基于笔)。【注意：非缠论标准，仅为绘图演示】"""
        logging.debug(f"运行简化版线段检测，周期 {level}...")
        # 实际线段划分需要特征序列和缺口处理，逻辑复杂。
        # 这里仅做一个非常粗略的模拟，例如，将连续方向相反的笔连接成段，
        # 或者按固定数量的笔分组，这 *不是* 缠论线段。
        # 为了绘图，我们甚至可以简单地认为每条笔都是一个线段。
        xianduan_list = []
        for bi in bi_list:
            # 最简化：直接将每条笔视为一个线段用于绘图
            xd = {
                'start_dt': bi['start_dt'],
                'end_dt': bi['end_dt'],
                'high': bi['high'],
                'low': bi['low'],
                'type': bi['type'], # 线段方向同笔方向
                'level': level,
                'num_bi': 1 # 假设一条笔构成一段
            }
            xianduan_list.append(xd)

        logging.warning(f"[{level}] 线段检测使用的是高度简化逻辑（一笔成段），仅用于绘图，不符合缠论标准！")
        logging.debug(f"简化版线段检测完成。生成 {len(xianduan_list)} 个'线段'（基于一笔成段）。")
        return xianduan_list


    def _detect_zhongshu(self, xianduan_list: List[Dict], level: str) -> List[Dict]:
        """简化版中枢检测 (基于简化的线段)。【注意：非缠论标准】"""
        logging.debug(f"运行简化版中枢检测，周期 {level}...")
        zhongshu_list = []
        if len(xianduan_list) < 3:
            logging.debug("线段数量不足 (<3)，无法形成中枢。")
            return []

        # 简化中枢识别：寻找任意连续三段，如果有重叠则认为是中枢
        for i in range(len(xianduan_list) - 2):
            try:
                xd1 = xianduan_list[i]
                xd2 = xianduan_list[i+1]
                xd3 = xianduan_list[i+2]
            except IndexError:
                break # 防止列表越界

            # 计算三段的高低点，放宽条件，只要有任何重叠就视为中枢
            highs = [xd1['high'], xd2['high'], xd3['high']]
            lows = [xd1['low'], xd2['low'], xd3['low']]
            
            try:
                # 中枢上轨：所有段中最低的高点
                zg = min(highs)
                # 中枢下轨：所有段中最高的低点
                zd = max(lows)
                # 中枢形成过程中的最高点和最低点
                gg = max(highs)
                dd = min(lows)
                
                # 只要下轨不高于上轨，就认为有重叠形成中枢
                if zg >= zd:
                    # 找到了中枢
                    start_dt = xd1['start_dt']
                    end_dt = xd3['end_dt']
                    involved_xd_indices = [i, i+1, i+2]
                    
                    zs = {
                        'start_dt': start_dt,
                        'end_dt': end_dt,
                        'zg': zg, # 中枢上轨 (高点)
                        'zd': zd, # 中枢下轨 (低点)
                        'gg': gg, # 中枢区间最高价
                        'dd': dd, # 中枢区间最低价
                        'level': level,
                        'segments_indices': involved_xd_indices,
                        'type': '标准' # 简化类型
                    }
                    zhongshu_list.append(zs)
            except Exception as e:
                logging.error(f"计算中枢重叠时出错 (索引 {i}): {e}")
                continue # 发生错误时继续下一个索引

        logging.debug(f"简化版中枢检测完成。找到 {len(zhongshu_list)} 个中枢。")
        return zhongshu_list


    # --------------------------------------------------------------------------
    # 分析与绘图
    # --------------------------------------------------------------------------

    def run_level_analysis(self, level: str):
        """运行指定周期的简化版缠论分析。"""
        if level not in self.data or self.data[level].empty:
            logging.warning(f"周期 '{level}' 无可用数据。跳过分析。")
            return

        logging.info(f"--- 开始分析周期: {level} ---")
        df = self.data[level].copy()

        # 1. 检测分型
        fx_df = self._detect_fenxing(df)
        self.fenxing[level] = fx_df # 存储包含分型信息的 DataFrame
        logging.info(f"[{level}] 分型分析完成。")

        # 2. 检测笔
        df_with_fx = df.join(fx_df[['fx_high', 'fx_low', 'fx_type']]) # 合并数据和分型信息
        self.bi[level] = self._detect_bi(df_with_fx, level)
        logging.info(f"[{level}] 笔分析完成。发现 {len(self.bi[level])} 笔 (简化)。")

        # 3. 检测线段 (简化版)
        self.xianduan[level] = self._detect_xianduan(self.bi[level], level)
        logging.info(f"[{level}] 线段分析完成。发现 {len(self.xianduan[level])} 线段 (简化)。")

        # 4. 检测中枢 (简化版)
        self.zhongshu[level] = self._detect_zhongshu(self.xianduan[level], level)
        logging.info(f"[{level}] 中枢分析完成。发现 {len(self.zhongshu[level])} 中枢 (简化)。")

        logging.info(f"--- 周期 {level} 分析结束 ---")


    def run_full_analysis(self):
        """对所有加载的、被请求分析的周期运行分析。"""
        logging.info("\n=== 开始完整的多周期分析 ===")
        for period in self.periods_requested: # 只分析请求的且已成功加载数据的周期
            if period in self.data:
                 self.run_level_analysis(period)
            else:
                 logging.warning(f"跳过周期 {period} 的分析，因为数据加载失败。")
        logging.info("=== 完整的多周期分析完成 ===\n")


    def plot_level_analysis(self, level: str, num_records: int = 200):
        """
        绘制指定周期的 K 线图和简化版缠论元素。
        中枢绘制为着色区域，不标示价格，仅标示 "ZS"。
        图表将保存到文件。
        """
        if level not in self.data or self.data[level].empty:
            logging.error(f"周期 '{level}' 无可用数据用于绘图。")
            return
        if level not in self.fenxing or level not in self.bi or level not in self.zhongshu:
             logging.warning(f"周期 '{level}' 的分析结果不完整。绘图可能不全。")

        # 选择最后 num_records 条数据用于绘图
        df_plot = self.data[level].iloc[-num_records:].copy()
        if df_plot.empty or len(df_plot) < 5: # 需要几条数据才能绘图
            logging.error(f"周期 '{level}' 的数据点过少 ({len(df_plot)})，无法绘图 (需要至少 {num_records} 条)。")
            return

        logging.info(f"--- 开始绘制周期: {level} ({len(df_plot)} 条记录) ---")

        # --- 绘图设置 ---
        plt.style.use('ggplot') # 使用 ggplot 样式
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), dpi=120, # 增加尺寸
                                       gridspec_kw={'height_ratios': [3, 1]}) # 设置子图高度比例
        
        # --- 正确绘制K线图 ---
        x_indices = np.arange(len(df_plot))
        k_width = 0.6  # K线宽度
        
        # 准备OHLC数据
        opens = df_plot['open'].values
        highs = df_plot['high'].values
        lows = df_plot['low'].values
        closes = df_plot['close'].values
        
        # 遍历数据绘制每根K线
        for i in range(len(df_plot)):
            x = x_indices[i]
            open_price = opens[i]
            high_price = highs[i]
            low_price = lows[i]
            close_price = closes[i]
            
            # 确定K线颜色 - 收盘价高于开盘价为上涨(红色)，否则为下跌(绿色)
            if close_price >= open_price:
                color = chan_colors['kline_up']
                body_bottom = open_price
                body_height = close_price - open_price
            else:
                color = chan_colors['kline_down']
                body_bottom = close_price
                body_height = open_price - close_price
            
            # 绘制K线实体
            if abs(body_height) < 1e-6:  # 开盘价等于收盘价的情况
                # 绘制一条横线
                ax1.plot([x - k_width/2, x + k_width/2], [open_price, open_price], 
                         color=color, linewidth=1.0, zorder=3)
            else:
                # 绘制矩形实体
                rect = Rectangle((x - k_width/2, body_bottom), k_width, body_height,
                                facecolor=color, edgecolor='black', linewidth=0.5, zorder=3)
                ax1.add_patch(rect)
            
            # 绘制上影线
            if high_price > max(open_price, close_price):
                ax1.plot([x, x], [max(open_price, close_price), high_price], 
                         color='black', linewidth=0.8, zorder=2)
            
            # 绘制下影线
            if low_price < min(open_price, close_price):
                ax1.plot([x, x], [min(open_price, close_price), low_price], 
                         color='black', linewidth=0.8, zorder=2)
        
        # --- 绘制成交量 ---
        for i, (dt, row) in enumerate(df_plot.iterrows()):
            x = x_indices[i]
            # 成交量颜色与K线一致：涨为红，跌为绿
            if row['close'] >= row['open']:
                color = chan_colors['volume_up']
            else:
                color = chan_colors['volume_down']
            ax2.bar(x, row['volume'], width=k_width, color=color, alpha=0.7)
        
        # --- Plotting Fenxing (on ax1) ---
        fx_plot = self.fenxing[level].loc[df_plot.index] # Filter fenxing for the plotted range
        fx_top_points = fx_plot.dropna(subset=['fx_high'])
        fx_bottom_points = fx_plot.dropna(subset=['fx_low'])
        # Map datetimes to x-indices
        dt_to_x = {dt: i for i, dt in enumerate(df_plot.index)}
        # Plot Top Fenxing Markers
        if not fx_top_points.empty:
            x_tops = [dt_to_x.get(dt) for dt in fx_top_points.index if dt_to_x.get(dt) is not None]
            y_tops = fx_top_points['fx_high'].values
            ax1.scatter(x_tops, y_tops, color=chan_colors['fx_top'], marker='v', s=60, label='顶分型', zorder=5)
        # Plot Bottom Fenxing Markers
        if not fx_bottom_points.empty:
            x_bottoms = [dt_to_x.get(dt) for dt in fx_bottom_points.index if dt_to_x.get(dt) is not None]
            y_bottoms = fx_bottom_points['fx_low'].values
            ax1.scatter(x_bottoms, y_bottoms, color=chan_colors['fx_bottom'], marker='^', s=60, label='底分型', zorder=5)
        
        # --- Plotting Bi (笔) (on ax1) ---
        plot_start_dt = df_plot.index[0]
        plot_end_dt = df_plot.index[-1]
        plotted_bi_count = 0
        for bi in self.bi.get(level, []):
            # Check if the Bi is within the plot range
            if bi['start_dt'] >= plot_start_dt and bi['end_dt'] <= plot_end_dt:
                try:
                    x_start = dt_to_x[bi['start_dt']]
                    x_end = dt_to_x[bi['end_dt']]
                    bi_color = chan_colors['bi_up'] if bi['type'] == 'up' else chan_colors['bi_down']
                    label = '笔' if plotted_bi_count == 0 else None # Only label first one
                    ax1.plot([x_start, x_end], [bi['start_price'], bi['end_price']],
                            color=bi_color, linestyle='-', linewidth=1.5, marker='.', markersize=4,
                            label=label, zorder=4)
                    plotted_bi_count += 1
                except KeyError:
                    # Skip if start/end datetime not found in the current plot's index mapping
                    logging.debug(f"Bi from {bi['start_dt']} to {bi['end_dt']} skipped (out of plot range or missing in map).")
        
        # --- Plotting Zhongshu (中枢) (on ax1) ---
        plotted_zs_count = 0
        for zs in self.zhongshu.get(level, []):
            # Filter Zhongshu within the plot's time range
            if zs['start_dt'] >= plot_start_dt and zs['end_dt'] <= plot_end_dt:
                try:
                    x_start = dt_to_x[zs['start_dt']]
                    x_end = dt_to_x[zs['end_dt']]
                    # Draw the rectangle for the Zhongshu (ZG, ZD)
                    zs_height = zs['zg'] - zs['zd']
                    if zs_height > 0: # Only draw if valid height
                        zs_rect = plt.Rectangle((x_start, zs['zd']), x_end - x_start, zs_height,
                                                facecolor=chan_colors['zs_bg'],
                                                edgecolor=chan_colors['zs_border'],
                                                linewidth=0.5, alpha=0.5, zorder=2) # Behind K-lines/Bi
                        ax1.add_patch(zs_rect)
                        # Add "ZS" text near the center of the Zhongshu rectangle
                        text_x = (x_start + x_end) / 2
                        text_y = (zs['zg'] + zs['zd']) / 2
                        label = '中枢' if plotted_zs_count == 0 else None
                        ax1.text(text_x, text_y, "ZS", color=chan_colors['zs_text'],
                                  fontsize=9, ha='center', va='center', alpha=0.9)
                        plotted_zs_count += 1
                except KeyError:
                    # Skip if start/end datetime not found
                    logging.debug(f"Zhongshu from {zs['start_dt']} to {zs['end_dt']} skipped (out of plot range or missing in map).")

        # Choose appropriate number of ticks based on data points
        num_xticks = min(10, len(x_indices) // 10) # Aim for ~10 labels max
        xtick_indices = np.linspace(0, len(x_indices) - 1, num_xticks, dtype=int)
        xtick_labels = [df_plot.index[i].strftime('%Y-%m-%d %H:%M') if level.endswith('min') else df_plot.index[i].strftime('%Y-%m-%d') for i in xtick_indices]
        ax1.set_xticks(xtick_indices)
        ax1.set_xticklabels([]) # Hide labels on top plot
        ax2.set_xticks(xtick_indices)
        ax2.set_xticklabels(xtick_labels, rotation=30, ha='right')
        # Y-axis formatting
        ax1.set_ylabel('价格', color=chan_colors['text'])
        ax2.set_ylabel('成交量', color=chan_colors['text'])
        ax1.tick_params(axis='y', labelcolor=chan_colors['text'])
        ax2.tick_params(axis='y', labelcolor=chan_colors['text'])
        # Auto-adjust Y limits with some padding
        min_low = df_plot['low'].min()
        max_high = df_plot['high'].max()
        padding = (max_high - min_low) * 0.05 # 5% padding
        ax1.set_ylim(min_low - padding, max_high + padding)
        max_volume = df_plot['volume'].max()
        ax2.set_ylim(0, max_volume * 1.1) # 10% padding above max volume
        # Grids
        ax1.grid(True, linestyle='--', linewidth=0.5, color=chan_colors['grid'])
        ax2.grid(True, linestyle='--', linewidth=0.5, color=chan_colors['grid'])
        # Title and Legend
        start_date_str = df_plot.index[0].strftime('%Y-%m-%d %H:%M')
        end_date_str = df_plot.index[-1].strftime('%Y-%m-%d %H:%M')
        fig.suptitle(f"{self.symbol} - {level} 缠论分析 ({start_date_str} to {end_date_str})",
                    fontsize=16, color=chan_colors['title'], fontweight='bold')
        # Create combined legend (handling potential missing elements)
        handles, labels = [], []
        
        # 添加K线的图例项
        handles.append(Rectangle((0, 0), 1, 1, facecolor=chan_colors['kline_up'], 
                                edgecolor='black', linewidth=0.5))
        labels.append('阳线(上涨)')
        
        handles.append(Rectangle((0, 0), 1, 1, facecolor=chan_colors['kline_down'], 
                                edgecolor='black', linewidth=0.5))
        labels.append('阴线(下跌)')
        
        # 手动添加中枢标识
        if plotted_zs_count > 0:
            handles.append(Rectangle((0, 0), 1, 1, facecolor=chan_colors['zs_bg'], 
                                    edgecolor=chan_colors['zs_border'], alpha=0.5))
            labels.append('中枢')
            
        # Collect handles/labels from ax1 plots
        h1, l1 = ax1.get_legend_handles_labels()
        handles.extend(h1)
        labels.extend(l1)
        if handles and labels:
            fig.legend(handles, labels, loc='upper right', fontsize=9)
        else:
            logging.warning(f"[{level}] 无法生成图例，因为没有可绘制的元素。")
        # --- Save and Close ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        filename = f"{self.symbol}_{level}_chan_analysis_{self.end_date}.png"
        save_path = os.path.join(self.plot_save_dir, filename)
        try:
            plt.savefig(save_path)
            logging.info(f"图表已保存到: {save_path}")
        except Exception as e:
            logging.error(f"保存图表 '{save_path}' 失败: {e}")

    def plot_all_requested_levels(self, num_records: int = 200):
        """
        为所有请求分析的周期绘制图表。
        """
        logging.info("\n=== 开始绘制所有请求周期的图表 ===")
        for period in self.periods_requested: # 只绘制请求的且已成功加载数据的周期
            if period in self.data:
                self.plot_level_analysis(period, num_records=num_records)
            else:
                logging.warning(f"跳过周期 {period} 的绘图，因为数据加载失败。")
        logging.info("=== 所有请求周期的绘图完成 ===\n")
# ==================================================
#                    示例用法
# ==================================================
if __name__ == "__main__":
    # --- 配置 ---
    symbol = "sz000001"  # 平安银行
    # 注意：必须使用正确的市场前缀：sh（上海）或sz（深圳）
    # 当前版本仅支持个股，不支持指数
    # 个股代码示例：平安银行sz000001，贵州茅台sh600519，格力电器sz000651
    
    # 请求的周期列表
    # periods_to_analyze = ['30min', '5min', '1min']
    periods_to_analyze = ['30min', '5min']  # 仅分析 30分钟和5分钟
    # 结束日期 (可选, None 表示最新)
    end_date = None  # 例如 "20231231"
    # 1分钟K线的数据量 (影响历史回溯长度)
    data_length_1min = 1000  # 获取更多数据用于更长的历史和更低频率的重采样
    # --- 执行分析 ---
    analyzer = ChanAnalyzer(symbol=symbol,
                            periods=periods_to_analyze,
                            end_date=end_date,
                            data_len_min=data_length_1min)
    # 运行所有请求周期的分析
    analyzer.run_full_analysis()
    # --- 绘制图表 ---
    # 可以为每个周期指定不同的K线数量进行绘制
    plot_kline_counts = {
        '1min': 240,  # 显示最近 4 小时 (假设1分钟K线)
        '5min': 150,  # 显示最近数天的5分钟K线
        '30min': 100  # 显示更多交易日的30分钟K线
    }
    logging.info("\n=== 开始生成图表 ===")
    for period in analyzer.periods_requested:  # 遍历实际请求并成功加载的周期
        if period in analyzer.data:
            num_to_plot = plot_kline_counts.get(period, 120)  # 使用默认值120或指定值
            analyzer.plot_level_analysis(period, num_records=num_to_plot)
        else:
            logging.warning(f"无法为周期 {period} 生成图表，因为分析数据缺失。")
    logging.info("=== 图表生成完毕 ===")