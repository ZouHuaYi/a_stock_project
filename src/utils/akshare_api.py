import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta

class AkshareAPI:
    """Akshare API 数据处理，获取股票信息"""
    def __init__(self):
        """初始化数据接口"""
        pass
    
    def get_stock_info(self, stock_code):
        """获取股票基本信息
        
        Args:
            stock_code (str): 股票代码
            
        Returns:
            dict: 股票基本信息
        """
        try:
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            info_dict = {}
            for _, row in stock_info.iterrows():
                info_dict[row['item']] = row['value']
            return info_dict
        except Exception as e:
            print(f"获取股票基本信息失败: {e}")
            return None
    
    def get_stock_name(self, stock_code):
        """获取股票名称
        
        Args:
            stock_code (str): 股票代码
            
        Returns:
            str: 股票名称
        """
        try:
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            stock_name = stock_info.loc[stock_info['item'] == '股票简称', 'value'].iloc[0]
            return stock_name
        except Exception as e:
            print(f"获取股票名称失败: {e}")
            return None
    
    def get_stock_history(self, stock_code, period="daily", years=1, adjust="qfq"):
        """获取股票历史数据
        
        Args:
            stock_code (str): 股票代码
            period (str, optional): 数据周期. Defaults to "daily".
            years (int, optional): 获取几年的数据. Defaults to 1.
            adjust (str, optional): 复权方式，"qfq"前复权，"hfq"后复权，None不复权. Defaults to "qfq".
            
        Returns:
            pandas.DataFrame: 股票历史数据
        """
        try:
            # 计算日期范围
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y%m%d')
            
            # 获取历史数据
            data = ak.stock_zh_a_hist(
                symbol=stock_code, 
                period=period, 
                start_date=start_date, 
                end_date=end_date,
                adjust=adjust
            )
            
            # 处理数据格式
            data['日期'] = pd.to_datetime(data['日期'])
            data.set_index('日期', inplace=True)
            data.rename(columns={
                '开盘': 'Open',
                '收盘': 'Close',
                '最高': 'High',
                '最低': 'Low',
                '成交量': 'Volume'
            }, inplace=True)
            
            return data
        except Exception as e:
            print(f"获取股票历史数据失败: {e}")
            return None
    
    def get_financial_reports(self, stock_code, start_year="2023"):
        """获取财务报表数据
        
        Args:
            stock_code (str): 股票代码
            start_year (str, optional): 起始年份. Defaults to "2023".
            
        Returns:
            pandas.DataFrame: 财务报表数据
        """
        try:
            financial_reports = ak.stock_financial_analysis_indicator(symbol=stock_code, start_year=start_year)
            return financial_reports
        except Exception as e:
            print(f"获取财务报表失败: {e}")
            return None
    
    def get_stock_list(self):
        """获取A股股票列表
        
        Returns:
            pandas.DataFrame: A股股票列表
        """
        try:
            stock_list = ak.stock_info_a_code_name()
            return stock_list
        except Exception as e:
            print(f"获取A股列表失败: {e}")
            return None
            
    def calculate_technical_indicators(self, df):
        """计算各种技术指标
        
        Args:
            df (pandas.DataFrame): 包含Open, High, Low, Close, Volume的股票数据
            
        Returns:
            dict: 各种技术指标
        """
        if df is None or df.empty:
            raise ValueError("没有可用的股票数据，请先获取数据")
            
        indicators = {}
        df = df.copy()
        
        # 移动平均线
        indicators['SMA_5'] = df.ta.sma(length=5, close='Close')
        indicators['SMA_10'] = df.ta.sma(length=10, close='Close') 
        indicators['SMA_20'] = df.ta.sma(length=20, close='Close')
        indicators['SMA_60'] = df.ta.sma(length=60, close='Close')
        
        # 相对强弱指数(RSI)
        indicators['RSI_6'] = df.ta.rsi(length=6, close='Close')
        indicators['RSI_12'] = df.ta.rsi(length=12, close='Close')
        
        # MACD
        macd = df.ta.macd(fast=12, slow=26, signal=9, close='Close')
        indicators['MACD'] = macd['MACD_12_26_9']
        indicators['MACD_signal'] = macd['MACDs_12_26_9']
        indicators['MACD_hist'] = macd['MACDh_12_26_9']
        
        # KDJ指标
        stoch = df.ta.stoch(high='High', low='Low', close='Close', k=9, d=3, smooth_k=3)
        indicators['KDJ_K'] = stoch['STOCHk_9_3_3']
        indicators['KDJ_D'] = stoch['STOCHd_9_3_3']
        indicators['KDJ_J'] = 3 * stoch['STOCHk_9_3_3'] - 2 * stoch['STOCHd_9_3_3']
        
        # 布林带
        bbands = df.ta.bbands(length=20, close='Close')
        indicators['BB_upper'] = bbands['BBU_20_2.0']
        indicators['BB_middle'] = bbands['BBM_20_2.0']
        indicators['BB_lower'] = bbands['BBL_20_2.0']
        
        return indicators
        
    def generate_technical_summary(self, data, indicators):
        """生成技术分析摘要
        
        Args:
            data (pandas.DataFrame): 股票数据
            indicators (dict): 技术指标
            
        Returns:
            dict: 技术分析摘要
        """
        if not indicators:
            raise ValueError("没有可用的技术指标，请先计算指标")
            
        last_close = data['Close'].iloc[-1]
        stock_code = data.get('stock_code', '')
        stock_name = data.get('stock_name', '')
        
        # 确保使用相同的索引
        last_index = data.index[-1]
        
        # 安全地获取指标值
        sma_5 = indicators['SMA_5'].loc[last_index] if last_index in indicators['SMA_5'].index else np.nan
        sma_10 = indicators['SMA_10'].loc[last_index] if last_index in indicators['SMA_10'].index else np.nan
        sma_20 = indicators['SMA_20'].loc[last_index] if last_index in indicators['SMA_20'].index else np.nan
        sma_60 = indicators['SMA_60'].loc[last_index] if last_index in indicators['SMA_60'].index else np.nan
        rsi_6 = indicators['RSI_6'].loc[last_index] if last_index in indicators['RSI_6'].index else np.nan
        rsi_12 = indicators['RSI_12'].loc[last_index] if last_index in indicators['RSI_12'].index else np.nan
        macd = indicators['MACD'].loc[last_index] if last_index in indicators['MACD'].index else np.nan
        macd_signal = indicators['MACD_signal'].loc[last_index] if last_index in indicators['MACD_signal'].index else np.nan
        kdj_k = indicators['KDJ_K'].loc[last_index] if last_index in indicators['KDJ_K'].index else np.nan
        kdj_d = indicators['KDJ_D'].loc[last_index] if last_index in indicators['KDJ_D'].index else np.nan
        kdj_j = indicators['KDJ_J'].loc[last_index] if last_index in indicators['KDJ_J'].index else np.nan
        
        summary = {
            '股票代码': stock_code,
            '股票名称': stock_name,
            '最新收盘价': round(last_close, 2),
            '5日均线': round(sma_5, 2) if not np.isnan(sma_5) else None,
            '10日均线': round(sma_10, 2) if not np.isnan(sma_10) else None,
            '20日均线': round(sma_20, 2) if not np.isnan(sma_20) else None,
            '60日均线': round(sma_60, 2) if not np.isnan(sma_60) else None,
            '6日RSI': round(rsi_6, 2) if not np.isnan(rsi_6) else None,
            '12日RSI': round(rsi_12, 2) if not np.isnan(rsi_12) else None,
            'MACD': round(macd, 4) if not np.isnan(macd) else None,
            'MACD信号线': round(macd_signal, 4) if not np.isnan(macd_signal) else None,
            'KDJ_K': round(kdj_k, 2) if not np.isnan(kdj_k) else None,
            'KDJ_D': round(kdj_d, 2) if not np.isnan(kdj_d) else None,
            'KDJ_J': round(kdj_j, 2) if not np.isnan(kdj_j) else None,
            '短期趋势': "上涨" if last_close > sma_5 > sma_10 else "下跌",
            '中期趋势': "上涨" if sma_10 > sma_20 > sma_60 else "下跌",
            'RSI状态': "超买" if rsi_6 > 80 or rsi_12 > 70 else "超卖" if rsi_6 < 20 or rsi_12 < 30 else "中性",
            'MACD信号': "金叉" if macd > macd_signal else "死叉",
            'KDJ信号': "超买" if kdj_j > 100 else "超卖" if kdj_j < 0 else "中性"
        }
        
        return summary
    
    def generate_financial_summary(self, stock_code, stock_name, financial_reports):
        """生成财务分析摘要
        
        Args:
            stock_code (str): 股票代码
            stock_name (str): 股票名称
            financial_reports (pandas.DataFrame): 财务报表数据
            
        Returns:
            str: 财务分析摘要
        """
        if financial_reports is None or financial_reports.empty:
            return None

        print(f"成功获取到 {len(financial_reports)} 条财务指标记录。")
       
        # 数据通常按报告日期降序排列，第一行是最新的
        latest_data = financial_reports.iloc[0] # 获取最新一期的数据

        print(f"\n最新报告期: {latest_data.name}") # Series的name通常是日期索引

        # 提取关键指标
        key_metrics = {}
        possible_metrics = {
            'roe': '净资产收益率(%)', # 盈利能力
            'net_profit_margin': '销售净利率(%)', # 盈利能力
            'gross_profit_margin': '销售毛利率(%)', # 盈利能力
            'debt_to_asset_ratio': '资产负债率(%)', # 偿债能力
            'current_ratio': '流动比率', # 短期偿债能力
            'quick_ratio': '速动比率', # 短期偿债能力
            'total_asset_turnover': '总资产周转率(次)', # 营运能力
            'eps': '基本每股收益(元)', # 每股指标
            'net_profit_growth_rate': '净利润同比增长率(%)' # 成长能力
        }

        print("\n尝试提取关键指标...")
        for key, col_name in possible_metrics.items():
            if col_name in latest_data.index:
                key_metrics[key] = latest_data[col_name]
                print(f"  - 提取到 {col_name}: {key_metrics[key]}")
            else:
                key_metrics[key] = 'N/A' # 如果找不到该列，标记为 N/A
                print(f"  - 未找到指标: {col_name}")

        # 生成财务分析摘要
        print("\n--- 财务分析摘要 ---")
        summary = f"公司代码: {stock_code}\n"
        summary += f"公司名称: {stock_name}\n"
        summary += f"最新报告期: {latest_data.name}\n\n"
        summary += "**盈利能力:**\n"
        summary += f"- 净资产收益率 (ROE): {key_metrics.get('roe', 'N/A')}% (衡量股东权益回报水平)\n"
        summary += f"- 销售净利率: {key_metrics.get('net_profit_margin', 'N/A')}% (衡量销售收入的盈利能力)\n"
        summary += f"- 销售毛利率: {key_metrics.get('gross_profit_margin', 'N/A')}% (衡量主营业务的初始盈利空间)\n\n"

        summary += "**偿债能力:**\n"
        summary += f"- 资产负债率: {key_metrics.get('debt_to_asset_ratio', 'N/A')}% (衡量总资产中通过负债筹集的比例，过高可能风险较大)\n"
        summary += f"- 流动比率: {key_metrics.get('current_ratio', 'N/A')} (衡量短期偿债能力，通常认为 > 2 较好)\n"
        summary += f"- 速动比率: {key_metrics.get('quick_ratio', 'N/A')} (更严格的短期偿债能力指标，通常认为 > 1 较好)\n\n"

        summary += "**营运能力:**\n"
        summary += f"- 总资产周转率: {key_metrics.get('total_asset_turnover', 'N/A')} 次 (衡量资产运营效率，越高越好)\n\n"

        summary += "**每股指标与成长性:**\n"
        summary += f"- 基本每股收益 (EPS): {key_metrics.get('eps', 'N/A')} 元 (衡量普通股股东每股盈利)\n"
        summary += f"- 净利润同比增长率: {key_metrics.get('net_profit_growth_rate', 'N/A')}% (衡量公司盈利的增长速度)\n\n"

        summary += "**初步结论:**\n"
        # 这里加入一些基于数据的判断逻辑
        roe_value = pd.to_numeric(key_metrics.get('roe'), errors='coerce')
        debt_ratio_value = pd.to_numeric(key_metrics.get('debt_to_asset_ratio'), errors='coerce')

        if roe_value is not None and roe_value > 15:  # 假设 ROE > 15% 算优秀
            summary += "- 公司盈利能力较强 (ROE 较高)。\n"
        elif roe_value is not None:
            summary += "- 公司盈利能力一般或有待观察。\n"

        if debt_ratio_value is not None and debt_ratio_value < 50:  # 假设资产负债率 < 50% 算稳健
            summary += "- 财务结构相对稳健 (资产负债率较低)。\n"
        elif debt_ratio_value is not None:
            summary += "- 需要关注公司的债务水平 (资产负债率较高)。\n"

        if key_metrics.get('net_profit_growth_rate') != 'N/A':
            growth_rate = pd.to_numeric(key_metrics.get('net_profit_growth_rate'), errors='coerce')
            if growth_rate is not None and growth_rate > 10:  # 假设增长率 > 10% 算良好
                summary += "- 公司具有一定的成长性。\n"
            elif growth_rate is not None:
                summary += "- 公司成长性需要进一步分析。\n"

        summary += "\n*注意: 此摘要仅基于部分关键财务指标的最新数据，未进行深入的趋势分析、行业对比和定性分析，不构成任何投资建议。*"
        
        return summary


