# 多级别缠论分析与交易策略系统

基于AKShare数据的多级别缠论分析框架，实现了缠论分型、笔、线段、中枢的识别，以及多级别联立分析和交易信号生成。

## 功能特点

- **多周期数据获取**：支持日线、30分钟、5分钟、1分钟等多个级别的数据
- **中枢自动识别**：实现了基于缠论标准的中枢识别逻辑
- **多级别联立分析**：高级别趋势与低级别买卖点联动验证
- **趋势强度矩阵**：构建4x4趋势状态矩阵，整合多级别分析结果
- **自动交易信号**：根据多级别联立状态生成买卖建议
- **图表可视化**：绘制带有分型、笔、线段、中枢标记的K线图
- **风控位计算**：智能设置止损位置，基于中枢边界

## 安装依赖

```bash
pip install akshare pandas numpy matplotlib mplfinance
```

## 快速开始

### 基本用法

```python
from analyzer.chan_analyzer import ChanAnalyzer

# 创建分析器实例
analyzer = ChanAnalyzer(
    symbol="sh000300",  # 沪深300指数
    periods=['daily', '30min', '5min'],  # 分析周期
    end_date=None,  # 默认最新日期
    data_len_min=1000  # 数据长度
)

# 运行完整分析
analyzer.run_full_analysis()

# 获取多级别趋势矩阵
trend_matrix = analyzer.build_trend_matrix()
print(trend_matrix)

# 生成交易信号
signal = analyzer.generate_trading_signal()
print(signal)

# 绘制分析图表
analyzer.plot_all_requested_levels()
```

### 使用多级别交易系统

```bash
python multi_level_trader.py --symbol sh000300 --periods daily,30min,5min --simulate
```

参数说明：
- `--symbol`：股票/指数代码，必须包含市场前缀，如`sh000300`、`sz000001`
- `--periods`：要分析的周期，用逗号分隔，默认`daily,30min,5min`
- `--end-date`：分析截止日期，格式YYYYMMDD，默认最新
- `--simulate`：是否进行模拟交易
- `--initial-capital`：模拟交易初始资金，默认100000
- `--use-llm`：是否使用LLM增强分析（需自行实现接口）

## 多级别联立分析框架

系统基于以下缠论原理构建：

1. **走势分解原理**：任何走势都可分解为a1A1+a5A5+a30A30形式
2. **中枢定义**：连续三笔有重叠区域
3. **趋势定义**：中枢不重叠构成趋势，中枢重叠则升级
4. **买卖点类型**：
   - 一买：中枢下方第一个底分型确认
   - 二买：跌破中枢下轨后回拉确认突破
   - 三买：中枢上方突破回调不破上轨
   - 一卖/二卖/三卖：与买点逻辑相反

## 示例决策逻辑

| 场景                | 信号组合                          | 操作建议                  | 风险控制               |
|---------------------|-----------------------------------|---------------------------|------------------------|
| 日线涨+30分盘整     | 5分三买+1分放量突破               | 半仓介入                  | 跌破30分中枢下沿止损    |
| 日线跌+30分背驰     | 5分二买+1分底背驰                 | 轻仓抄底                  | 新低止损               |
| 多级别共振下跌      | 日线三卖+30分三卖+5分弱势反弹     | 持币观望                  | 等待日线底分型         |

## 项目结构

```
├── analyzer/
│   └── chan_analyzer.py    # 缠论核心分析类
├── multi_level_trader.py   # 多级别交易系统
└── README.md               # 项目说明
```

## 注意事项

- 本系统使用AKShare获取数据，可能需要网络代理
- 缠论分析为简化实现，仅用于教学和演示
- 实盘交易前请充分测试并结合其他分析方法
- 风险自负，投资有风险

## 许可证

MIT 