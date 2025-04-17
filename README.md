# A股选股与分析工具

这是一个基于Python的A股选股与分析工具，可以帮助您分析股票的量价关系，进行选股和评估股票的投资价值。该工具提供命令行界面，方便在终端中使用。

## 功能特点

- **选股功能**：基于量能、技术指标等多种选股策略
- **股票分析**：支持量价关系分析、黄金分割等多种分析方法
- **数据管理**：自动获取和更新股票数据
- **图表生成**：生成专业的技术分析图表
- **命令行接口**：方便的命令行操作

## 安装说明

1. 克隆仓库：

```bash
git clone https://github.com/your-username/a_stock_project.git
cd a_stock_project
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 配置数据库（MySQL）：

确保您已安装并启动MySQL服务，然后创建数据库：

```sql
CREATE DATABASE stock_data_news CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

4. 更新配置：

将 `config/default_config.py` 文件改名为 `config/default_config.py`，
更新数据库连接配置和其他设置，更新配置 AI 模型的配置。

## 使用方法

### 查看命令
```bash
python run.py -h
python run.py update -h
```

### 初始化数据

首次使用前，请更新股票数据：

```bash
python run.py update [--basic] [--daily] [--full]
```

### 选股功能

执行量能选股：

```bash
python run.py select volume --days 120 --threshold 2 --limit 50
```

参数说明：
- `volume`：选股器类型，可选 volume(量能选股)、technical(技术选股)、combined(组合选股)
- `--days`：回溯数据天数
- `--threshold`：选股分数阈值
- `--limit`：限制结果数量
- `--output`：输出文件名

### 分析功能

执行量价分析：

```bash
python run.py analyze volprice 000001 --days 60
```

参数说明：
- `volprice`：分析器类型，可选 volprice(量价分析)、golden(黄金分割分析)、openai(AI分析)
- `000001`：股票代码
- `--days`：回溯数据天数
- `--end-date`：结束日期，格式：YYYY-MM-DD
- `--save-chart`：是否保存图表
- `--output`：输出文件名(不含扩展名)

### 更新数据

增量更新股票数据：

```bash
python run.py update
```

全量更新股票数据：

```bash
python run.py update --full
```

## 输出结果

- 选股结果保存在 `datas/results/` 目录下
- 分析报告保存在 `datas/reports/` 目录下
- 图表保存在 `datas/charts/` 目录下
- 日志保存在 `logs/` 目录下

## 项目结构

```
a_stock_project/
├── run.py                 # 主入口文件
├── requirements.txt       # 依赖包列表
├── README.md              # 项目说明文档
├── config/                # 配置文件目录
├── data/                  # 数据访问层
├── utils/                 # 工具类
├── selector/              # 选股模块
├── analyzer/              # 分析模块
├── cli/                   # 命令行接口
└── output/                # 输出数据目录
    ├── charts/            # 图表输出
    ├── reports/           # 报告输出
    └── results/           # 选股结果输出
```

## 许可证

本项目采用 MIT 许可证。详情请参阅 LICENSE 文件。 

# 股票分析工具

## 项目介绍

本项目是一个股票分析工具，提供各种分析器用于分析股票的技术指标和市场行为。项目采用模块化设计，可以轻松扩展新的分析功能。

## 分析器列表

### 基础分析器 (BaseAnalyzer)

基础分析器提供了通用的数据获取和处理功能，所有专用分析器都继承自这个基类。

功能：
- 获取股票数据（从数据库或API）
- 获取股票名称和技术指标
- 保存分析结果到数据库
- 提供标准化的分析流程

### 量价分析器 (VolPriceAnalyzer)

量价分析器用于分析股票价格和交易量之间的关系，可以检测洗盘特征。

功能：
- 检测量价关系
- 识别洗盘特征
- 分析成交量萎缩、横盘震荡等模式
- 生成量价关系图

### 黄金分割分析器 (GoldenCutAnalyzer)

黄金分割分析器用于计算斐波那契回调水平，帮助识别可能的支撑位和阻力位。

功能：
- 识别主要波段的高低点
- 计算斐波那契回调水平
- 生成带有斐波那契水平的K线图
- 分析当前价格相对于斐波那契水平的位置

### openai分析器 (DeepseekAnalyzer)

利用openai大模型进行股票的深度分析，结合技术指标和基本面数据，提供更全面的分析报告。

功能：
- 综合技术指标分析
- 获取财务数据和新闻情绪
- 生成全面分析报告
- 提供投资建议

## 使用方法

### 黄金分割分析器使用示例

```python
from analyzer.golden_cut_analyzer import GoldenCutAnalyzer

# 创建分析器实例
analyzer = GoldenCutAnalyzer(
    stock_code="600519",  # 股票代码
    stock_name="贵州茅台",  # 股票名称（可选）
    end_date="2023-04-10",  # 结束日期（可选，默认为当前日期）
    days=180  # 回溯天数（可选，默认使用配置值）
)

# 运行分析
result = analyzer.run_analysis()

# 输出分析结果
print(f"分析结果状态: {result['status']}")
print(f"分析描述: {result['description']}")

if 'chart_path' in result:
    print(f"图表已保存至: {result['chart_path']}")
```

## 特性

- **模块化设计**：每个分析器都是独立的模块，可以单独使用或组合使用
- **标准化接口**：所有分析器遵循相同的接口规范，方便扩展
- **容错机制**：分析器会处理各种异常情况，确保程序不会崩溃
- **日志记录**：详细的日志记录，方便调试和追踪问题
- **数据库支持**：可以将分析结果保存到数据库中，方便查询和共享
- **模拟数据**：支持使用模拟数据进行测试和开发

## 项目结构

```
a_stock_project/
├── analyzer/                  # 分析器模块
│   ├── base_analyzer.py       # 基础分析器
│   ├── vol_price_analyzer.py  # 量价分析器
│   ├── golden_cut_analyzer.py # 黄金分割分析器
│   └── ai_analyzer.py         # ai分析器
├── data/                      # 数据相关模块
│   └── db_manager.py          # 数据库管理器
├── utils/                     # 工具模块
│   ├── logger.py              # 日志工具
│   └── akshare_api.py         # AkShare API封装
├── config/                    # 配置模块
│   └── default_config.py      # 默认配置
├── output/                    # 输出目录
│   ├── analyzer/              # 量价分析结果
│   ├── golden_cut/            # 黄金分割分析结果
│   └── deepseek/              # DeepSeek分析结果
├── run.py                     # 黄金分割分析器测试脚本
└── README.md                  # 项目说明文档
``` 