# Google搜索API设置指南

本文档介绍如何设置Google自定义搜索API（Google Custom Search JSON API）用于股票信息搜索功能。

## 先决条件

在开始之前，您需要：

1. 一个Google账号
2. 访问Google Cloud Console的权限
3. 创建自定义搜索引擎的权限

## 设置步骤

### 1. 创建Google Cloud项目

1. 访问 [Google Cloud Console](https://console.cloud.google.com/)
2. 点击页面顶部的项目下拉菜单，然后点击"新建项目"
3. 输入项目名称（例如："Stock-Search-API"）
4. 点击"创建"按钮

### 2. 启用Custom Search API

1. 在Cloud Console的导航菜单中，选择"API和服务" > "库"
2. 在搜索框中输入"Custom Search"
3. 点击搜索结果中的"Custom Search API"
4. 点击"启用"按钮

### 3. 创建API密钥

1. 在导航菜单中，选择"API和服务" > "凭据"
2. 点击"创建凭据"下拉菜单，然后选择"API密钥"
3. 系统会生成一个新的API密钥，请将其复制并保存在安全的地方
4. （可选）点击"限制密钥"来设置API密钥的限制，以提高安全性

### 4. 创建自定义搜索引擎

1. 访问 [Google Programmable Search Engine](https://programmablesearchengine.google.com/cse/all)
2. 点击"添加"按钮创建一个新的搜索引擎
3. 在"需要搜索的站点"字段中，可以添加特定的财经网站，如：
   - `finance.sina.com.cn/*`
   - `finance.eastmoney.com/*`
   - `finance.qq.com/*`
   - `business.sohu.com/*`
   
   也可以选择"搜索整个网络"
4. 输入搜索引擎名称（例如："Stock Info Search"）
5. 点击"创建"按钮
6. 在创建后的管理页面，找到"搜索引擎ID"（形如：`012345678901234567890:abcdefghijk`），复制并保存

### 5. 配置项目环境变量

将API密钥和搜索引擎ID设置为环境变量：

Windows命令行：
```cmd
set GOOGLE_API_KEY=your_api_key_here
set GOOGLE_SEARCH_CX=your_search_engine_id_here
```

Linux/Mac终端：
```bash
export GOOGLE_API_KEY=your_api_key_here
export GOOGLE_SEARCH_CX=your_search_engine_id_here
```

要永久设置环境变量，请根据操作系统设置相应的配置文件。

## 使用方法

完成设置后，您可以使用`GoogleSearchAPI`类进行搜索：

```python
from utils.google_api import GoogleSearchAPI

# 创建API实例
api = GoogleSearchAPI()

# 搜索特定股票信息
results = api.search_stock_info(
    stock_code="600519", 
    stock_name="贵州茅台",
    max_results=5,
    days=7,  # 最近7天的内容
    site_list=["finance.sina.com.cn", "finance.eastmoney.com"]
)

# 打印结果
for result in results:
    print(f"标题: {result['title']}")
    print(f"链接: {result['link']}")
    print(f"摘要: {result['snippet']}")
    print("-" * 50)
```

## 配额和限制

Google Custom Search JSON API有以下限制：

- 免费版每天可以执行100次查询
- 超过免费配额后，每1000次查询收费$5
- 每次请求最多返回10个结果
- 请求频率限制为每用户每秒10次查询

请注意管理您的API使用情况，避免产生意外费用。

## 常见问题排查

1. **API密钥错误**：确保环境变量`GOOGLE_API_KEY`正确设置
2. **搜索引擎ID错误**：确保环境变量`GOOGLE_SEARCH_CX`正确设置
3. **超出配额**：检查Google Cloud Console中的配额使用情况
4. **返回空结果**：尝试修改搜索查询或检查自定义搜索引擎设置

## 参考资料

- [Custom Search JSON API 文档](https://developers.google.com/custom-search/v1/overview)
- [Google Cloud 控制台](https://console.cloud.google.com/)
- [Programmable Search Engine 控制台](https://programmablesearchengine.google.com/) 