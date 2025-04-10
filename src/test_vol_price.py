from analysis.vol_price import VolPriceAnalysis
import os

def test_single_stock(stock_code, stock_name=None, days=60):
    """测试单个股票的量价关系分析"""
    # 创建保存目录
    save_path = "./output/vol_price_analysis"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 创建分析器实例
    analyzer = VolPriceAnalysis(
        stock_code=stock_code,
        stock_name=stock_name,
        days=days,
        save_path=save_path
    )
    
    # 运行分析
    print(f"\n开始分析 {stock_name or stock_code} 的量价关系...")
    result = analyzer.run_analysis(save_chart=True)
    
    # 打印分析结果
    if result['status'] == 'success':
        print("\n===== 分析结果 =====")
        print(f"股票: {result['stock_name']}({result['stock_code']})")
        print(f"日期: {result['date']}")
        
        if result['is_washing']:
            print(f"洗盘可能性: {result['wash_confidence']}%")
        else:
            print("未检测到明显洗盘")
            
        print("\n检测到的特征:")
        for pattern in result['patterns']:
            print(f"- {pattern['description']}")
            
        print("\n分析结论:")
        print(result['description'])
        print("\n分析图表已保存至:", save_path)
    else:
        print(f"分析失败: {result.get('message', '未知错误')}")

def batch_test_stocks(stock_list):
    """批量测试多只股票"""
    for stock in stock_list:
        stock_code = stock['code']
        stock_name = stock.get('name')
        days = stock.get('days', 60)
        
        try:
            test_single_stock(stock_code, stock_name, days)
        except Exception as e:
            print(f"分析 {stock_name or stock_code} 时出错: {e}")

if __name__ == "__main__":
    # 测试单个股票
    test_single_stock("000001", "平安银行")
    
    # 批量测试多只股票
    # stocks_to_test = [
    #     {"code": "000001", "name": "平安银行"},
    #     {"code": "600519", "name": "贵州茅台"},
    #     {"code": "000858", "name": "五粮液"},
    #     {"code": "600036", "name": "招商银行"},
    #     {"code": "601318", "name": "中国平安"}
    # ]
    # 
    # batch_test_stocks(stocks_to_test) 