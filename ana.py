from src.analysis.deepseek_stock import AStockAnalyzer
if __name__ == "__main__":
    analyzer = AStockAnalyzer("000001", "1y")
    # 获取数据
    if analyzer.fetch_stock_data():
        # 获取财务报表
        analyzer.fetch_financial_reports()
        
        # 获取新闻舆情 (模拟数据)
        analyzer.fetch_news_sentiment(days=60)
        
        # 计算技术指标
        analyzer.calculate_technical_indicators()
        
        # 绘制分析图表
        analyzer.plot_analysis_charts(save_path="technical_analysis.png")
        
        # 生成新闻词云
        analyzer.generate_word_cloud(save_path="news_wordcloud.png")
        
        # 使用DeepSeek进行分析
        analysis = analyzer.analyze_with_deepseek(
            additional_context="白酒行业龙头，具有较强品牌溢价能力"
        )
        
        if analysis:
            print("\nDeepSeek 分析报告:")
            print(analysis)
            
            # 保存完整报告
            analyzer.save_full_report("astock_analysis_report.txt")
        else:
            print("生成分析报告失败")
    else:
        print("获取股票数据失败，请检查股票代码后重试")
