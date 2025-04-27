"""
修复pandas_ta中numpy.NaN导入问题的脚本
"""
import os
import sys

# 文件路径
file_path = os.path.join('venv', 'Lib', 'site-packages', 'pandas_ta', 'momentum', 'squeeze_pro.py')

try:
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        sys.exit(1)
        
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # 输出当前引用情况
    numpy_import_line = None
    for line in content.split('\n'):
        if 'numpy import' in line and 'npNaN' in line:
            numpy_import_line = line
            print(f"找到导入语句: {line}")
            break
    
    if not numpy_import_line:
        print("未找到需要修改的导入语句")
        sys.exit(1)
    
    # 替换NaN为nan
    if 'from numpy import NaN as npNaN' in content:
        updated_content = content.replace('from numpy import NaN as npNaN', 'from numpy import nan as npNaN')
        print("已替换 'from numpy import NaN as npNaN' 为 'from numpy import nan as npNaN'")
    else:
        print("文件中不包含 'from numpy import NaN as npNaN'")
        sys.exit(1)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)
    
    print("文件已成功修复！")
    
except Exception as e:
    print(f"发生错误: {e}") 