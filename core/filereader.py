import os

from pathlib import Path

def pdfreader(file_path):
    """读取PDF文件并返回text文本"""
    assert os.path.exists(file_path), "File not found"
    assert file_path.endswith(".pdf"), "File format not supported"

    from llama_index.readers.file import PDFReader
    doc = PDFReader().load_data(file=Path(file_path))

    # 简单的拼接起来之后返回纯文本
    text = "\n\n".join([d.get_content() for d in doc])
    return text

def plainreader(file_path):
    """读取普通文本文件并返回text文本"""
    assert os.path.exists(file_path), "File not found"

    # 尝试不同的编码方式读取数据
    encodings_to_try = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
    
    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as f:
                text = f.read()
            # 如果成功读取，返回文本内容
            return text
        except UnicodeDecodeError:
            # 如果是最后一种编码方式仍然失败
            if enc == encodings_to_try[-1]:
                # 使用二进制模式读取，然后用latin-1编码（可以读取任何字节）
                with open(file_path, 'rb') as f:
                    content = f.read()
                    return content.decode('latin-1')
            # 否则尝试下一种编码
            continue

def csvreader(file_path, **kwargs):
    """读取CSV文件并返回text文本"""
    assert os.path.exists(file_path), "File not found"
    assert file_path.endswith(".csv"), "File format not supported"
    
    import csv
    
    # 获取CSV读取参数
    delimiter = kwargs.get("delimiter", ",")
    encoding = kwargs.get("encoding", "utf-8")
    
    # 尝试不同的编码方式读取数据
    encodings_to_try = [encoding, 'utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
    rows = []
    
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=delimiter)
                # 获取标题行
                header = next(csv_reader, None)
                if header:
                    rows.append(",".join(header))
                # 读取数据行
                for row in csv_reader:
                    rows.append(",".join(row))
            # 如果成功读取，跳出循环
            break
        except UnicodeDecodeError:
            # 如果是最后一种编码方式仍然失败
            if enc == encodings_to_try[-1]:
                # 使用二进制模式读取，然后用latin-1编码（可以读取任何字节）
                with open(file_path, 'rb') as csvfile:
                    content = csvfile.read()
                    text = content.decode('latin-1')
                    lines = text.splitlines()
                    for line in lines:
                        rows.append(line)
            # 否则尝试下一种编码
            continue
    
    # 将所有行拼接成单个文本字符串
    text = "\n".join(rows)
    
    return text

def excelreader(file_path, **kwargs):
    """读取Excel文件并返回text文本"""
    assert os.path.exists(file_path), "File not found"
    assert file_path.endswith((".xlsx", ".xls")), "File format not supported"
    
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for reading Excel files. Please install it using 'pip install pandas'")
    
    # 获取Excel读取参数
    sheet_name = kwargs.get("sheet_name", 0)  # 默认读取第一个工作表
    header = kwargs.get("header", 0)  # 默认第一行为标题
    
    try:
        # 使用pandas读取Excel文件
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
        
        # 将DataFrame转换为文本
        rows = []
        
        # 添加列名作为标题行
        rows.append(",".join(df.columns.astype(str)))
        
        # 添加数据行
        for _, row in df.iterrows():
            rows.append(",".join(row.astype(str)))
        
        # 将所有行拼接成单个文本字符串
        text = "\n".join(rows)
        
        return text
    except Exception as e:
        # 出现错误时提供更详细的错误信息
        import traceback
        error_info = traceback.format_exc()
        raise Exception(f"读取Excel文件失败: {str(e)}\n{error_info}")

