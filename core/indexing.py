import os
from pathlib import Path
from llama_index.core import Document
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import FlatReader, DocxReader
from src.utils.logging_config import setup_logger

logger = setup_logger("server-chat")
from src.utils import hashstr


def chunk(content, params=None):
    """
    将内容分块
    如果传入的是字符串，则按照文本分块
    如果传入的是文件路径，则按照文件类型分块
    如果传入的是列表，则按照每个元素分块
    """
    from src.utils import setup_logger
    logger = setup_logger("Indexing")

    # 默认参数
    chunk_size = params.get("chunk_size", 8000) if params else 5000
    chunk_overlap = params.get("chunk_overlap", 300) if params else 200

    # 创建节点列表
    nodes = []

    try:
        # 处理传入的内容
        if isinstance(content, str):
            # 判断是否是文件路径，先进行编码处理以应对中文路径
            file_exists = False
            processed_path = content

            try:
                file_exists = os.path.exists(content)
                logger.info(f"检查文件路径: {content}, 存在: {file_exists}")
            except UnicodeEncodeError:
                # 处理可能的编码问题
                try:
                    # 尝试转换路径编码
                    processed_path = content.encode('utf-8').decode('latin-1')
                    file_exists = os.path.exists(processed_path)
                    logger.info(f"编码转换后路径: {processed_path}, 存在: {file_exists}")
                except Exception:
                    file_exists = False
                    logger.warning(f"编码转换失败: {content}")

            if file_exists:
                # 按文件类型处理
                logger.info(f"开始处理文件: {processed_path}")
                try:
                    if processed_path.endswith('.csv'):
                        # CSV文件处理
                        logger.info("检测到CSV文件")
                        from src.core.filereader import csvreader
                        csv_params = params.get("csv_params", {}) if params else {}
                        text = csvreader(processed_path, **csv_params)
                        logger.info(f"CSV文件读取完成，文本长度: {len(text)}")
                        # 文本分块
                        text_splitter = create_text_splitter(chunk_size, chunk_overlap)
                        chunks = text_splitter.split_text(text)
                        nodes = [Node(text=chunk) for chunk in chunks]
                        logger.info(f"CSV文件分块完成，共 {len(nodes)} 个块")
                    elif processed_path.endswith(('.xlsx', '.xls')):
                        # Excel文件处理
                        logger.info("检测到Excel文件")
                        from src.core.filereader import excelreader
                        excel_params = params.get("excel_params", {}) if params else {}
                        text = excelreader(processed_path, **excel_params)
                        logger.info(f"Excel文件读取完成，文本长度: {len(text)}")
                        # 文本分块
                        text_splitter = create_text_splitter(chunk_size, chunk_overlap)
                        chunks = text_splitter.split_text(text)
                        nodes = [Node(text=chunk) for chunk in chunks]
                        logger.info(f"Excel文件分块完成，共 {len(nodes)} 个块")
                    elif processed_path.endswith(('.docx', '.doc')):
                        logger.info("检测到Word文档")
                        try:
                            parser = SimpleFileNodeParser()
                            docs = DocxReader().load_data(Path(processed_path))
                            nodes = parser.get_nodes_from_documents(docs)
                            logger.info(f"Word文档处理完成，共 {len(nodes)} 个节点")
                        except Exception as e:
                            logger.error(f"使用DocxReader处理失败，尝试使用备用方法: {str(e)}")
                            # 如果DocxReader失败，使用plainreader作为备用
                            from src.plugins.filereader import plainreader
                            text = plainreader(processed_path)
                            logger.info(f"备用方法读取完成，文本长度: {len(text)}")
                            text_splitter = create_text_splitter(chunk_size, chunk_overlap)
                            chunks = text_splitter.split_text(text)
                            nodes = [Node(text=chunk) for chunk in chunks]
                            logger.info(f"备用方法分块完成，共 {len(nodes)} 个块")
                    else:
                        # 其他文件类型，按文本处理
                        logger.info(f"检测到其他文件类型: {processed_path}")
                        from src.core.filereader import plainreader
                        text = plainreader(processed_path)
                        logger.info(f"文件读取完成，文本长度: {len(text)}")
                        # 文本分块
                        text_splitter = create_text_splitter(chunk_size, chunk_overlap)
                        chunks = text_splitter.split_text(text)
                        nodes = [Node(text=chunk) for chunk in chunks]
                        logger.info(f"文件分块完成，共 {len(nodes)} 个块")
                except Exception as e:
                    # 处理任何可能的错误，并记录详细日志
                    import traceback
                    logger.error(f"处理文件 {processed_path} 时出错: {str(e)}")
                    logger.debug(traceback.format_exc())
                    # 作为纯文本处理
                    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
                    chunks = text_splitter.split_text(f"错误: 无法处理文件 {processed_path}，原因: {str(e)}")
                    nodes = [Node(text=chunk) for chunk in chunks]
                    logger.info(f"错误处理完成，共 {len(nodes)} 个块")
            else:
                # 普通文本字符串
                logger.info(f"处理普通文本字符串，长度: {len(content)}")
                text_splitter = create_text_splitter(chunk_size, chunk_overlap)
                chunks = text_splitter.split_text(content)
                nodes = [Node(text=chunk) for chunk in chunks]
                logger.info(f"文本分块完成，共 {len(nodes)} 个块")

        elif isinstance(content, list):
            # 处理列表内容，常用于CSV文件返回的多个文本块
            logger.info(f"处理列表内容，共 {len(content)} 个元素")
            for item in content:
                if isinstance(item, str):
                    if len(item) > chunk_size * 2:
                        # 如果单个条目太长，进行分块
                        text_splitter = create_text_splitter(chunk_size, chunk_overlap)
                        sub_chunks = text_splitter.split_text(item)
                        nodes.extend([Node(text=chunk) for chunk in sub_chunks])
                    else:
                        # 否则每个条目作为一个块
                        nodes.append(Node(text=item))
            logger.info(f"列表处理完成，共 {len(nodes)} 个块")
        else:
            logger.warning(f"不支持的内容类型: {type(content)}")
            nodes = []

    except Exception as e:
        import traceback
        logger.error(f"chunk函数执行失败: {str(e)}")
        logger.debug(traceback.format_exc())
        # 返回一个错误节点
        nodes = [Node(text=f"处理失败: {str(e)}")]

    logger.info(f"chunk函数执行完成，最终返回 {len(nodes)} 个节点")
    return nodes


def create_text_splitter(chunk_size, chunk_overlap):
    """创建文本分割器"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )


class Node:
    """简单的节点类，用于保存文本和元数据"""

    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata or {}
