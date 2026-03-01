from typing import List, Optional, Dict, Any
import os
import json
from pymilvus import MilvusClient, MilvusException
from datetime import datetime
import logging
import time
from src.models.embedding import EmbeddingModel
from src.utils import setup_logger, hashstr

# 设置日志记录
logger = logging.getLogger(__name__)

def chunk_text(text: str, max_length: int = 30000) -> List[str]:
    """将文本分块，确保每块不超过最大长度限制
    
    Args:
        text: 要分块的文本
        max_length: 每块的最大长度（默认30000，远小于Milvus的65536限制）
    
    Returns:
        List[str]: 分块后的文本列表
    """
    if len(text) <= max_length:
        return [text]
        
    chunks = []
    current_chunk = ""
    
    # 按句子分割文本
    sentences = text.split('。')
    
    for sentence in sentences:
        # 如果单个句子就超过了限制，需要强制切分
        if len(sentence) > max_length:
            for i in range(0, len(sentence), max_length):
                chunks.append(sentence[i:i + max_length])
            continue
            
        # 如果当前块加上新句子会超出限制，保存当前块并开始新块
        if len(current_chunk) + len(sentence) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence + "。"
        else:
            current_chunk += sentence + "。"
            
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

class KnowledgeBase:

    def __init__(self, config=None, embed_model=None) -> None:
        self.config = config or {}
        self.embed_model = embed_model

        self.client = None
        if not self.connect_to_milvus():
            raise ConnectionError("Failed to connect to Milvus")

    def process_file(self, file_path: str, kb_name: str, file_id: str) -> bool:
        """处理单个文件并添加到知识库
        
        Args:
            file_path: 文件路径
            kb_name: 知识库名称
            file_id: 文件ID
        
        Returns:
            bool: 是否处理成功
        """
        try:
            # 读取文件内容
            from src.core.indexing import chunk
            if file_path.endswith('.pdf'):
                from src.core.database import DataBaseManager
                dbm = DataBaseManager(self.config)
                texts = dbm.read_text(file_path)
                nodes = chunk(texts)
            else:
                nodes = chunk(file_path)

            if not nodes:
                logger.warning(f"文件内容为空: {file_path}")
                return False

            # 添加文件信息
            file_info = {
                "file_id": file_id,
                "filename": os.path.basename(file_path),
                "path": file_path,
                "type": file_path.split(".")[-1].lower(),
                "created_at": time.time()
            }
            
            # 将文件信息分块（如果需要）
            info_text = str(file_info)
            info_chunks = chunk_text(info_text)
            
            # 分批添加文件信息
            for chunk_idx, info_chunk in enumerate(info_chunks):
                chunk_id = f"{file_id}_info_{chunk_idx}"
                try:
                    self.add_documents(
                        docs=[info_chunk],
                        collection_name=kb_name,
                        file_id=chunk_id
                    )
                except Exception as e:
                    logger.error(f"添加文件信息块失败: {str(e)}")
                    continue

            # 分批处理文本内容
            batch_size = 50  # 每批处理的节点数
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                # 处理每个文本块
                for node_idx, node in enumerate(batch):
                    # 将大文本分块
                    text_chunks = chunk_text(node.text)
                    
                    # 添加每个分块
                    for chunk_idx, text_chunk in enumerate(text_chunks):
                        chunk_id = f"{file_id}_{i}_{node_idx}_{chunk_idx}"
                        try:
                            self.add_documents(
                                docs=[text_chunk],
                                collection_name=kb_name,
                                file_id=chunk_id
                            )
                        except Exception as e:
                            logger.error(f"添加文本块失败: {str(e)}")
                            continue
                
                # 清理内存
                del batch
                import gc
                gc.collect()

            logger.info(f"文件处理完成: {file_path}")
            return True

        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {str(e)}")
            return False

    def connect_to_milvus(self):
        """
        连接到 Milvus 服务。
        使用配置中的 URI，如果没有配置，则使用默认值。
        """
        try:
            #uri = os.getenv('MILVUS_URI', self.config.get('milvus_uri', "http://milvus:19530"))
            #uri = os.getenv('MILVUS_URI', self.config.get('milvus_uri', "http://192.168.20.50:19530"))
            uri = os.getenv('MILVUS_URI')
            
            # 添加连接重试机制
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"尝试连接Milvus (第{attempt + 1}次): {uri}")
                    self.client = MilvusClient(uri=uri)
                    
                    # 测试连接
                    self.client.list_collections()
                    logger.info(f"成功连接到Milvus: {uri}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"连接Milvus失败 (第{attempt + 1}次): {str(e)}")
                    if attempt < max_retries - 1:
                        logger.info(f"等待{retry_delay}秒后重试...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        logger.error(f"连接Milvus最终失败: {str(e)}")
                        return False
                        
        except Exception as e:
            logger.error(f"连接Milvus失败: {str(e)}")
            return False

    def get_collection_names(self):
        return self.client.list_collections()

    def get_collections(self):
        if not self.client:
            if not self.connect_to_milvus():
                return []
        collections_name = self.client.list_collections()
        collections = []
        for collection_name in collections_name:
            collection = self.get_collection_info(collection_name)
            collections.append(collection)

        return collections

    def get_collection_info(self, collection_name):
        if not self.client:
            if not self.connect_to_milvus():
                return {}
        collection = self.client.describe_collection(collection_name)
        collection.update(self.client.get_collection_stats(collection_name))
        # collection["id"] = hashstr(collection_name)
        return collection

    def add_collection(self, collection_name, dimension=None):
        if not self.client:
            if not self.connect_to_milvus():
                return False
        if self.client.has_collection(collection_name=collection_name):
            logger.warning(f"Collection {collection_name} not found, create it")
            self.client.drop_collection(collection_name=collection_name)
        logger.info(f"collection_name  Milvus at {collection_name}")
        self.client.create_collection(
            collection_name=collection_name,
            dimension= dimension,  # The vectors we will use in this demo has 768 dimensions
        )

    def add_documents(self, docs, collection_name, **kwargs):
        """添加已经分块之后的文本"""
        # 检查连接状态
        if not self.client:
            logger.info("Milvus客户端未连接，尝试连接...")
            if not self.connect_to_milvus():
                logger.error("无法连接到Milvus，跳过文档添加")
                return None
        
        # 检查 collection 是否存在
        import random
        if not self.client.has_collection(collection_name=collection_name):
            logger.error(f"Collection {collection_name} not found, create it")
            self.add_collection(collection_name,1024)

        vectors = self.embed_model.encode(docs)

        data = [{
            "id": int(random.random() * 1e12),
            "vector": vectors[i],
            "text": docs[i],
            "hash": hashstr(docs[i], with_salt=True),
            **kwargs} for i in range(len(vectors))]

        res = self.client.insert(collection_name=collection_name, data=data)
        return res

    def search(self, query, collection_name, limit=3):
        if not self.client:
            if not self.connect_to_milvus():
                return []
        query_vectors = self.embed_model.encode_queries([query])
        return self.search_by_vector(query_vectors[0], collection_name, limit)

    def search_by_vector(self, vector, collection_name, limit=3):
        if not self.client:
            if not self.connect_to_milvus():
                return []
        res = self.client.search(
            collection_name=collection_name,  # target collection
            data=[vector],  # query vectors
            limit=limit,  # number of returned entities
            output_fields=["text", "file_id"],  # specifies fields to be returned
        )

        # 转换搜索结果为纯字典格式，避免 pymilvus 搜索结果对象与 pydantic 的兼容问题
        processed_results = []
        for item in res[0]:
            processed_item = {
                "id": item.get("id"),
                "distance": item.get("distance"),
                "entity": {
                    "text": item.get("entity", {}).get("text", ""),
                    "file_id": item.get("entity", {}).get("file_id", "")
                }
            }
            processed_results.append(processed_item)
        
        return processed_results

    def examples(self, collection_name, limit=20):
        if not self.client:
            if not self.connect_to_milvus():
                return []
        res = self.client.query(
            collection_name=collection_name,
            limit=10,
            output_fields=["id", "text"],
        )
        return res

    def search_by_id(self, collection_name, id, output_fields=["id", "text"]):
        if not self.client:
            if not self.connect_to_milvus():
                return []
        res = self.client.get(collection_name, id, output_fields=output_fields)
        return res
