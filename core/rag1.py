from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import os
from pymilvus import connections, Collection
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import fitz
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

# 加载 .env 文件中定义的环境变量
_ = load_dotenv(find_dotenv(), override=True)

# 初始化 OpenAI 客户端
client = OpenAI()

# 连接到 Milvus 数据库
connections.connect("default", host="192.168.138.100", port="19530")

# 加载预训练的模型和分词器
model_path = "E:/AImodle/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# 插入数据
def insert_data(collection, text):
    vector = text_to_vector(text)
    collection.insert([{"embedding": vector, "content": text}])

def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():  # 禁用梯度计算
        outputs = model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
    return vector

class ChatDoc:

    # 将文本转换为向量
    def text_to_vector(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        vector = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
        return vector

    def convertDataToMilvus(self):
        chat = ChatDoc()
        pdf_path = "D:\\test2.pdf"
        text_content = chat.extract_text_from_pdf(pdf_path)
        print("提取的PDF文本内容：")
        print(text_content)

        # 将文本转换为向量
        inputs = tokenizer(text_content, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():  # 禁用梯度计算
            outputs = model(**inputs)

        # 获取最后一层的隐藏状态（通常用于文本嵌入）
        last_hidden_state = outputs.last_hidden_state
        # 获取 [CLS] 令牌的嵌入（通常用于句子级别的表示）
        vector_store1 = last_hidden_state[:, 0, :].numpy()


        print("文本向量：", vector_store1)
        # 连接到 Milvus
        connections.connect("default", host="192.168.138.100", port="19530")

        # 定义集合 schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),

        ]
        schema = CollectionSchema(fields, description="default collection")

        # 创建集合（如果集合不存在）
        collection_name = "default"
        collection = Collection(name=collection_name, schema=schema)

        # 确保 vector_store1 是一个浮点向量列表
        if isinstance(vector_store1, np.ndarray):
            vector_store1 = vector_store1.tolist()

        # 插入向量数据
        # 注意：Milvus 的 insert 方法需要一个列表，其中每个元素是一个浮点向量
        # data_to_insert = ["embedding":vector_store1,"content":text_content]

        # mr = collection.insert(data_to_insert)

        for text in text_content:
            insert_data(collection, text)
        # 加载集合
        # collection.load()

    def extract_text_from_pdf(self, pdf_path):
        document = fitz.open(pdf_path)
        text_content = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text_content += page.get_text()
        return text_content

    # 插入数据
    def insert_data(self,collection, text):
        vector = text_to_vector(text)
        mr= collection.insert([{"embedding": vector, "content": text}])
        print("插入结果：", mr)

    def get_text_from_docx(self):
        # 连接到 Milvus 数据库
        connections.connect("default", host="192.168.138.100", port="19530")
        # 加载预训练的模型和分词器
        model_path = "E:/AImodle/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)  # 使用 AutoModel 而不是 AutoModelForSequenceClassification

        # 将关键字转换为向量
        query_text = "我是哪里人？"
        inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():  # 禁用梯度计算
            outputs = model(**inputs)
        query_vector = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
        # 获取集合（假设集合名为 'my_collection'，向量字段名为 'embedding'）
        collection = Collection("default")
        collection.load()

        # 设置搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        # 执行相似度查询
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=5,  # 返回最相似的 5 条数据
            output_fields=["content"]  # 假设文本内容字段名为 'content'
        )

        # 输出查询结果
        for result in results[0]:
            print(f"Content: {result.entity.get('content')}, Similarity score: {result.score}")


if __name__ == "__main__":
    chat = ChatDoc()

    # 插入 Milvus
    # chat.convertDataToMilvus()
    chat.get_text_from_docx()