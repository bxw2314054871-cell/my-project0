import os
import json
import time
import traceback
from pathlib import Path
import tempfile
from src.plugins import pdf2txt
from src.utils import hashstr, setup_logger, is_text_pdf
from src.models.embedding import get_embedding_model
import numpy as np
from typing import List
import cv2

logger = setup_logger("DataBaseManager")


class DataBaseManager:

    def __init__(self, config=None) -> None:
        self.config = config
        
        # 尝试不同的数据库路径
        possible_paths = [
            os.path.join(config.save_dir, "data", "database.json"),
            os.path.join("/app/saves/data", "database.json"),
            os.path.join(tempfile.gettempdir(), "app_data", "database.json")
        ]
        
        self.database_path = None
        for path in possible_paths:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                # 测试文件可写性
                with open(path, "a") as f:
                    pass
                    
                self.database_path = path
                logger.info(f"使用数据库路径: {path}")
                break
            except Exception as e:
                logger.warning(f"无法使用数据库路径 {path}: {e}")
                continue
                
        if not self.database_path:
            error_msg = "无法找到可写的数据库路径"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.embed_model = get_embedding_model(config)

        # 总是初始化知识库，不再检查enable_knowledge_base
        from src.core.knowledgebase import KnowledgeBase
        self.knowledge_base = KnowledgeBase(config, self.embed_model)
        
        # 图数据库仍然保持原有的条件判断
        if hasattr(self.config, 'enable_knowledge_graph') and self.config.enable_knowledge_graph:
            from src.core.graphbase import GraphDatabase
            self.graph_base = GraphDatabase(self.config, self.embed_model)
            self.graph_base.start()
        else:
            self.graph_base = None

        self.data = {"databases": [], "graph": {}}

        try:
            self._load_databases()
            self._update_database()
        except Exception as e:
            logger.error(f"初始化数据库失败: {e}")
            # 创建新的空数据库文件
            self._save_databases()

    def _load_databases(self):
        """将数据库的信息保存到本地的文件里面"""
        if not os.path.exists(self.database_path):
            return

        try:
            with open(self.database_path, "r") as f:
                data = json.load(f)
                self.data = {
                    "databases": [DataBaseLite(**db) for db in data["databases"]],
                    "graph": data["graph"]
                }

            # 检查所有文件，如果出现状态是 processing 的，那么设置为 failed
            for db in self.data["databases"]:
                for file in db.files:
                    if file["status"] == "processing" or file["status"] == "waiting":
                        file["status"] = "failed"
                
                # 确保每个数据库都有 embed_model
                if not hasattr(db, 'embed_model') or not db.embed_model:
                    logger.warning(f"Database {db.db_id} has no embed_model, setting to {self.config.embed_model}")
                    db.embed_model = self.config.embed_model
            
            self._save_databases()
        except Exception as e:
            logger.error(f"加载数据库失败: {e}")
            raise

    def _save_databases(self):
        """将数据库的信息保存到本地的文件里面"""
        import time
        import random
        from filelock import FileLock
        
        lock_path = f"{self.database_path}.lock"
        temp_path = f"{self.database_path}.tmp"
        
        # 创建文件锁
        lock = FileLock(lock_path, timeout=10)  # 10秒超时
        
        max_retries = 3
        retry_delay = 1  # 初始延迟1秒
        
        for attempt in range(max_retries):
            try:
                with lock:
                    self._update_database()
                    os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
                    
                    # 写入临时文件
                    with open(temp_path, "w", encoding='utf-8') as f:
                        json.dump({
                            "databases": [db.to_dict() for db in self.data["databases"]],
                            "graph": self.data["graph"]
                        }, f, ensure_ascii=False, indent=4)
                    
                    # 原子性地替换文件
                    os.replace(temp_path, self.database_path)
                    return  # 成功保存,退出函数
                    
            except Exception as e:
                logger.error(f"保存数据库失败(尝试 {attempt + 1}/{max_retries}): {e}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # 如果不是最后一次尝试,则等待后重试
                if attempt < max_retries - 1:
                    # 使用指数退避策略
                    sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(sleep_time)
                else:
                    raise  # 最后一次尝试也失败,抛出异常

    def _update_database(self):
        self.id2db = {db.db_id: db for db in self.data["databases"]}
        self.name2db = {db.name: db for db in self.data["databases"]}
        self.metaname2db = {db.metaname: db for db in self.data["databases"]}

    def get_databases(self):
        self._update_database()
        knowledge_base_collections = self.knowledge_base.get_collection_names()
        if len(self.data["databases"]) != len(knowledge_base_collections):
            logger.warning(f"Database number not match, {knowledge_base_collections}")

        for db in self.data["databases"]:
            try:
                # 检查集合是否存在，如果不存在则创建
                if db.metaname not in knowledge_base_collections:
                    logger.info(f"Collection {db.metaname} not found, creating...")
                    self.knowledge_base.add_collection(db.metaname, db.dimension or 1024)
                    
                # 获取集合信息
                collection_info = self.knowledge_base.get_collection_info(db.metaname)
                db.update(collection_info)
            except Exception as e:
                logger.error(f"Error processing database {db.name}: {str(e)}")
                # 设置一个默认的状态
                db.update({"status": "error", "error_message": str(e)})

        return {"databases": [db.to_dict() for db in self.data["databases"]]}

    def get_graph(self):
        if self.config.enable_knowledge_graph:
            self.data["graph"].update(self.graph_base.get_database_info("neo4j"))
            return {"graph": self.data["graph"]}
        else:
            return {"message": "Graph base not enabled", "graph": {}}

    def create_database(self, database_name, description, db_type, dimension=None):
        """创建新的数据库"""
        logger.info(f"Creating database: {database_name} with dimension: {dimension}")

        # 创建数据库元数据
        metaname = "t" + hashstr(database_name + str(time.time()))
        logger.info(f"Creating collection with metaname: {metaname}")

        # 创建向量数据库集合
        self.knowledge_base.add_collection(
            collection_name=metaname,
            dimension=dimension or 1024
        )

        # 创建数据库对象
        database = DataBaseLite(
            name=database_name,
            description=description,
            db_type=db_type,
            dimension=dimension,
            metaname=metaname,
            embed_model=self.config.embed_model  # 设置embed_model
        )

        # 保存到数据库列表
        self.data["databases"].append(database)
        self._save_databases()

        logger.info(f"Successfully created database: {database_name}")

        # 检查数据库数量是否匹配
        collection_names = self.knowledge_base.get_collection_names()
        if len(collection_names) != len(self.data["databases"]):
            logger.warning(f"Database number not match, {collection_names}")

        return database.to_dict()

    def add_filesAuto(self, db_id, files, params=None):
        """添加文件到知识库
        
        Args:
            db_id: 数据库ID
            files: 文件路径列表
            params: 额外参数
        """
        logger.info(f"开始添加文件到数据库 {db_id}")
        logger.info(f"文件列表: {files}")
        logger.info(f"参数: {params}")

        # 预处理文件
        new_files = []
        for file in files:
            logger.info(f"开始处理文件: {file}")
            if not os.path.exists(file):
                error_msg = f"文件不存在: {file}"
                logger.error(error_msg)
                return {"message": error_msg, "status": "failed"}
            logger.info(f"文件存在性检查通过: {file}")
            
            # 从数据库获取文件信息
            try:
                from src.models.file import File
                from src.config.database import SessionLocal
                logger.info(f"开始从数据库获取文件信息: {file}")
                db_session = SessionLocal()
                file_record = db_session.query(File).filter(File.file_path == file).first()
                
                if not file_record:
                    error_msg = f"数据库中没有找到文件记录: {file}"
                    logger.error(error_msg)
                    return {"message": error_msg, "status": "failed"}
                logger.info(f"成功获取文件记录: {file_record.filename}")
                filepath="http://192.168.1.16:5173/"+file_record.file_path
                # 构建文件信息文本
                file_info = f"""
文件名: {file_record.filename}
文件描述: {file_record.description or '无'}
功能说明: {file_record.function_desc or '无'}
所属领域: {file_record.domain or '无'}
创建时间: {file_record.created_at}
文件路径: {filepath}


                """.strip()
                logger.info(f"已构建文件信息文本: {file_record.filename}")
                
                new_file = {
                    "file_id": "file_" + hashstr(file + str(time.time())),
                    "filename": file_record.filename,
                    "path": file,
                    "type": file.split(".")[-1].lower(),
                    "status": "waiting",
                    "created_at": time.time(),
                    "file_info": file_info
                }
                #db.files.append(new_file)
                new_files.append(new_file)
                logger.info(f"已添加文件到处理队列: {file_record.filename}")
                
            except Exception as e:
                error_msg = f"从数据库获取文件信息失败: {str(e)}"
                logger.error(error_msg)
                logger.error(f"错误详情: {traceback.format_exc()}")
                return {"message": error_msg, "status": "failed"}
            finally:
                db_session.close()
                logger.info("数据库会话已关闭")

        # 处理文件
        from src.core.indexing import chunk
        success_count = 0
        failed_count = 0

        collection_names=self.knowledge_base.get_collection_names()
        autokb = 'autokb'
        is_in_list = autokb in collection_names
        if is_in_list:
            logger.info(f"autokb is in the list")
        else:
            self.knowledge_base.add_collection("autokb", dimension=1024)

        for new_file in new_files:
            file_id = new_file["file_id"]
            # idx = self.get_idx_by_fileid(db, file_id)
            # db.files[idx]["status"] = "processing"
            logger.info(f"开始处理文件: {new_file['filename']}")

            try:
                # 首先添加文件信息
                logger.info(f"正在添加文件信息到向量数据库: {new_file['filename']}")

                self.knowledge_base.add_documents(
                    docs=[new_file["file_info"]],
                    collection_name="autokb",#db.metaname,
                    file_id=file_id
                )
                logger.info(f"文件信息添加成功: {new_file['filename']}")
                
                # 然后处理文件内容
                logger.info(f"开始处理文件内容: {new_file['filename']}")
                if new_file["type"] == "pdf":
                    logger.info(f"检测到PDF文件，开始读取文本: {new_file['filename']}")
                    texts = self.read_text(new_file["path"])
                    logger.info(f"PDF文本读取完成，开始分块: {new_file['filename']}")
                    nodes = chunk(texts, params=params)
                else:
                    logger.info(f"开始处理普通文件: {new_file['filename']}")
                    nodes = chunk(new_file["path"], params=params)

                if not nodes:
                    error_msg = f"无法从文件中提取内容: {new_file['filename']}"
                    logger.error(error_msg)
                    # db.files[idx]["status"] = "failed"
                    failed_count += 1
                    continue
                logger.info(f"文件内容处理完成，共 {len(nodes)} 个文本块: {new_file['filename']}")

                # 添加文件内容
                logger.info(f"开始添加文件内容到向量数据库: {new_file['filename']}")
                self.knowledge_base.add_documents(
                    docs=[node.text for node in nodes],
                    collection_name="autokb",#db.metaname,
                    file_id=file_id
                )
                logger.info(f"文件内容添加成功: {new_file['filename']}")

                # idx = self.get_idx_by_fileid(db, file_id)
                # db.files[idx]["status"] = "done"
                success_count += 1
                logger.info(f"文件处理完成: {new_file['filename']}")

            except Exception as e:
                error_msg = f"处理文件失败 {new_file['filename']}: {str(e)}"
                logger.error(error_msg)
                logger.error(f"错误详情: {traceback.format_exc()}")
                # idx = self.get_idx_by_fileid(db, file_id)
                # db.files[idx]["status"] = "failed"
                # db.files[idx]["error"] = str(e)
                failed_count += 1

            self._save_databases()
            logger.info(f"数据库状态已保存: {new_file['filename']}")

        # 检查处理结果
        logger.info(f"所有文件处理完成，成功: {success_count}，失败: {failed_count}")
        if failed_count > 0:
            return {
                "message": f"处理完成，成功: {success_count}，失败: {failed_count}",
                "status": "partial_success",
                "success_count": success_count,
                "failed_count": failed_count
            }
        else:
            return {
                "message": "全部处理完成",
                "status": "success",
                "success_count": success_count
            }

    def process_table_data(self, file_path, file_type, params=None):
        """处理表格数据，返回结构化文本列表"""
        if file_type == "csv":
            from src.core.filereader import csvreader
            text = csvreader(file_path, **(params.get("csv_params", {}) if params else {}))
        elif file_type in ["xlsx", "xls"]:
            from src.core.filereader import excelreader
            text = excelreader(file_path, **(params.get("excel_params", {}) if params else {}))
        else:
            raise ValueError(f"Unsupported table file type: {file_type}")
        
        # 将文本转换为DataFrame
        import pandas as pd
        import io
        df = pd.read_csv(io.StringIO(text)) if file_type == "csv" else \
             pd.read_excel(file_path, **params.get("excel_params", {}))
        
        structured_texts = []
        
        # 1. 添加表格概述
        table_summary = f"表格概述:\n"
        table_summary += f"总行数: {len(df)}\n"
        table_summary += f"总列数: {len(df.columns)}\n"
        table_summary += f"列名: {', '.join(df.columns)}\n"
        structured_texts.append(table_summary)
        
        # 2. 处理每列的统计信息
        for col in df.columns:
            col_summary = f"列名: {col}\n"
            if pd.api.types.is_numeric_dtype(df[col]):
                col_summary += f"数据类型: 数值\n"
                col_summary += f"最小值: {df[col].min()}\n"
                col_summary += f"最大值: {df[col].max()}\n"
                col_summary += f"平均值: {df[col].mean():.2f}\n"
            else:
                col_summary += f"数据类型: 文本\n"
                col_summary += f"唯一值数量: {df[col].nunique()}\n"
            structured_texts.append(col_summary)
        
        # 3. 处理行数据
        batch_size = 10  # 每批处理的行数
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_text = f"行 {i+1} 到 {min(i+batch_size, len(df))} 的数据:\n"
            for _, row in batch_df.iterrows():
                row_text = "行数据: "
                for col in df.columns:
                    row_text += f"{col}: {row[col]}, "
                structured_texts.append(row_text.rstrip(", "))
            
        return structured_texts

    def process_document_with_tables(self, file_path: str, file_type: str) -> List[str]:
        """处理文档，包括表格识别"""
        try:
            from paddleocr import PPStructure
            import cv2
            import numpy as np
            import fitz
            import os
            
            logger.info(f"开始处理PDF文件: {file_path}, 文件大小: {os.path.getsize(file_path)} 字节")
            
            # 初始化PPStructure，关闭文字方向检测
            table_engine = PPStructure(
                show_log=True,
                layout=True,
                table=True,
                ocr=True,
                recovery=True,
                use_angle_cls=False,  # 关闭文字方向检测
                lang="ch",
                det_limit_side_len=1500,  # 降低检测图像的最大边长
                det_db_thresh=0.3,  # 降低文本检测阈值
                det_db_box_thresh=0.5,  # 降低文本框检测阈值
                rec_batch_num=6,  # 减小批处理大小
                table_max_len=500,  # 降低表格最大长度
                drop_score=0.3  # 降低文本识别置信度阈值
            )
            logger.info("PPStructure初始化成功")
            
            # 使用PyMuPDF读取PDF
            doc = fitz.open(file_path)
            logger.info(f"成功打开PDF文件，共 {len(doc)} 页")
            
            all_texts = []
            
            for page_num in range(len(doc)):
                logger.info(f"处理第 {page_num + 1} 页")
                
                # 获取页面
                page = doc[page_num]
                
                # 将PDF页面转换为图像，增加分辨率
                zoom = 2.5  # 提高分辨率
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # 转换为numpy数组
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                
                if pix.n == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                # 图像预处理
                img_array = self.preprocess_image(img_array)
                
                try:
                    # 使用PPStructure处理图像
                    result = table_engine(img_array)
                    
                    # 处理识别结果
                    for region in result:
                        if not isinstance(region, dict):
                            continue
                            
                        region_type = region.get('type', '')
                        
                        if region_type == 'table':
                            # 处理表格
                            if 'res' in region and isinstance(region['res'], dict):
                                table_html = region['res'].get('html', '')
                                if table_html:
                                    all_texts.append(f"<table>{table_html}</table>")
                        
                        elif region_type == 'text':
                            # 处理文本
                            if 'res' in region:
                                text_list = []
                                if isinstance(region['res'], list):
                                    for line in region['res']:
                                        if isinstance(line, dict) and 'text' in line:
                                            text = line['text'].strip()
                                            if text:
                                                text_list.append(text)
                                elif isinstance(region['res'], str):
                                    text = region['res'].strip()
                                    if text:
                                        text_list.append(text)
                                
                                if text_list:
                                    all_texts.append('\n'.join(text_list))
                
                except Exception as e:
                    logger.warning(f"处理页面 {page_num + 1} 时出错: {str(e)}")
                    continue
                
            doc.close()
            
            # 合并相邻的文本块
            merged_texts = []
            current_text = ""
            
            for text in all_texts:
                if text.startswith("<table>"):
                    if current_text:
                        merged_texts.append(current_text.strip())
                        current_text = ""
                    merged_texts.append(text)
                else:
                    if current_text:
                        current_text += "\n"
                    current_text += text
            
            if current_text:
                merged_texts.append(current_text.strip())
            
            return merged_texts
            
        except Exception as e:
            logger.error(f"处理文档时出错: {str(e)}")
            return []

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理以提高OCR质量"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 自适应二值化
            binary = cv2.adaptiveThreshold(
                gray, 
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,  # 邻域大小
                2    # 常数差值
            )
            
            # 降噪
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # 锐化
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
            
        except Exception as e:
            logger.warning(f"图像预处理失败: {str(e)}")
            return image

    def add_files(self, db_id, files, params=None):
        """添加文件到知识库
        
        Args:
            db_id: 数据库ID
            files: 文件路径列表
            params: 额外参数，支持csv_params和excel_params
        """
        logger.info(f"开始添加文件到数据库 {db_id}")
        db = self.get_kb_by_id(db_id)
        
        if db is None:
            # 尝试从知识库获取信息
            collection_info = self.knowledge_base.get_collection_info(db_id)
            if collection_info is not None:
                # 如果集合存在，创建对应的数据库对象
                db = DataBaseLite(
                    name=collection_info.get("name", db_id),
                    description=collection_info.get("description", ""),
                    db_type="text",
                    dimension=collection_info.get("dimension", 1024),
                    embed_model=self.config.embed_model,
                    db_id=db_id,
                    metaname=db_id
                )
                self.data["databases"].append(db)
                self._save_databases()
            else:
                logger.error(f"数据库 {db_id} 不存在")
                return {"message": f"数据库 {db_id} 不存在", "status": "failed"}

        if db.embed_model != self.config.embed_model:
            logger.error(f"Embed model不匹配, {db.embed_model} != {self.config.embed_model}")
            return {"message": f"Embed model不匹配, 当前: {self.config.embed_model}", "status": "failed"}

        # 预处理文件队列
        new_files = []
        for file in files:
            new_file = {
                "file_id": "file_" + hashstr(file + str(time.time())),
                "filename": os.path.basename(file),
                "path": file,
                "type": file.split(".")[-1].lower(),
                "status": "waiting",
                "created_at": time.time()
            }
            db.files.append(new_file)
            new_files.append(new_file)

        from src.core.indexing import chunk
        success_count = 0
        failed_count = 0

        for new_file in new_files:
            file_id = new_file["file_id"]
            idx = self.get_idx_by_fileid(db, file_id)
            db.files[idx]["status"] = "processing"

            try:
                # 处理PDF文件（包含表格）
                if new_file["type"] == "pdf":
                    logger.info(f"处理PDF文件(包含表格识别): {new_file['filename']}")
                    try:
                        # 尝试使用表格识别处理
                        structured_texts = self.process_document_with_tables(new_file["path"], new_file["type"])
                        
                        # 添加结构化内容到向量数据库
                        self.knowledge_base.add_documents(
                            docs=structured_texts,
                            collection_name=db.metaname,
                            file_id=file_id
                        )
                    except Exception as e:
                        logger.warning(f"表格识别处理失败，将使用普通文本处理: {str(e)}")
                        # 如果表格处理失败，回退到普通文本处理
                        texts = self.read_text(new_file["path"])
                        nodes = chunk(texts, params=params)
                        
                        if not nodes:
                            raise ValueError(f"无法从PDF文件中提取内容: {new_file['filename']}")
                        
                        self.knowledge_base.add_documents(
                            docs=[node.text for node in nodes],
                            collection_name=db.metaname,
                            file_id=file_id
                        )
                    
                # 处理其他文本文件
                else:
                    logger.info(f"处理文本文件: {new_file['filename']}")
                    nodes = chunk(new_file["path"], params=params)
                    
                    if not nodes:
                        raise ValueError(f"无法从文件中提取内容: {new_file['filename']}")
                    
                    self.knowledge_base.add_documents(
                        docs=[node.text for node in nodes],
                        collection_name=db.metaname,
                        file_id=file_id
                    )

                idx = self.get_idx_by_fileid(db, file_id)
                db.files[idx]["status"] = "done"
                success_count += 1
                logger.info(f"文件处理成功: {new_file['filename']}")

            except Exception as e:
                error_msg = f"处理文件失败 {new_file['filename']}: {str(e)}"
                logger.error(error_msg)
                logger.error(f"错误详情: {traceback.format_exc()}")
                idx = self.get_idx_by_fileid(db, file_id)
                db.files[idx]["status"] = "failed"
                db.files[idx]["error"] = str(e)
                failed_count += 1

            self._save_databases()

        # 返回处理结果
        if failed_count > 0:
            return {
                "message": f"处理完成，成功: {success_count}，失败: {failed_count}",
                "status": "partial_success",
                "success_count": success_count,
                "failed_count": failed_count
            }
        else:
            return {
                "message": "全部处理完成",
                "status": "success",
                "success_count": success_count
            }

    def get_database_info(self, db_id):
        """获取数据库信息，如果不存在则创建新的数据库"""
        db = self.get_kb_by_id(db_id)
        if db is None:
            try:
                # 生成一个唯一的数据库名称
                import time
                unique_name = f"db_{int(time.time())}"
                
                database_info = self.create_database(
                    database_name=unique_name,
                    description=f"Auto-created database for {db_id}",
                    db_type="text",
                    dimension=1024
                )
                return database_info
            except Exception as e:
                logger.error(f"Failed to create database: {str(e)}")
                raise
        else:
            # 确保embed_model属性存在
            if not hasattr(db, 'embed_model') or db.embed_model is None:
                db.embed_model = self.config.embed_model
                self._save_databases()
            
            # 更新集合信息
            collection_info = self.knowledge_base.get_collection_info(db.metaname)
            db.update(collection_info)
            return db.to_dict()

    def read_text(self, file, params=None):
        support_format = [".pdf", ".txt", ".md"]
        assert os.path.exists(file), "File not found"
        logger.info(f"Try to read file {file}")

        if not os.path.isfile(file):
            logger.error(f"Directory not supported now!")
            raise NotImplementedError("Directory not supported now!")

        if file.endswith(".pdf"):
            if is_text_pdf(file):
                from src.core.filereader import pdfreader
                return pdfreader(file)
            else:
                from src.plugins import pdf2txt
                return pdf2txt(file, return_text=True)

        elif file.endswith(".txt") or file.endswith(".md"):
            from src.core.filereader import plainreader
            return plainreader(file)

        else:
            logger.error(f"File format not supported, only support {support_format}")
            raise Exception(f"File format not supported, only support {support_format}")

    def delete_file(self, db_id, file_id):
        db = self.get_kb_by_id(db_id)
        file_idx_to_delete = self.get_idx_by_fileid(db, file_id)

        self.knowledge_base.client.delete(
            collection_name=db.metaname,
            filter=f"file_id == '{file_id}'"),

        del db.files[file_idx_to_delete]
        self._save_databases()

    def get_file_info(self, db_id, file_id):
        db = self.get_kb_by_id(db_id)
        if db is None:
            return {"message": "database not found"}, 404
        lines = self.knowledge_base.client.query(
            collection_name=db.metaname,
            filter=f"file_id == '{file_id}'",
            output_fields=["id", "text", "file_id", "hash"]
        )
        return {"lines": lines}

    def chunking(self, text, params=None):
        chunk_method = params.get("chunk_method", "fixed")
        chunk_size = params.get("chunk_size", 500)

        """将文本切分成固定大小的块"""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def delete_database(self, db_id):
        db = self.get_kb_by_id(db_id)
        if db is None:
            return {"message": "database not found"}, 404

        self.knowledge_base.client.drop_collection(db.metaname)
        self.data["databases"] = [d for d in self.data["databases"] if d.db_id != db_id]
        self._save_databases()
        return {"message": "删除成功"}

    def get_kb_by_id(self, db_id):
        for db in self.data["databases"]:
            if db.db_id == db_id:
                return db
        return None

    def get_idx_by_fileid(self, db, file_id):
        for idx, f in enumerate(db.files):
            if f["file_id"] == file_id:
                return idx


class DataBaseLite:
    def __init__(self, name, description, db_type, dimension=None, **kwargs) -> None:
        self.name = name
        self.description = description
        self.db_type = db_type
        self.dimension = dimension or 1024
        self.files = []
        self.metaname = "t" + hashstr(name + str(time.time()))
        self.created_at = time.time()
        self.embed_model = kwargs.get('embed_model')
        self.db_id = kwargs.get('db_id') or hashstr(self.metaname)
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    def id2file(self, file_id):
        for file in self.files:
            if file["file_id"] == file_id:
                return file
        return None

    def update(self, metadata):
        self.__dict__.update(metadata)

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "db_type": self.db_type,
            "dimension": self.dimension,
            "files": self.files,
            "metaname": self.metaname,
            "created_at": self.created_at,
            "embed_model": self.embed_model,
            "db_id": self.db_id,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    def __str__(self):
        return self.to_json()

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
            texts = self.read_text(file_path)
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
                self.knowledge_base.add_documents(
                    docs=[info_chunk],
                    collection_name=kb_name,
                    file_id=chunk_id
                )
            except Exception as e:
                logger.error(f"添加文件信息块失败: {str(e)}")
                continue

        # 分批处理文本内容
        batch_size = 30  # 减小批处理大小
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            try:
                self.knowledge_base.add_documents(
                    docs=[node.text for node in batch],
                    collection_name=kb_name,
                    file_id=file_id
                )
            except Exception as e:
                logger.error(f"添加文本块失败: {str(e)}")
                continue

        return True
    except Exception as e:
        logger.error(f"处理文件失败: {str(e)}")
        return False