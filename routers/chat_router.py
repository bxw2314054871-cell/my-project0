import json
import asyncio
from datetime import datetime
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse, Response
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session
from src.core import HistoryManager
from src.core.startup import startup
from src.utils.logging_config import setup_logger
from src.services.ollama_service import OllamaService
logger = setup_logger("chat_router")

# 创建路由器实例
chat = APIRouter(prefix="/chat", tags=["chat"])
# 创建线程池
executor = ThreadPoolExecutor()

refs_pool = {}

@chat.get("/")
async def get_chat_history():
    """获取聊天历史"""
    return {"message": "Hello World"}

@chat.post("/")
async def chat_post(
        query: str = Body(...),
        meta: dict = Body(None),
        history: list = Body(...),
        cur_res_id: str = Body(...)

):
    logger.info(f"=== 开始处理聊天请求 ===")
    logger.info(f"查询内容: {query}")
    logger.info(f"元数据: {meta}")
    logger.info(f"历史记录长度: {len(history)}")
    logger.info(f"当前响应ID: {cur_res_id}")
    
    history_manager = HistoryManager(history)

    def make_chunk(content, status, history):
        chunk_data = {
            "response": content,
            "history": history,
            "model_name": startup.config.model_name,
            "status": status,
            "meta": meta
        }
        logger.debug(f"生成数据块: {chunk_data}")
        return json.dumps(chunk_data, ensure_ascii=False).encode('utf-8') + b"\n"

    async def generate_ollama_response():
        """测试Ollama服务的基本功能"""
        logger.info("开始测试Ollama服务...")
        try:
            if meta.get("enable_retrieval"):
                chunk = make_chunk("", "searching", history=None)
                yield chunk

                new_query, refs = startup.retriever(query, history_manager.messages, meta)
                refs_pool[cur_res_id] = refs
            else:
                new_query = query

            messages = history_manager.get_history_with_msg(new_query, max_rounds=meta.get('history_round'))
            history_manager.add_user(query)
            logger.debug(f"Web history: {history_manager.messages}")
            logger.info(f"准备发送到模型的消息: {query}")

            content = ""
            service = OllamaService()
            
            # 使用配置中的模型名称
            model_name = getattr(startup.model, 'model_name', 'deepseek-r1:32b')
            logger.info(f"使用Ollama模型: {model_name}")
            
            async_response = await service.generate_response(
                prompt=query,
                model=model_name,
                stream=True
            )
            
            if async_response is None:
                logger.error("Ollama服务返回空响应")
                error_chunk = make_chunk("Ollama服务返回空响应", "error", history=history_manager.messages)
                yield error_chunk
                return

            logger.info("开始处理流式响应...")
            for response_text in async_response:
                logger.debug(f"收到响应文本: {response_text}")
                if response_text.startswith("错误:"):
                    error_chunk = make_chunk(response_text, "error", history=history_manager.messages)
                    yield error_chunk
                    return
                    
                content += response_text
                output_tokens = len(content)
                chunk = make_chunk(content, "loading", history=history_manager.update_ai(content))
                yield chunk

            # 记录Token使用情况
            try:
                # 发送最终的响应
                final_chunk = make_chunk(
                    content,
                    "complete",
                    history=history_manager.update_ai(content)
                )
                yield final_chunk
            except Exception as e:
                logger.error(f"记录Token使用情况时发生错误: {str(e)}")


        except Exception as e:
            logger.error(f"Ollama响应处理过程中发生错误: {str(e)}")
            logger.error("错误详情:", exc_info=True)
            error_chunk = make_chunk(f"处理响应时发生错误: {str(e)}", "error", history=history_manager.messages)
            yield error_chunk

    async def generate_response():
        logger.info("=== 开始标准响应生成 ===")
        if meta.get("enable_retrieval"):
            logger.info("启用检索功能")
            chunk = make_chunk("", "searching", history=None)
            yield chunk

            new_query, refs = startup.retriever(query, history_manager.messages, meta)
            refs_pool[cur_res_id] = refs
        else:
            new_query = query
            logger.info("未启用检索功能，使用原始查询")

        messages = history_manager.get_history_with_msg(new_query, max_rounds=meta.get('history_round'))
        history_manager.add_user(query)
        logger.debug(f"Web history: {history_manager.messages}")
        logger.info(f"最终发送给模型的消息: {messages}")

        content = ""

        try:
            # 获取预测结果
            logger.info("开始调用模型预测...")
            logger.info(f"模型类型: {type(startup.model)}")
            logger.info(f"模型名称: {getattr(startup.model, 'model_name', 'unknown')}")
            
            response_stream = startup.model.predict(messages, stream=True)
            logger.info(f"获取到响应流，类型: {type(response_stream)}")

            # 处理不同类型的响应流
            if hasattr(response_stream, '__aiter__'):  # 异步迭代器
                logger.info("使用异步迭代器处理响应流")
                async for delta in response_stream:
                    logger.debug(f"收到响应片段: {delta}")
                    if not delta.content:
                        logger.debug("跳过空内容片段")
                        continue

                    if hasattr(delta, 'is_full') and delta.is_full:
                        content += delta.content
                        output_tokens = len(content)  # 完整响应的token数
                        logger.info(f"收到完整响应，长度: {output_tokens}")
                    else:
                        content += delta.content
                        output_tokens = len(content)  # 累计输出token数
                        logger.debug(f"累计内容长度: {output_tokens}")

                    chunk = make_chunk(content, "loading", history=history_manager.update_ai(content))
                    yield chunk
            else:
                logger.info("使用普通迭代器处理响应流")
                for delta in response_stream:
                    logger.debug(f"收到响应片段: {delta}")
                    if not delta.content:
                        logger.debug("跳过空内容片段")
                        continue

                    if hasattr(delta, 'is_full') and delta.is_full:
                        content += delta.content
                        output_tokens = len(content)
                        logger.info(f"收到完整响应，长度: {output_tokens}")
                    else:
                        content += delta.content
                        output_tokens = len(content)
                        logger.debug(f"累计内容长度: {output_tokens}")

                    chunk = make_chunk(content, "loading", history=history_manager.update_ai(content))
                    yield chunk
                    
        except Exception as e:
            logger.error(f"模型预测过程中发生错误: {str(e)}")
            logger.error("错误详情:", exc_info=True)
            error_chunk = make_chunk(f"模型响应出错: {str(e)}", "error", history=history_manager.messages)
            yield error_chunk
            return

        try:
            logger.info(f"发送最终响应，内容长度: {len(content)}")
            # 发送最终的响应，包含token使用信息
            final_chunk = make_chunk(
                content,
                "complete",
                history=history_manager.update_ai(content)
            )
            yield final_chunk
            logger.info("=== 响应生成完成 ===")
        except Exception as e:
            logger.error(f"发送最终响应时发生错误: {str(e)}")
            logger.error("错误详情:", exc_info=True)
            
    def get_response():
        # 检查是否是Ollama模型
        model_name = getattr(startup.model, 'model_name', '')
        model_provider = getattr(startup.config, 'model_provider', '')
        
        logger.info(f"=== 选择响应生成方式 ===")
        logger.info(f"模型名称: {model_name}")
        logger.info(f"模型提供商: {model_provider}")
        logger.info(f"模型类型: {type(startup.model)}")
        
        # 检查模型提供商或模型名称
        if (model_provider == 'ollama' or 
            (hasattr(startup.model, 'model_name') and 
             startup.model.model_name in ['deepseek-r1:32b', 'deepseek-r1:7b', 'deepseek-r1:32b'])):
            logger.info(f"使用Ollama模型处理请求: {model_name}")
            return StreamingResponse(generate_ollama_response(), media_type='application/json')
        else:
            logger.info(f"使用标准模型处理请求: {model_name} (provider: {model_provider})")
            return StreamingResponse(generate_response(), media_type='application/json')

    return get_response()

@chat.post("/call")
async def call(query: str = Body(...), meta: dict = Body(None)):
    """非流式响应接口"""
    logger.info(f"收到非流式请求: {query}")
    try:
        # 直接调用模型预测，不使用异步执行器
        response = startup.model.predict(query, stream=False)
        
        logger.info(f"模型响应类型: {type(response)}")
        
        if response is None:
            logger.error("模型返回空响应")
            return {"response": "模型返回空响应"}
        
        # 处理不同类型的响应
        if hasattr(response, 'content'):
            # 如果是 GeneralResponse 对象
            content = response.content
            logger.info(f"获取到响应内容: {content}")
        elif hasattr(response, '__iter__'):
            # 如果是生成器或迭代器
            logger.info("检测到生成器响应，收集所有内容")
            content = ""
            for chunk in response:
                if hasattr(chunk, 'content'):
                    content += chunk.content
                else:
                    content += str(chunk)
            logger.info(f"收集到的完整内容: {content}")
        else:
            # 其他类型，直接转换为字符串
            content = str(response)
            logger.info(f"转换为字符串的内容: {content}")
            
        logger.debug({"query": query, "response": content})
        return {"response": content}
        
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        logger.error("错误详情:", exc_info=True)
        return {"response": f"处理请求失败: {str(e)}"}

@chat.get("/refs")
def get_refs(cur_res_id: str):
    global refs_pool
    refs = refs_pool.pop(cur_res_id, None)
    
    # 确保返回的是可序列化的数据
    if refs is not None:
        # 处理 knowledge_base 结果
        if "knowledge_base" in refs and "results" in refs["knowledge_base"]:
            for i, item in enumerate(refs["knowledge_base"]["results"]):
                if hasattr(item, "__dict__"):  # 检查是否是对象而非字典
                    refs["knowledge_base"]["results"][i] = dict(item)
                    
        # 处理 graph_base 结果
        if "graph_base" in refs and "results" in refs["graph_base"]:
            if hasattr(refs["graph_base"]["results"], "__dict__"):
                refs["graph_base"]["results"] = dict(refs["graph_base"]["results"])
    
    return {"refs": refs}