import requests
import os
import json
import logging
from typing import Optional, Dict, Any, Generator, Union
from src.utils.logging_config import setup_logger

logger = setup_logger("OllamaService")

class OllamaService:
    def __init__(self):
        self.base_url = os.getenv('OLLAMA_API_BASE', 'http://192.168.1.16:11434')
        logger.info(f"初始化Ollama服务，API地址: {self.base_url}")
    
    async def generate_response(
        self, 
        prompt: str, 
        model: str = "deepseek-r1:1.5b",
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """生成响应"""
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_predict": kwargs.get("max_tokens", 2048),
                    "stop": kwargs.get("stop", []),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.1)
                }
            }
            
            logger.debug(f"发送请求到Ollama: {url}")
            logger.debug(f"请求参数: {json.dumps(payload, ensure_ascii=False)}")
            
            response = requests.post(url, json=payload, stream=stream)
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response)
            else:
                result = response.json()
                logger.debug(f"Ollama响应: {json.dumps(result, ensure_ascii=False)}")
                return result.get('response')
                
        except requests.exceptions.RequestException as e:
            logger.error(f"请求Ollama服务失败: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"错误详情: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"调用Ollama时发生错误: {str(e)}")
            return None
            
    def _handle_stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """处理流式响应"""
        try:
            if response is None:
                logger.error("收到空的响应对象")
                yield "错误: 收到空的响应"
                return
                
            for line in response.iter_lines():
                if not line:
                    continue
                    
                try:
                    json_response = json.loads(line)
                    logger.debug(f"收到响应: {json_response}")
                    
                    if 'error' in json_response:
                        error_msg = json_response['error']
                        logger.error(f"Ollama返回错误: {error_msg}")
                        yield f"错误: {error_msg}"
                        return
                        
                    if 'response' in json_response:
                        response_text = json_response['response']
                        if response_text:  # 只在有实际内容时yield
                            logger.debug(f"生成响应文本: {response_text}")
                            yield response_text
                    else:
                        logger.warning(f"响应中没有'response'字段: {json_response}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"解析响应JSON时发生错误: {str(e)}, 原始数据: {line}")
                    continue
                    
        except Exception as e:
            logger.error(f"处理流式响应时发生错误: {str(e)}")
            logger.error("错误详情:", exc_info=True)
            yield f"处理响应时发生错误: {str(e)}"
            return
            
    async def list_models(self) -> Optional[Dict[str, Any]]:
        """获取可用模型列表"""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"获取到的模型列表: {json.dumps(result, ensure_ascii=False)}")
            return result
        except Exception as e:
            logger.error(f"获取模型列表失败: {str(e)}")
            return None
            
    async def check_model_status(self, model: str = "deepseek-r1:7b") -> bool:
        """检查模型状态"""
        try:
            models = await self.list_models()
            if not models:
                return False
                
            available_models = models.get('models', [])
            logger.info(f"可用的模型列表: {[m.get('name') for m in available_models]}")
            
            model_exists = any(m.get('name') == model for m in available_models)
            if model_exists:
                logger.info(f"找到模型: {model}")
            else:
                logger.warning(f"未找到模型: {model}")
            return model_exists
        except Exception as e:
            logger.error(f"检查模型状态失败: {str(e)}")
            return False 