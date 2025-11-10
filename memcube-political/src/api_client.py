"""
OpenAI API 客户端
支持重试、并发和错误处理
"""

import openai
import yaml
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """API响应数据类"""
    success: bool
    content: Any
    error: Optional[str] = None
    usage: Optional[Dict] = None
    model: Optional[str] = None

class OpenAIClient:
    """OpenAI API客户端"""

    def __init__(self, config_path: str = "config/api_keys.yaml"):
        """初始化API客户端"""
        self.config = self._load_config(config_path)
        self.client = self._init_client()
        self.default_config = {
            "temperature": 0.7,
            "max_tokens": 4000,
            "timeout": 60
        }

    def _load_config(self, config_path: str) -> Dict:
        """加载API配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if 'openai' not in config:
                raise ValueError("配置文件中未找到openai配置")

            return config['openai']
        except FileNotFoundError:
            raise FileNotFoundError(f"API配置文件未找到: {config_path}")
        except Exception as e:
            raise Exception(f"加载API配置失败: {e}")

    def _init_client(self) -> openai.OpenAI:
        """初始化OpenAI客户端"""
        try:
            client_config = {
                "api_key": self.config['api_key']
            }

            if 'organization' in self.config:
                client_config["organization"] = self.config['organization']

            if 'base_url' in self.config:
                client_config["base_url"] = self.config['base_url']

            return openai.OpenAI(**client_config)
        except Exception as e:
            logger.error(f"初始化OpenAI客户端失败: {e}")
            raise

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        max_retries: int = 3,
        **kwargs
    ) -> APIResponse:
        """
        发送聊天完成请求

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            max_retries: 最大重试次数
            **kwargs: 其他参数

        Returns:
            APIResponse对象
        """
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"发送请求到模型 {model}，尝试次数: {attempt + 1}")

                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

                content = response.choices[0].message.content
                usage = response.usage.model_dump() if response.usage else None

                logger.debug(f"请求成功，模型: {model}，token使用: {usage}")

                return APIResponse(
                    success=True,
                    content=content,
                    usage=usage,
                    model=model
                )

            except openai.RateLimitError as e:
                logger.warning(f"触发速率限制，等待后重试: {e}")
                wait_time = min(2 ** attempt, 60)  # 指数退避，最大60秒
                time.sleep(wait_time)

            except openai.APIError as e:
                logger.error(f"API错误: {e}")
                if attempt == max_retries:
                    return APIResponse(
                        success=False,
                        error=f"API错误: {e}"
                    )
                time.sleep(1)

            except openai.AuthenticationError as e:
                logger.error(f"认证错误: {e}")
                return APIResponse(
                    success=False,
                    error=f"认证错误: {e}"
                )

            except Exception as e:
                logger.error(f"未知错误: {e}")
                if attempt == max_retries:
                    return APIResponse(
                        success=False,
                        error=f"未知错误: {e}"
                    )
                time.sleep(1)

        return APIResponse(
            success=False,
            error="重试次数耗尽"
        )

    def json_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        max_retries: int = 3,
        **kwargs
    ) -> APIResponse:
        """
        发送JSON格式完成请求

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            max_retries: 最大重试次数
            **kwargs: 其他参数

        Returns:
            APIResponse对象，content为解析后的JSON
        """
        # 添加JSON模式参数
        json_kwargs = {
            "response_format": {"type": "json_object"},
            **kwargs
        }

        response = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            **json_kwargs
        )

        if not response.success:
            return response

        try:
            # 解析JSON响应
            json_content = json.loads(response.content)
            response.content = json_content
            return response
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return APIResponse(
                success=False,
                error=f"JSON解析失败: {e}",
                usage=response.usage,
                model=response.model
            )

    def batch_completion(
        self,
        message_batches: List[List[Dict[str, str]]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        max_workers: int = 5,
        **kwargs
    ) -> List[APIResponse]:
        """
        批量发送聊天完成请求

        Args:
            message_batches: 消息批次列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            max_workers: 最大并发数
            **kwargs: 其他参数

        Returns:
            APIResponse列表
        """
        import concurrent.futures

        def process_batch(messages):
            return self.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(process_batch, batch): i
                for i, batch in enumerate(message_batches)
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                try:
                    result = future.result()
                    results.append((batch_index, result))
                except Exception as e:
                    logger.error(f"批次 {batch_index} 处理失败: {e}")
                    results.append((batch_index, APIResponse(
                        success=False,
                        error=f"批次处理失败: {e}"
                    )))

        # 按原始顺序排序
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

# 全局客户端实例
_client_instance = None

def get_client(config_path: str = "config/api_keys.yaml") -> OpenAIClient:
    """获取全局API客户端实例"""
    global _client_instance
    if _client_instance is None:
        _client_instance = OpenAIClient(config_path)
    return _client_instance