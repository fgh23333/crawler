"""
API 客户端
统一支持OpenAI格式API，包含重试、并发和错误处理
支持OpenAI官方API、Gemini API（OpenAI兼容格式）以及其他兼容API
"""

import openai
import yaml
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """API响应数据类"""
    success: bool
    content: Any
    error: Optional[str] = None
    usage: Optional[Dict] = None
    model: Optional[str] = None

class UnifiedAPIClient:
    """统一API客户端，支持OpenAI格式的各种API服务"""

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

            # 检查新的统一配置格式
            if 'api_key' in config:
                return {
                    'type': 'openai_format',
                    'api_key': config['api_key'],
                    'base_url': config.get('base_url'),
                    'organization': config.get('organization', '')
                }
            # 检查旧格式配置（向后兼容）
            elif 'openai' in config:
                return {'type': 'openai_format', **config['openai']}
            elif 'gemini' in config:
                logger.warning("检测到旧的Gemini配置格式，建议更新为统一OpenAI格式")
                # 将Gemini配置转换为OpenAI格式
                gemini_config = config['gemini']
                return {
                    'type': 'openai_format',
                    'api_key': gemini_config['api_key'],
                    'base_url': gemini_config.get('base_url', 'https://generativelanguage.googleapis.com/v1beta/openai/'),
                    'organization': ''
                }
            else:
                raise ValueError("配置文件中未找到有效的API配置")

        except FileNotFoundError:
            raise FileNotFoundError(f"API配置文件未找到: {config_path}")
        except Exception as e:
            raise Exception(f"加载API配置失败: {e}")

    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            client_config = {
                "api_key": self.config['api_key']
            }

            # 添加base_url（如果配置了）
            if 'base_url' in self.config and self.config['base_url']:
                client_config["base_url"] = self.config['base_url']
                logger.info(f"使用自定义API端点: {self.config['base_url']}")

            # 添加organization（如果配置了）
            if 'organization' in self.config and self.config['organization']:
                client_config["organization"] = self.config['organization']

            return openai.OpenAI(**client_config)

        except Exception as e:
            logger.error(f"初始化API客户端失败: {e}")
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
                logger.debug(f"发送API请求到模型 {model}，尝试次数: {attempt + 1}")

                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

                content = response.choices[0].message.content
                usage = response.usage.model_dump() if response.usage else None

                logger.debug(f"API请求成功，模型: {model}，token使用: {usage}")

                return APIResponse(
                    success=True,
                    content=content,
                    usage=usage,
                    model=model
                )

            except openai.RateLimitError as e:
                logger.warning(f"触发速率限制，等待后重试: {e}")
                wait_time = min(2 ** attempt, 60)
                time.sleep(wait_time)

            except openai.APIError as e:
                logger.error(f"API错误: {e}")
                if attempt == max_retries:
                    return APIResponse(success=False, error=f"API错误: {e}", content="")
                time.sleep(1)

            except openai.AuthenticationError as e:
                logger.error(f"认证错误: {e}")
                return APIResponse(success=False, error=f"认证错误: {e}", content="")

            except Exception as e:
                logger.error(f"未知错误: {e}")
                if attempt == max_retries:
                    return APIResponse(success=False, error=f"未知错误: {e}", content="")
                time.sleep(1)

        return APIResponse(success=False, error="重试次数耗尽", content="")

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
        # 添加JSON格式参数
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
            import json

            # 检查response.content是否为None或空
            if response.content is None or response.content == "":
                logger.warning("API返回内容为空或None")
                return APIResponse(
                    success=False,
                    error="API返回内容为空",
                    usage=response.usage,
                    model=response.model,
                    content=None
                )

            # 确保content是字符串类型
            if isinstance(response.content, str):
                json_content = json.loads(response.content)
            elif isinstance(response.content, (dict, list)):
                # 如果已经是解析好的JSON对象，直接使用
                json_content = response.content
            else:
                logger.error(f"未知的响应类型: {type(response.content)}, 内容: {response.content}")
                return APIResponse(
                    success=False,
                    error=f"未知的响应类型: {type(response.content)}",
                    usage=response.usage,
                    model=response.model,
                    content=None
                )

            response.content = json_content
            return response
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 原始内容: {response.content}")
            return APIResponse(
                success=False,
                error=f"JSON解析失败: {e}",
                usage=response.usage,
                model=response.model,
                content=None
            )
        except Exception as e:
            logger.error(f"处理响应时发生未知错误: {e}")
            return APIResponse(
                success=False,
                error=f"处理响应失败: {e}",
                usage=response.usage,
                model=response.model,
                content=None
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
                        error=f"批次处理失败: {e}",
                        content=""
                    )))

        # 按原始顺序排序
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    def get_model_info(self) -> Dict[str, Any]:
        """获取当前API配置信息"""
        return {
            "config_type": self.config.get('type', 'unknown'),
            "base_url": self.config.get('base_url', 'default'),
            "has_api_key": bool(self.config.get('api_key')),
            "has_organization": bool(self.config.get('organization'))
        }

# 全局客户端实例
_client_instance = None

def get_client(config_path: str = "config/api_keys.yaml") -> UnifiedAPIClient:
    """获取全局API客户端实例"""
    global _client_instance
    if _client_instance is None:
        _client_instance = UnifiedAPIClient(config_path)
    return _client_instance

# 为了向后兼容，保留OpenAIClient别名
OpenAIClient = UnifiedAPIClient