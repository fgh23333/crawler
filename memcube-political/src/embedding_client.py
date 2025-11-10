"""
Embedding客户端 - 支持sentence-transformers和ollama
"""

import numpy as np
import requests
import json
import logging
from typing import List, Union
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class EmbeddingClient:
    """Embedding客户端，支持多种后端"""

    def __init__(self, config: dict):
        """初始化embedding客户端"""
        self.config = config.get('embedding', {})
        self.model_name = self.config.get('model_name', 'bge-m3')
        self.model_type = self.config.get('model_type', 'sentence-transformers')
        self.batch_size = self.config.get('batch_size', 16)
        self.device = self.config.get('device', 'cpu')

        self._model = None
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型"""
        if self.model_type == 'ollama':
            logger.info(f"使用Ollama模型: {self.model_name}")
            # 测试ollama连接
            ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
            try:
                response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [model.get('name', '') for model in models]
                    if self.model_name not in model_names:
                        logger.warning(f"模型 {self.model_name} 未在ollama中找到")
                        logger.info(f"可用模型: {model_names}")
                    else:
                        logger.info(f"✅ Ollama连接成功，模型 {self.model_name} 可用")
                else:
                    logger.error(f"Ollama连接失败: {response.status_code}")
            except Exception as e:
                logger.error(f"无法连接到Ollama服务: {e}")
                logger.info("请确保Ollama服务正在运行")

        elif self.model_type == 'sentence-transformers':
            logger.info(f"加载Sentence Transformers模型: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"✅ 模型加载成功")
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                raise

        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        编码文本为embedding向量

        Args:
            texts: 要编码的文本列表
            show_progress: 是否显示进度

        Returns:
            embedding向量数组
        """
        if not texts:
            return np.array([])

        if self.model_type == 'ollama':
            return self._encode_with_ollama(texts, show_progress)
        elif self.model_type == 'sentence-transformers':
            return self._encode_with_sentence_transformers(texts, show_progress)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _encode_with_ollama(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """使用Ollama进行编码"""
        ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
        embeddings = []

        from tqdm import tqdm
        if show_progress:
            texts = tqdm(texts, desc="Embedding (Ollama)")

        for text in texts:
            try:
                response = requests.post(
                    f"{ollama_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get('embedding', [])
                    embeddings.append(embedding)
                else:
                    logger.error(f"Embedding请求失败: {response.status_code}")
                    # 返回零向量作为fallback
                    embeddings.append([0.0] * 1024)  # bge-m3是1024维

            except Exception as e:
                logger.error(f"编码文本失败: {e}")
                # 返回零向量作为fallback
                embeddings.append([0.0] * 1024)

        return np.array(embeddings)

    def _encode_with_sentence_transformers(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """使用Sentence Transformers进行编码"""
        if self._model is None:
            raise RuntimeError("模型未初始化")

        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

    def get_embedding_dimension(self) -> int:
        """获取embedding维度"""
        if self.model_type == 'ollama':
            # bge-m3 是 1024 维
            return 1024
        elif self.model_type == 'sentence-transformers':
            if self._model is None:
                raise RuntimeError("模型未初始化")
            return self._model.get_sentence_embedding_dimension()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

# 便捷函数
def get_embedding_client(config_path: str = "config/config.yaml") -> EmbeddingClient:
    """
    获取embedding客户端实例

    Args:
        config_path: 配置文件路径

    Returns:
        EmbeddingClient实例
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

    return EmbeddingClient(config)