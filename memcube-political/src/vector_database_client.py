import uuid
import os
import logging
import yaml
import numpy as np
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

def _to_uuid(text: str) -> str:
    """Convert text to UUID"""
    try:
        uuid.UUID(text)
        return text
    except:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text.encode('utf-8', errors='ignore').decode('utf-8')))

class VectorDatabaseClient(ABC):
    """向量数据库客户端抽象基类"""

    @abstractmethod
    def connect(self) -> bool:
        """连接到数据库"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开数据库连接"""
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """创建集合"""
        pass

    @abstractmethod
    def insert(self, collection_name: str, ids: List[str], vectors: List[np.ndarray],
               metadatas: List[Dict[str, Any]]) -> bool:
        """插入向量"""
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: np.ndarray,
               top_k: int = 10, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """向量搜索"""
        pass

    @abstractmethod
    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """删除向量"""
        pass

    @abstractmethod
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        """列出所有集合"""
        pass

    @abstractmethod
    def batch_search(self, collection_name: str, query_vectors: List[np.ndarray],
                     top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """批量向量搜索"""
        pass


class ChromaClient(VectorDatabaseClient):
    """Chroma向量数据库客户端（轻量级本地存储）"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.collections = {}

    def connect(self) -> bool:
        """连接到Chroma数据库"""
        try:
            import chromadb

            # 创建Chroma客户端
            persist_directory = self.config.get('persist_directory')
            if persist_directory:
                Path(persist_directory).mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(path=persist_directory)
            else:
                self.client = chromadb.Client()

            logger.info("成功连接到Chroma数据库")
            return True

        except ImportError:
            logger.error("ChromaDB未安装，请运行: pip install chromadb")
            return False
        except Exception as e:
            logger.error(f"连接Chroma数据库失败: {e}")
            return False

    def disconnect(self):
        """断开Chroma连接"""
        if self.client:
            self.collections.clear()
            logger.info("Chroma连接已断开")

    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """创建Chroma集合"""
        try:
            if collection_name in self.collections:
                logger.warning(f"Chroma集合 {collection_name} 已存在")
                return True

            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            self.collections[collection_name] = collection
            logger.info(f"创建Chroma集合 {collection_name} 成功")
            return True
        except Exception as e:
            logger.error(f"创建Chroma集合失败: {e}")
            return False

    def insert(self, collection_name: str, ids: List[str], vectors: List[np.ndarray],
               metadatas: List[Dict[str, Any]]) -> bool:
        """插入向量到Chroma"""
        try:
            if collection_name not in self.collections:
                self.create_collection(collection_name, len(vectors[0]))

            collection = self.collections[collection_name]

            # 转换向量为列表格式
            vector_list = [vector.tolist() if isinstance(vector, np.ndarray) else vector
                          for vector in vectors]

            # 确保ids长度匹配
            if len(ids) != len(vectors):
                # 生成唯一ID
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

            collection.add(
                ids=ids,
                embeddings=vector_list,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            logger.error(f"插入Chroma向量失败: {e}")
            return False

    def search(self, collection_name: str, query_vector: np.ndarray,
               top_k: int = 10, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chroma向量搜索"""
        try:
            if collection_name not in self.collections:
                logger.error(f"Chroma集合 {collection_name} 不存在")
                return []

            collection = self.collections[collection_name]

            query_vector_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

            results = collection.query(
                query_embeddings=[query_vector_list],
                n_results=top_k,
                where=where
            )

            # 格式化结果
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    formatted_results.append({
                        'id': doc_id,
                        'score': results['distances'][0][i] if results['distances'] else 0,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    })

            return formatted_results
        except Exception as e:
            logger.error(f"Chroma向量搜索失败: {e}")
            return []

    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """从Chroma删除向量"""
        try:
            if collection_name not in self.collections:
                return False

            collection = self.collections[collection_name]
            collection.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"删除Chroma向量失败: {e}")
            return False

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取Chroma集合统计信息"""
        try:
            if collection_name not in self.collections:
                return {}

            collection = self.collections[collection_name]
            count = collection.count()

            return {
                'name': collection_name,
                'count': count,
                'metadata': collection.metadata
            }
        except Exception as e:
            logger.error(f"获取Chroma集合统计信息失败: {e}")
            return {}

    def list_collections(self) -> List[str]:
        """列出所有Chroma集合"""
        try:
            return list(self.collections.keys())
        except Exception as e:
            logger.error(f"列出Chroma集合失败: {e}")
            return []

    def check_concepts_exist(self, collection_name: str, concept_ids: List[str]) -> Dict[str, bool]:
        """检查指定概念是否已存在于Chroma向量数据库中"""
        try:
            # 检查集合是否存在
            if collection_name not in self.collections:
                logger.info(f"Chroma集合 {collection_name} 不存在，需要创建")
                return {concept_id: False for concept_id in concept_ids}

            collection = self.collections[collection_name]

            # 获取集合统计信息
            count = collection.count()
            if count == 0:
                logger.info(f"Chroma集合 {collection_name} 为空，需要添加数据")
                return {concept_id: False for concept_id in concept_ids}

            # 检查特定概念是否存在
            existing_concepts = set()

            # 分批查询以避免一次性查询太多数据
            batch_size = 100
            for i in range(0, len(concept_ids), batch_size):
                batch_ids = concept_ids[i:i + batch_size]

                # 使用where子句查询
                try:
                    results = collection.get(
                        where={"name": {"$in": batch_ids}},
                        include=['metadatas']
                    )

                    # 提取存在的概念ID
                    for metadata in results['metadatas']:
                        concept_name = metadata.get('name')
                        if concept_name and concept_name in batch_ids:
                            existing_concepts.add(concept_name)
                            logger.debug(f"概念 '{concept_name}' 已存在于Chroma中")

                except Exception as batch_error:
                    logger.warning(f"批量查询Chroma概念失败: {batch_error}")
                    # 如果批量查询失败，降级为逐个查询
                    for concept_id in batch_ids:
                        try:
                            result = collection.get(
                                where={"name": concept_id},
                                include=['metadatas']
                            )
                            if result['metadatas']:
                                existing_concepts.add(concept_id)
                        except:
                            continue

            # 返回检查结果
            result = {concept_id: (concept_id in existing_concepts) for concept_id in concept_ids}

            existing_count = sum(result.values())
            logger.info(f"Chroma检查完成: {existing_count}/{len(concept_ids)} 个概念已存在")

            return result

        except Exception as e:
            logger.error(f"检查Chroma中概念存在性失败: {e}")
            # 发生错误时，假设都不存在，重新添加
            return {concept_id: False for concept_id in concept_ids}

    def batch_search(self, collection_name: str, query_vectors: List[np.ndarray],
                     top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """Chroma批量向量搜索"""
        try:
            if collection_name not in self.collections:
                logger.error(f"Chroma集合 {collection_name} 不存在")
                return []

            collection = self.collections[collection_name]

            query_vectors_list = [vector.tolist() if isinstance(vector, np.ndarray) else vector
                                 for vector in query_vectors]

            results = collection.query(
                query_embeddings=query_vectors_list,
                n_results=top_k
            )

            # 格式化批量结果
            batch_results = []
            if results['ids']:
                for batch_idx in range(len(results['ids'])):
                    batch_formatted_results = []
                    if results['ids'][batch_idx]:
                        for i, doc_id in enumerate(results['ids'][batch_idx]):
                            batch_formatted_results.append({
                                'id': doc_id,
                                'score': results['distances'][batch_idx][i] if results['distances'] and batch_idx < len(results['distances']) else 0,
                                'metadata': results['metadatas'][batch_idx][i] if results['metadatas'] and batch_idx < len(results['metadatas']) and results['metadatas'][batch_idx] and i < len(results['metadatas'][batch_idx]) else {}
                            })
                    batch_results.append(batch_formatted_results)

            return batch_results
        except Exception as e:
            logger.error(f"Chroma批量向量搜索失败: {e}")
            return []


class QdrantClient(VectorDatabaseClient):
    """Qdrant向量数据库客户端（高性能）"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None

    def connect(self) -> bool:
        """连接到Qdrant数据库"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.client = QdrantClient(
                host=self.config['host'],
                port=self.config['port']
            )

            # 测试连接
            self.client.get_collections()

            logger.info(f"成功连接到Qdrant数据库: {self.config['host']}:{self.config['port']}")
            return True

        except ImportError:
            logger.error("Qdrant客户端未安装，请运行: pip install qdrant-client")
            return False
        except Exception as e:
            logger.error(f"连接Qdrant数据库失败: {e}")
            return False

    def disconnect(self):
        """断开Qdrant连接"""
        if self.client:
            self.client.close()
            logger.info("Qdrant连接已断开")

    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """创建Qdrant集合"""
        try:
            from qdrant_client.models import Distance, VectorParams

            distance_map = {
                'Cosine': Distance.COSINE,
                'Euclidean': Distance.EUCLID,
                'Dot': Distance.DOT
            }
            distance_type = distance_map.get(self.config.get('distance', 'Cosine'), Distance.COSINE)

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_type
                )
            )
            logger.info(f"创建Qdrant集合 {collection_name} 成功")
            return True
        except Exception as e:
            # 集合可能已存在
            if "already exists" in str(e):
                logger.warning(f"Qdrant集合 {collection_name} 已存在")
                return True
            logger.error(f"创建Qdrant集合失败: {e}")
            return False

    def insert(self, collection_name: str, ids: List[str], vectors: List[np.ndarray],
               metadatas: List[Dict[str, Any]]) -> bool:
        """插入向量到Qdrant"""
        try:
            from qdrant_client.models import PointStruct

            points = []
            for i, (doc_id, vector, metadata) in enumerate(zip(ids, vectors, metadatas)):
                point = PointStruct(
                    id=_to_uuid(doc_id),
                    vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
                    payload=metadata
                )
                points.append(point)

            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            return True
        except Exception as e:
            logger.error(f"插入Qdrant向量失败: {e}")
            return False

    def search(self, collection_name: str, query_vector: np.ndarray,
               top_k: int = 10, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Qdrant向量搜索"""
        try:
            from qdrant_client.models import Filter

            search_filter = None
            if where:
                # 构建Qdrant过滤器
                search_filter = Filter(**where)

            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
                limit=top_k,
                query_filter=search_filter
            )

            formatted_results = []
            for hit in results:
                formatted_results.append({
                    'id': str(hit.id),
                    'score': hit.score,
                    'metadata': hit.payload
                })

            return formatted_results
        except Exception as e:
            logger.error(f"Qdrant向量搜索失败: {e}")
            return []

    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """从Qdrant删除向量"""
        try:
            from qdrant_client.models import Filter

            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        {"key": "id", "match": {"any": ids}}
                    ]
                )
            )
            return True
        except Exception as e:
            logger.error(f"删除Qdrant向量失败: {e}")
            return False

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取Qdrant集合统计信息"""
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'count': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"获取Qdrant集合统计信息失败: {e}")
            return {}

    def list_collections(self) -> List[str]:
        """列出所有Qdrant集合"""
        try:
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            logger.error(f"列出Qdrant集合失败: {e}")
            return []

    def check_concepts_exist(self, collection_name: str, concept_ids: List[str]) -> Dict[str, bool]:
        """检查指定概念是否已存在于向量数据库中"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny

            # 检查集合是否存在
            if collection_name not in self.list_collections():
                logger.info(f"Qdrant集合 {collection_name} 不存在，需要创建")
                return {concept_id: False for concept_id in concept_ids}

            # 获取集合统计信息
            stats = self.get_collection_stats(collection_name)
            if not stats or stats.get('count', 0) == 0:
                logger.info(f"Qdrant集合 {collection_name} 为空，需要添加数据")
                return {concept_id: False for concept_id in concept_ids}

            # 检查特定概念是否存在
            existing_concepts = set()

            # 分批查询以避免一次性查询太多数据
            batch_size = 100
            for i in range(0, len(concept_ids), batch_size):
                batch_ids = concept_ids[i:i + batch_size]

                # 使用过滤器查询
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key="name",
                            match=MatchAny(any=batch_ids)
                        )
                    ]
                )

                # 查询结果（只返回ID，不返回向量）
                search_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_condition,
                    limit=len(batch_ids),
                    with_payload=True,
                    with_vectors=False
                )

                # 提取存在的概念ID
                for point in search_result[0]:  # scroll返回(points, next_page_offset)
                    concept_name = point.payload.get('name', point.id)
                    if concept_name in batch_ids:
                        existing_concepts.add(concept_name)
                        logger.debug(f"概念 '{concept_name}' 已存在于Qdrant中")

            # 返回检查结果
            result = {concept_id: (concept_id in existing_concepts) for concept_id in concept_ids}

            existing_count = sum(result.values())
            logger.info(f"Qdrant检查完成: {existing_count}/{len(concept_ids)} 个概念已存在")

            return result

        except Exception as e:
            logger.error(f"检查Qdrant中概念存在性失败: {e}")
            # 发生错误时，假设都不存在，重新添加
            return {concept_id: False for concept_id in concept_ids}

    def batch_search(self, collection_name: str, query_vectors: List[np.ndarray],
                     top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """Qdrant批量向量搜索"""
        try:
            batch_results = []
            for query_vector in query_vectors:
                results = self.search(collection_name, query_vector, top_k)
                batch_results.append(results)
            return batch_results
        except Exception as e:
            logger.error(f"Qdrant批量向量搜索失败: {e}")
            return []


class FAISSClient(VectorDatabaseClient):
    """FAISS向量数据库客户端（内存高性能）"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.indices = {}
        self.metadata_stores = {}

    def connect(self) -> bool:
        """连接到FAISS（初始化）"""
        try:
            import faiss

            # 创建索引目录
            index_path = Path(self.config.get('index_path', './data/faiss_index'))
            index_path.mkdir(parents=True, exist_ok=True)

            logger.info("FAISS初始化成功")
            return True
        except ImportError:
            logger.error("FAISS未安装，请运行: pip install faiss-cpu 或 faiss-gpu")
            return False
        except Exception as e:
            logger.error(f"FAISS初始化失败: {e}")
            return False

    def disconnect(self):
        """断开FAISS连接（保存索引）"""
        try:
            import faiss

            index_path = Path(self.config.get('index_path', './data/faiss_index'))

            for collection_name, index in self.indices.items():
                index_file = index_path / f"{collection_name}.index"
                metadata_file = index_path / f"{collection_name}_metadata.pkl"

                # 保存索引
                faiss.write_index(index, str(index_file))

                # 保存元数据
                if collection_name in self.metadata_stores:
                    with open(metadata_file, 'wb') as f:
                        pickle.dump(self.metadata_stores[collection_name], f)

            logger.info("FAISS索引已保存")
        except Exception as e:
            logger.error(f"保存FAISS索引失败: {e}")

    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """创建FAISS索引"""
        try:
            import faiss

            if collection_name in self.indices:
                logger.warning(f"FAISS索引 {collection_name} 已存在")
                return True

            index_type = self.config.get('index_type', 'IVF_PQ')
            dimension = vector_size

            if index_type == 'Flat':
                index = faiss.IndexFlatIP(dimension)
            elif index_type == 'IVF_PQ':
                nlist = min(int(np.sqrt(1000)), dimension)  # 聚类数量
                m = min(64, dimension // 4)  # PQ编码维度
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
            elif index_type == 'HNSW':
                index = faiss.IndexHNSWFlat(dimension, 32)
            else:
                index = faiss.IndexFlatIP(dimension)

            self.indices[collection_name] = index
            self.metadata_stores[collection_name] = {}

            logger.info(f"创建FAISS索引 {collection_name} 成功")
            return True
        except Exception as e:
            logger.error(f"创建FAISS索引失败: {e}")
            return False

    def insert(self, collection_name: str, ids: List[str], vectors: List[np.ndarray],
               metadatas: List[Dict[str, Any]]) -> bool:
        """插入向量到FAISS"""
        try:
            import faiss

            if collection_name not in self.indices:
                self.create_collection(collection_name, len(vectors[0]))

            index = self.indices[collection_name]
            metadata_store = self.metadata_stores[collection_name]

            # 转换向量为numpy数组
            vectors_array = np.array([vector if isinstance(vector, np.ndarray) else np.array(vector)
                                     for vector in vectors], dtype=np.float32)

            # 归一化向量（用于余弦相似度）
            faiss.normalize_L2(vectors_array)

            # 添加到索引
            start_idx = index.ntotal
            index.add(vectors_array)

            # 存储元数据
            for i, (doc_id, metadata) in enumerate(zip(ids, metadatas)):
                metadata_store[start_idx + i] = {
                    'id': doc_id,
                    'metadata': metadata
                }

            return True
        except Exception as e:
            logger.error(f"插入FAISS向量失败: {e}")
            return False

    def search(self, collection_name: str, query_vector: np.ndarray,
               top_k: int = 10, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """FAISS向量搜索"""
        try:
            import faiss

            if collection_name not in self.indices:
                logger.error(f"FAISS索引 {collection_name} 不存在")
                return []

            index = self.indices[collection_name]
            metadata_store = self.metadata_stores[collection_name]

            # 准备查询向量
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector, dtype=np.float32)
            else:
                query_vector = query_vector.astype(np.float32)

            # 归一化查询向量
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            faiss.normalize_L2(query_vector)

            # 搜索
            scores, indices = index.search(query_vector, top_k)

            # 格式化结果
            formatted_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx in metadata_store:
                    result = {
                        'id': metadata_store[idx]['id'],
                        'score': float(score),
                        'metadata': metadata_store[idx]['metadata']
                    }

                    # 应用过滤器
                    if where is None or self._match_filters(result['metadata'], where):
                        formatted_results.append(result)

            return formatted_results
        except Exception as e:
            logger.error(f"FAISS向量搜索失败: {e}")
            return []

    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """检查元数据是否匹配过滤器"""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """从FAISS删除向量（FAISS不支持删除，需要重建索引）"""
        logger.warning("FAISS不支持删除操作，建议使用其他向量数据库")
        return False

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取FAISS集合统计信息"""
        try:
            if collection_name not in self.indices:
                return {}

            index = self.indices[collection_name]
            metadata_store = self.metadata_stores[collection_name]

            return {
                'name': collection_name,
                'count': index.ntotal,
                'index_type': self.config.get('index_type', 'IVF_PQ'),
                'dimension': index.d if hasattr(index, 'd') else 0
            }
        except Exception as e:
            logger.error(f"获取FAISS集合统计信息失败: {e}")
            return {}

    def list_collections(self) -> List[str]:
        """列出所有FAISS集合"""
        return list(self.indices.keys())

    def check_concepts_exist(self, collection_name: str, concept_ids: List[str]) -> Dict[str, bool]:
        """检查指定概念是否已存在于FAISS索引中"""
        try:
            # 检查集合是否存在
            if collection_name not in self.indices:
                logger.info(f"FAISS索引 {collection_name} 不存在，需要创建")
                return {concept_id: False for concept_id in concept_ids}

            index = self.indices[collection_name]
            metadata_store = self.metadata_stores[collection_name]

            # 获取索引统计信息
            if index.ntotal == 0:
                logger.info(f"FAISS索引 {collection_name} 为空，需要添加数据")
                return {concept_id: False for concept_id in concept_ids}

            # 检查特定概念是否存在
            existing_concepts = set()

            # FAISS存储的是索引位置到ID的映射
            for i in range(index.ntotal):
                # 获取存储的ID
                if i in metadata_store:
                    metadata = metadata_store[i]
                    concept_name = metadata.get('name')
                    if concept_name and concept_name in concept_ids:
                        existing_concepts.add(concept_name)
                        logger.debug(f"概念 '{concept_name}' 已存在于FAISS中")

            # 返回检查结果
            result = {concept_id: (concept_id in existing_concepts) for concept_id in concept_ids}

            existing_count = sum(result.values())
            logger.info(f"FAISS检查完成: {existing_count}/{len(concept_ids)} 个概念已存在")

            return result

        except Exception as e:
            logger.error(f"检查FAISS中概念存在性失败: {e}")
            # 发生错误时，假设都不存在，重新添加
            return {concept_id: False for concept_id in concept_ids}

    def batch_search(self, collection_name: str, query_vectors: List[np.ndarray],
                     top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """FAISS批量向量搜索"""
        try:
            import faiss

            if collection_name not in self.indices:
                logger.error(f"FAISS索引 {collection_name} 不存在")
                return []

            index = self.indices[collection_name]
            metadata_store = self.metadata_stores[collection_name]

            # 准备查询向量
            query_vectors_array = np.array([
                vector if isinstance(vector, np.ndarray) else np.array(vector)
                for vector in query_vectors
            ], dtype=np.float32)

            # 归一化查询向量
            faiss.normalize_L2(query_vectors_array)

            # 批量搜索
            scores, indices = index.search(query_vectors_array, top_k)

            # 格式化批量结果
            batch_results = []
            for batch_idx in range(len(query_vectors)):
                batch_formatted_results = []
                for score, idx in zip(scores[batch_idx], indices[batch_idx]):
                    if idx != -1 and idx in metadata_store:
                        result = {
                            'id': metadata_store[idx]['id'],
                            'score': float(score),
                            'metadata': metadata_store[idx]['metadata']
                        }
                        batch_formatted_results.append(result)
                batch_results.append(batch_formatted_results)

            return batch_results
        except Exception as e:
            logger.error(f"FAISS批量向量搜索失败: {e}")
            return []


def get_vector_database_client(config_path: str = "config/config.yaml") -> Optional[VectorDatabaseClient]:
    """获取向量数据库客户端实例"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        vector_config = config.get('vector_database', {})

        if not vector_config.get('enabled', False):
            logger.info("向量数据库未启用，使用内存模式")
            return None

        db_type = vector_config.get('type', 'chroma').lower()

        if db_type == 'chroma':
            return ChromaClient(vector_config['chroma'])
        elif db_type == 'qdrant':
            return QdrantClient(vector_config['qdrant'])
        elif db_type == 'faiss':
            return FAISSClient(vector_config['faiss'])
        else:
            logger.error(f"不支持的向量数据库类型: {db_type}")
            return None

    except Exception as e:
        logger.error(f"初始化向量数据库客户端失败: {e}")
        return None


# 全局客户端实例
_vector_client_instance = None

def get_vector_client(config_path: str = "config/config.yaml") -> Optional[VectorDatabaseClient]:
    """获取全局向量数据库客户端实例"""
    global _vector_client_instance
    if _vector_client_instance is None:
        _vector_client_instance = get_vector_database_client(config_path)
        if _vector_client_instance:
            _vector_client_instance.connect()
    return _vector_client_instance


def close_vector_client():
    """关闭全局向量数据库客户端连接"""
    global _vector_client_instance
    if _vector_client_instance:
        _vector_client_instance.disconnect()
        _vector_client_instance = None


# 政治理论概念向量搜索工具类
class PoliticalTheoryVectorSearch:
    """政治理论概念向量搜索工具"""

    def __init__(self, vector_client: Optional[VectorDatabaseClient] = None):
        self.vector_client = vector_client or get_vector_client()
        self.collection_name = "political_concepts"

    def index_concepts(self, concepts: List[Dict[str, Any]], embeddings: List[np.ndarray]) -> bool:
        """索引政治理论概念"""
        if not self.vector_client:
            logger.error("向量数据库客户端未初始化")
            return False

        if len(concepts) == 0 or len(embeddings) == 0:
            logger.warning("没有概念或embedding数据需要索引")
            return True

        # 获取配置，检查是否启用存在性检查
        try:
            # 直接读取配置文件避免循环导入
            import yaml
            # 获取项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            config_path = os.path.join(project_root, 'config', 'config.yaml')

            logger.debug(f"尝试读取配置文件: {config_path}")

            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                vector_config = config.get('vector_database', {})
                options = vector_config.get('options', {})
                check_existing = options.get('check_existing', True)
                logger.info(f"配置文件加载成功，check_existing={check_existing}")
            else:
                logger.warning(f"配置文件不存在: {config_path}，使用默认设置")
                check_existing = True  # 默认启用
        except Exception as e:
            logger.error(f"读取配置文件失败，使用默认设置: {e}")
            check_existing = True  # 默认启用

        # 检查哪些概念已经存在
        concept_names = [concept.get('name', '') for concept in concepts if concept.get('name')]
        existing_check = {}

        if check_existing and concept_names:
            logger.info(f"检查 {len(concept_names)} 个概念在向量数据库中的存在性...")

            # 如果向量客户端支持存在性检查
            if hasattr(self.vector_client, 'check_concepts_exist'):
                existing_check = self.vector_client.check_concepts_exist(self.collection_name, concept_names)
            else:
                logger.info("当前向量数据库不支持存在性检查，直接添加所有概念")
        else:
            logger.info("存在性检查已禁用，直接添加所有概念")

        # 创建集合
        vector_size = len(embeddings[0]) if len(embeddings) > 0 else 1024
        self.vector_client.create_collection(self.collection_name, vector_size)

        # 准备数据，只添加不存在的概念
        new_concepts = []
        new_embeddings = []
        new_ids = []
        new_metadatas = []

        for i, (concept, embedding) in enumerate(zip(concepts, embeddings)):
            concept_name = concept.get('name', '')

            if concept_name and existing_check.get(concept_name, False):
                logger.debug(f"概念 '{concept_name}' 已存在于Qdrant中，跳过")
                continue

            # 添加到新概念列表
            new_concepts.append(concept)
            new_embeddings.append(embedding)

            # 准备ID和元数据
            concept_id = concept.get('id', concept.get('name', f"concept_{i}"))
            new_ids.append(concept_id)

            metadata = {
                'name': concept.get('name', ''),
                'definition': concept.get('definition', ''),
                'category': concept.get('category', ''),
                'authenticity_score': concept.get('authenticity_score', 0.0),
                'related_concepts': concept.get('related_concepts', []),
                'sources': concept.get('sources', [])
            }
            new_metadatas.append(metadata)

        # 统计结果
        total_concepts = len(concepts)
        skipped_concepts = total_concepts - len(new_concepts)
        logger.info(f"概念存在性检查完成: 跳过 {skipped_concepts} 个已存在概念，添加 {len(new_concepts)} 个新概念")

        # 如果没有新概念需要添加，直接返回成功
        if len(new_concepts) == 0:
            logger.info("所有概念都已存在于Qdrant中，无需添加")
            return True

        # 使用过滤后的数据
        concepts, embeddings, ids, metadatas = new_concepts, new_embeddings, new_ids, new_metadatas

        # 批量插入
        return self.vector_client.insert(self.collection_name, ids, embeddings, metadatas)

    def search_similar_concepts(self, query_embedding: np.ndarray,
                               top_k: int = 10, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索相似的政治理论概念"""
        if not self.vector_client:
            logger.error("向量数据库客户端未初始化")
            return []

        # 构建过滤器
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        # 执行搜索
        results = self.vector_client.search(
            self.collection_name,
            query_embedding,
            top_k=top_k,
            where=where_filter
        )

        return results

    def find_concept_relationships(self, concept_name: str,
                                  relationship_types: List[str] = None) -> List[Dict[str, Any]]:
        """查找概念关系"""
        if not self.vector_client:
            logger.error("向量数据库客户端未初始化")
            return []

        # 这里可以基于向量相似度和元数据过滤查找相关概念
        # 具体实现取决于业务逻辑
        pass

    def get_concept_statistics(self) -> Dict[str, Any]:
        """获取概念索引统计信息"""
        if not self.vector_client:
            return {}

        stats = self.vector_client.get_collection_stats(self.collection_name)
        return stats