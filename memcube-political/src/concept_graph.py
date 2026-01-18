"""
政治理论概念图扩增系统
第三部分：基于种子概念进行迭代扩增，构建完整的概念图谱
"""

import json
import logging
import pickle
import time
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import yaml
import networkx as nx

try:
    from .api_client import get_client, APIResponse
    from .prompt_templates import PromptTemplates
    from .embedding_client import get_embedding_client
    from .graph_database_client import get_graph_client, close_graph_client
    from .vector_database_client import get_vector_client, close_vector_client, PoliticalTheoryVectorSearch
except ImportError:
    from api_client import get_client, APIResponse
    from prompt_templates import PromptTemplates
    from embedding_client import get_embedding_client
    from graph_database_client import get_graph_client, close_graph_client
    from vector_database_client import get_vector_client, close_vector_client, PoliticalTheoryVectorSearch

logger = logging.getLogger(__name__)

@dataclass
class ConceptExpansionResult:
    """概念扩增结果数据类"""
    concept_id: str
    center_concept: str
    status: str  # 'success', 'error', 'no_concepts'
    new_concepts: Optional[List[str]] = None
    returned_center: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: Optional[str] = None

class ConceptGraph:
    """政治理论概念图"""

    def __init__(self, seed_concepts: List[str], config_path: str = "config/config.yaml"):
        """初始化概念图"""
        self.config = self._load_config(config_path)
        self.client = get_client()
        self.templates = PromptTemplates()

        # 初始化数据库客户端
        self.graph_client = get_graph_client(config_path)
        self.vector_client = get_vector_client(config_path)
        self.embedding_client = get_embedding_client(config_path)

        # 初始化向量搜索工具
        self.vector_search = PoliticalTheoryVectorSearch(self.vector_client) if self.vector_client else None

        # 初始化NetworkX图（作为内存备用）
        self.graph = nx.Graph()
        self.concept_embeddings = {}

        # 添加种子概念
        self.seed_concepts = seed_concepts
        
        # 初始化属性
        self.concept_mapping = {}  # 概念映射表，用于去重
        self.concept_validity = {}  # concept -> 有效性分数
        self.concept_authenticity = {}  # concept -> 真实性验证结果
        
        # 初始化概念图
        self._initialize_graph(seed_concepts)

        # 统计信息
        self.iteration_count = 0
        self.convergence_history = []
        self.validity_stats = {'verified': 0, 'valid': 0, 'invalid': 0}
        self.authenticity_stats = {
            'verified': 0,      # 验证总数
            'authentic': 0,     # 真实概念数
            'synthetic': 0,     # 生成概念数
            'unknown': 0        # 未知概念数
        }

        # 初始化种子概念
        self._initialize_graph(seed_concepts)

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def _initialize_graph(self, seed_concepts: List[str]):
        """初始化概念图"""
        logger.info(f"初始化概念图，种子概念数量: {len(seed_concepts)}")

        # 清理种子概念
        cleaned_seeds = [concept.strip() for concept in seed_concepts if concept.strip()]

        # 计算embedding
        logger.info("正在为种子概念计算embedding...")
        seed_embeddings = self.embedding_client.encode(cleaned_seeds, show_progress=True)

        # 构建图 - 优先使用图数据库，否则使用NetworkX
        if self.graph_client:
            logger.info("使用图数据库存储概念")
            # 批量创建节点到图数据库
            batch_nodes = [(concept, {
                'name': concept,
                'definition': f'政治理论概念: {concept}',
                'category': 'seed_concept',
                'is_seed': True,
                'created_at': datetime.now().isoformat()
            }) for concept in cleaned_seeds]
            self.graph_client.batch_create_nodes(batch_nodes)

        # 构建NetworkX图（内存备用）
        for concept, embedding in zip(cleaned_seeds, seed_embeddings):
            self.graph.add_node(concept, is_seed=True, created_at=datetime.now().isoformat())
            self.concept_embeddings[concept] = embedding
            self.concept_mapping[concept] = concept

        # 将概念添加到向量数据库
        if self.vector_search and seed_embeddings is not None:
            concepts_data = [{
                'id': concept,
                'name': concept,
                'definition': f'政治理论概念: {concept}',
                'category': 'seed_concept',
                'is_seed': True,
                'created_at': datetime.now().isoformat()
            } for concept in cleaned_seeds]

            self.vector_search.index_concepts(concepts_data, seed_embeddings)

        logger.info(f"概念图初始化完成，种子概念数: {len(cleaned_seeds)}")

    def _is_similar_to_existing(self, new_concept: str, new_embedding: np.ndarray) -> Optional[str]:
        """检查新概念是否与已有概念相似"""
        # 优先使用向量数据库搜索
        if self.vector_search:
            try:
                similar_concepts = self.vector_search.search_similar_concepts(
                    new_embedding, top_k=1
                )
                if similar_concepts:
                    similarity_score = similar_concepts[0].get('score', 0)
                    similarity_threshold = self.config['concept_expansion']['similarity_threshold']
                    if similarity_score >= similarity_threshold:
                        return similar_concepts[0].get('metadata', {}).get('name')
            except Exception as e:
                logger.warning(f"向量数据库搜索失败，使用内存模式: {e}")

        # 回退到内存模式
        if not self.concept_embeddings:
            return None

        # 计算与所有已有概念的相似度
        existing_concepts = list(self.concept_embeddings.keys())
        existing_embeddings = np.array([self.concept_embeddings[concept] for concept in existing_concepts])

        # 计算余弦相似度
        similarities = np.dot(existing_embeddings, new_embedding) / (
            np.linalg.norm(existing_embeddings, axis=1) * np.linalg.norm(new_embedding) + 1e-8
        )

        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]

        similarity_threshold = self.config['concept_expansion']['similarity_threshold']

        if max_similarity >= similarity_threshold:
            return existing_concepts[max_similarity_idx]

        return None

    def expand_single_concept(self, center_concept: str, neighbors: List[str], concept_id: str) -> ConceptExpansionResult:
        """扩增单个概念"""
        try:
            logger.debug(f"开始扩增概念: {center_concept}")

            neighbors_text = ", ".join(neighbors) if neighbors else "无"

            messages = [
                {
                    "role": "system",
                    "content": self.templates.get_concept_expansion_system_prompt()
                },
                {
                    "role": "user",
                    "content": self.templates.get_concept_expansion_user_prompt(center_concept, neighbors)
                }
            ]

            response = self.client.json_completion(
                messages=messages,
                model=self.config['api']['model_expander'],
                temperature=0.7,
                max_tokens=2000
            )

            if response.success:
                try:
                    # 检查response.content是否为空或None
                    if not response.content or response.content is None:
                        logger.warning(f"API响应内容为空，概念: {center_concept}")
                        return ConceptExpansionResult(
                            concept_id=concept_id,
                            center_concept=center_concept,
                            status="no_content",
                            new_concepts=[],
                            returned_center=center_concept,
                            timestamp=self._get_timestamp()
                        )

                    # 确保response.content是字符串类型
                    if isinstance(response.content, str):
                        parsed_json = json.loads(response.content)
                    else:
                        parsed_json = response.content

                    returned_center = parsed_json.get("center_concept", center_concept)
                    new_concepts = parsed_json.get("new_concepts", [])

                    # 检查是否为"无新增概念"的情况
                    if len(new_concepts) == 1 and new_concepts[0].strip() == "NO NEW CONCEPTS":
                        logger.debug(f"概念 {center_concept} 无新概念")
                        return ConceptExpansionResult(
                            concept_id=concept_id,
                            center_concept=center_concept,
                            status="no_concepts",
                            new_concepts=[],
                            timestamp=self._get_timestamp()
                        )

                    # 正常处理新概念
                    new_concepts = [concept.strip() for concept in new_concepts if concept.strip()]

                    # 验证新概念的有效性
                    verified_concepts = self._validate_new_concepts(new_concepts, center_concept)

                    logger.debug(f"概念 {center_concept} 扩增成功，新概念数: {len(new_concepts)}，验证后: {len(verified_concepts)}")

                    return ConceptExpansionResult(
                        concept_id=concept_id,
                        center_concept=center_concept,
                        status="success",
                        new_concepts=verified_concepts,
                        returned_center=returned_center,
                        timestamp=self._get_timestamp()
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败 {center_concept}: {e}, response.content: {response.content}")
                    return ConceptExpansionResult(
                        concept_id=concept_id,
                        center_concept=center_concept,
                        status="json_error",
                        error_message=f"JSON解析失败: {str(e)}",
                        timestamp=self._get_timestamp()
                    )
                except Exception as e:
                    logger.error(f"处理响应内容时出错 {center_concept}: {e}")
                    return ConceptExpansionResult(
                        concept_id=concept_id,
                        center_concept=center_concept,
                        status="processing_error",
                        error_message=f"处理失败: {str(e)}",
                        timestamp=self._get_timestamp()
                    )
            else:
                logger.error(f"概念扩增失败 {center_concept}: {response.error}")
                return ConceptExpansionResult(
                    concept_id=concept_id,
                    center_concept=center_concept,
                    status="error",
                    error_message=response.error,
                    timestamp=self._get_timestamp()
                )

        except Exception as e:
            logger.error(f"扩增概念 {center_concept} 时出错: {e}")
            return ConceptExpansionResult(
                concept_id=concept_id,
                center_concept=center_concept,
                status="error",
                error_message=str(e),
                timestamp=self._get_timestamp()
            )

    def expand_concepts_batch(self, max_workers: int = 10) -> Dict:
        """批量扩增概念"""
        concepts_to_expand = list(self.graph.nodes())
        total_concepts = len(concepts_to_expand)

        logger.info(f"开始批量概念扩增 {total_concepts} 个概念")

        batch_results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_concept = {}

            for idx, concept in enumerate(concepts_to_expand):
                concept_id = f"concept_{idx:06d}"
                neighbors = list(self.graph.neighbors(concept)) if self.graph.has_node(concept) else []
                future = executor.submit(self.expand_single_concept, concept, neighbors, concept_id)
                future_to_concept[future] = (concept_id, concept)

            completed = 0
            success_count = 0

            for future in as_completed(future_to_concept):
                concept_id, concept = future_to_concept[future]
                completed += 1

                try:
                    result = future.result()
                    batch_results[concept_id] = result

                    if result.status == "success":
                        success_count += 1
                        status_symbol = "✓"
                    elif result.status == "no_concepts":
                        status_symbol = "○"
                    else:
                        status_symbol = "✗"

                    if completed % 100 == 0 or completed == len(concepts_to_expand):
                        logger.info(f"  已完成: {completed}/{len(concepts_to_expand)} (成功: {success_count}) {status_symbol}")

                except Exception as e:
                    logger.error(f"  异常: {concept_id} - {str(e)}")
                    batch_results[concept_id] = ConceptExpansionResult(
                        concept_id=concept_id,
                        center_concept=concept,
                        status="error",
                        error_message=str(e)
                    )

        # 保存批处理结果
        self._save_batch_results(batch_results)

        # 统计生成的概念数量
        total_new_concepts = sum(len(r.new_concepts) for r in batch_results.values()
                               if r.status == "success" and r.new_concepts)

        # 统计跳过的概念数量（无新增概念）
        skipped_concepts = sum(1 for r in batch_results.values()
                              if r.status == "no_concepts")

        logger.info(f"批量扩增完成:")
        logger.info(f"  处理概念数: {total_concepts}")
        logger.info(f"  成功扩增: {success_count}")
        logger.info(f"  跳过概念: {skipped_concepts}")
        logger.info(f"  新概念总数: {total_new_concepts}")

        return batch_results

    def update_graph(self, batch_results: Dict):
        """更新概念图"""
        logger.info("开始更新概念图...")

        # 收集所有新概念
        all_new_concepts = []
        concept_to_centers = {}  # 记录每个新概念对应的中心概念

        for result in batch_results.values():
            if result.status == "success" and result.new_concepts:
                center_concept = result.center_concept
                for new_concept in result.new_concepts:
                    if new_concept.strip():
                        cleaned_concept = new_concept.strip()
                        all_new_concepts.append(cleaned_concept)
                        if cleaned_concept not in concept_to_centers:
                            concept_to_centers[cleaned_concept] = []
                        concept_to_centers[cleaned_concept].append(center_concept)

        if not all_new_concepts:
            logger.info("没有新概念需要添加")
            return 0, 0, 0

        # 使用embedding去重
        nodes_added, edges_added, embedding_duplicates = self._deduplicate_and_add_concepts(
            all_new_concepts, concept_to_centers
        )

        logger.info(f"图更新完成: 新增节点 {nodes_added}, 新增边 {edges_added}, 去重概念 {embedding_duplicates}")

        return nodes_added, edges_added, embedding_duplicates

    def _deduplicate_and_add_concepts(self, all_new_concepts: List[str], concept_to_centers: Dict) -> Tuple[int, int, int]:
        """去重并添加概念到图中"""
        # 使用现有的embedding_client实例，保持缓存一致性
        embedding_client = self.embedding_client

        num_all_new_concepts = len(all_new_concepts)
        logger.info(f"收到 {num_all_new_concepts} 个新概念（未去重）")

        # 分离已知概念和需要计算embedding的概念
        concepts_need_embedding = []
        concept_targets = {}  # concept -> target_concept 映射

        for concept in all_new_concepts:
            target = self.concept_mapping.get(concept)
            if target:
                # 已知概念，直接使用映射
                concept_targets[concept] = target
            else:
                # 需要计算embedding的新概念
                concepts_need_embedding.append(concept)

        # 只对未知概念计算embedding
        if concepts_need_embedding:
            unique_concepts = list(set(concepts_need_embedding))
            logger.info(f"正在对 {len(unique_concepts)} 个新概念进行embedding去重...")

            # 检查缓存中已有的概念
            cached_concepts = []
            uncached_concepts = []
            for concept in unique_concepts:
                if concept in self.concept_embeddings:
                    cached_concepts.append(concept)
                else:
                    uncached_concepts.append(concept)

            if cached_concepts:
                logger.info(f"发现 {len(cached_concepts)} 个概念已缓存embedding，跳过计算")
                logger.debug(f"缓存的概念: {cached_concepts[:5]}...")  # 只显示前5个

            if uncached_concepts:
                logger.info(f"需要对 {len(uncached_concepts)} 个概念计算新的embedding")
                new_embeddings = embedding_client.encode(uncached_concepts, show_progress=True)
            else:
                logger.info("所有概念都已在缓存中，无需计算新的embedding")
                new_embeddings = []

            # 逐个处理概念
            total_concepts = len(unique_concepts)
            processed_count = 0

            # 首先处理已缓存的概念
            for concept in cached_concepts:
                if concept in self.concept_embeddings:
                    processed_count += 1
                    if processed_count % 500 == 0 or processed_count == total_concepts:
                        logger.info(f"  处理进度: {processed_count}/{total_concepts} ({processed_count/total_concepts*100:.1f}%)")

            # 然后处理需要新计算embedding的概念
            if uncached_concepts:
                for idx, (new_concept, new_embedding) in enumerate(zip(uncached_concepts, new_embeddings), 1):
                    actual_idx = processed_count + idx
                    if actual_idx % 500 == 0 or actual_idx == total_concepts:
                        logger.info(f"  处理进度: {actual_idx}/{total_concepts} ({actual_idx/total_concepts*100:.1f}%)")

                # 检查是否与已有概念相似
                similar_concept = self._is_similar_to_existing(new_concept, new_embedding)

                if similar_concept:
                    # 发现相似概念，建立映射
                    self.concept_mapping[new_concept] = similar_concept
                    concept_targets[new_concept] = similar_concept
                else:
                    # 全新概念，添加到图中并建立自映射
                    self.graph.add_node(new_concept, created_at=datetime.now().isoformat())
                    self.concept_embeddings[new_concept] = new_embedding
                    self.concept_mapping[new_concept] = new_concept
                    concept_targets[new_concept] = new_concept

        # 添加边（连接到所有相关的中心概念）
        edges_added = 0
        for concept in all_new_concepts:
            target_concept = concept_targets[concept]

            for center_concept in concept_to_centers[concept]:
                # 确保概念存在于图中
                if self.graph.has_node(center_concept) and self.graph.has_node(target_concept):
                    # 添加边（NetworkX自动处理双向连接）
                    if not self.graph.has_edge(center_concept, target_concept):
                        self.graph.add_edge(center_concept, target_concept,
                                          created_at=datetime.now().isoformat(),
                                          weight=1.0)
                        edges_added += 1

        existing_nodes = set(self.graph.nodes())
        nodes_added = len([c for c in concept_targets.values() if c not in existing_nodes])
        embedding_duplicates = num_all_new_concepts - len(set(concept_targets.values()))

        return nodes_added, edges_added, embedding_duplicates

    def calculate_metrics(self) -> Dict:
        """计算收敛指标"""
        total_concepts = len(self.graph)
        total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2  # 无向图

        return {
            "nodes": total_concepts,
            "edges": total_edges,
            "iteration": self.iteration_count,
            "timestamp": self._get_timestamp()
        }

    def check_convergence(self, previous_metrics: Dict) -> Dict:
        """检查收敛条件"""
        current_metrics = self.calculate_metrics()

        # 计算增长率
        if previous_metrics:
            node_growth_rate = (current_metrics["nodes"] - previous_metrics["nodes"]) / previous_metrics["nodes"]
            edge_growth_rate = (current_metrics["edges"] - previous_metrics["edges"]) / previous_metrics["edges"] if previous_metrics["edges"] > 0 else 0
        else:
            node_growth_rate = 1.0
            edge_growth_rate = 1.0

        convergence_info = {
            "current_metrics": current_metrics,
            "previous_metrics": previous_metrics,
            "node_growth_rate": node_growth_rate,
            "edge_growth_rate": edge_growth_rate,
            "is_converged": False,
            "convergence_reason": None
        }

        # 检查收敛条件
        thresholds = self.config['concept_expansion']

        # 新概念增长率检查
        if node_growth_rate < thresholds['new_concept_rate_threshold']:
            convergence_info["is_converged"] = True
            convergence_info["convergence_reason"] = "new_concept_rate_low"

        # 新边增长率检查
        if edge_growth_rate < thresholds['new_edge_rate_threshold']:
            convergence_info["is_converged"] = True
            convergence_info["convergence_reason"] = "new_edge_rate_low"

        # 最大概念数检查
        if current_metrics["nodes"] >= self.config['convergence']['max_concepts']:
            convergence_info["is_converged"] = True
            convergence_info["convergence_reason"] = "max_concepts_reached"

        # 最大迭代次数检查
        if self.iteration_count >= self.config['concept_expansion']['max_iterations']:
            convergence_info["is_converged"] = True
            convergence_info["convergence_reason"] = "max_iterations_reached"

        return convergence_info

    def run_expansion_iteration(self) -> Dict:
        """运行单轮概念扩增迭代"""
        logger.info(f"开始第 {self.iteration_count + 1} 轮概念扩增迭代")

        # 批量扩增概念
        batch_results = self.expand_concepts_batch(
            max_workers=self.config['concept_expansion']['max_workers']
        )

        # 更新概念图
        nodes_added, edges_added, duplicates = self.update_graph(batch_results)

        # 更新迭代计数
        self.iteration_count += 1

        # 计算指标
        metrics = self.calculate_metrics()

        iteration_result = {
            "iteration": self.iteration_count,
            "metrics": metrics,
            "batch_results": batch_results,
            "nodes_added": nodes_added,
            "edges_added": edges_added,
            "duplicates_removed": duplicates,
            "timestamp": self._get_timestamp()
        }

        # 保存迭代结果
        self._save_iteration_result(iteration_result)

        logger.info(f"第 {self.iteration_count} 轮迭代完成: 节点数 {metrics['nodes']}, 边数 {metrics['edges']}")

        return iteration_result

    def run_full_expansion(self) -> List[Dict]:
        """运行完整的概念扩增流程"""
        logger.info("开始完整的概念图扩增流程")

        # 检查embedding缓存状态
        cached_count = len(self.concept_embeddings)
        if cached_count > 0:
            logger.info(f"发现已有 {cached_count} 个概念的embedding缓存，将重复使用")
            logger.info(f"当前内存中概念embedding缓存: {len(self.concept_embeddings)} 个概念")

        iteration_results = []
        previous_metrics = None

        for iteration in range(self.config['concept_expansion']['max_iterations']):
            logger.info(f"\n=== 迭代 {iteration + 1}/{self.config['concept_expansion']['max_iterations']} ===")

            # 运行单轮迭代
            result = self.run_expansion_iteration()
            iteration_results.append(result)

            # 自动保存到Neo4j（如果配置启用）
            concept_config = self.config['concept_expansion']
            if concept_config.get('save_to_neo4j_after_each_iteration', False):
                logger.info("自动保存当前概念图到Neo4j...")
                try:
                    self.save_to_neo4j()
                    logger.info("✅ 概念图已保存到Neo4j")
                except Exception as e:
                    logger.error(f"❌ 保存到Neo4j失败: {e}")

            # 检查是否在第一轮后停止
            if concept_config.get('stop_after_first_iteration', False) and iteration == 0:
                logger.info("配置要求第一轮后停止扩增")
                break

            # 检查收敛
            convergence_info = self.check_convergence(previous_metrics)
            self.convergence_history.append(convergence_info)

            # 输出收敛信息
            if convergence_info["is_converged"]:
                logger.info(f"收敛条件满足: {convergence_info['convergence_reason']}")
                logger.info(f"节点增长率: {convergence_info['node_growth_rate']:.4f}")
                logger.info(f"边增长率: {convergence_info['edge_growth_rate']:.4f}")
                break

            previous_metrics = convergence_info["current_metrics"]

            # 检查早停条件
            if len(self.convergence_history) >= self.config['convergence']['early_stop_patience']:
                recent_growth_rates = [h["node_growth_rate"] for h in self.convergence_history[-self.config['convergence']['early_stop_patience']:]]
                if all(rate < 0.01 for rate in recent_growth_rates):  # 连续几轮增长率都很低
                    logger.info("触发早停条件：连续几轮增长率都很低")
                    break

        # 保存最终结果
        self._save_final_results(iteration_results)

        logger.info(f"概念扩增流程完成，总迭代次数: {len(iteration_results)}")
        logger.info(f"最终概念图: {len(self.graph)} 个节点, {sum(len(neighbors) for neighbors in self.graph.values()) // 2} 条边")

        return iteration_results

    def save_to_neo4j(self):
        """保存概念图到Neo4j数据库"""
        logger.info("开始保存概念图到Neo4j...")

        try:
            # 确保图数据库客户端存在
            if not hasattr(self, 'graph_client') or self.graph_client is None:
                from .graph_database_client import get_graph_client
                self.graph_client = get_graph_client('config/config.yaml')

            # 测试连接
            if not self.graph_client.test_connection():
                raise Exception("Neo4j连接测试失败")

            logger.info("Neo4j连接成功，开始导入数据...")

            # 清空现有数据（可选，根据需求）
            # self.graph_client.clear_database()

            # 批量导入节点
            nodes_added = 0
            edges_added = 0

            # 准备节点数据
            node_operations = []
            for concept_name in self.graph.nodes():
                # 获取概念的embedding和属性
                embedding = self.concept_embeddings.get(concept_name, None)
                validity_score = self.concept_validity.get(concept_name, 0.5)

                node_data = {
                    'id': f"concept_{hash(concept_name)}",
                    'labels': ['Concept', 'PoliticalTheory'],
                    'properties': {
                        'name': concept_name,
                        'validity_score': validity_score,
                        'embedding': embedding.tolist() if embedding is not None else None,
                        'created_at': datetime.now().isoformat(),
                        'iteration': self.iteration_count
                    }
                }
                node_operations.append(node_data)

            # 批量添加节点
            if node_operations:
                logger.info(f"批量添加 {len(node_operations)} 个节点...")
                for i in range(0, len(node_operations), 50):  # 每批50个节点
                    batch = node_operations[i:i+50]
                    for node_op in batch:
                        try:
                            if self.graph_client.add_node(
                                node_op['id'],
                                node_op['labels'],
                                node_op['properties']
                            ):
                                nodes_added += 1
                        except Exception as e:
                            logger.warning(f"添加节点失败 {node_op['properties']['name']}: {e}")

            # 准备边数据
            edge_operations = []
            processed_edges = set()  # 避免重复边

            for source_concept, neighbors in self.graph.items():
                for target_concept in neighbors:
                    # 创建边的唯一标识
                    edge_id = tuple(sorted([source_concept, target_concept]))
                    if edge_id in processed_edges:
                        continue
                    processed_edges.add(edge_id)

                    # 计算边的权重（基于概念有效性）
                    source_validity = self.concept_validity.get(source_concept, 0.5)
                    target_validity = self.concept_validity.get(target_concept, 0.5)
                    edge_weight = (source_validity + target_validity) / 2

                    source_id = f"concept_{hash(source_concept)}"
                    target_id = f"concept_{hash(target_concept)}"

                    edge_data = {
                        'source_id': source_id,
                        'target_id': target_id,
                        'relationship_type': 'RELATED_TO',
                        'properties': {
                            'weight': edge_weight,
                            'relationship_strength': edge_weight,
                            'created_at': datetime.now().isoformat(),
                            'iteration': self.iteration_count
                        }
                    }
                    edge_operations.append(edge_data)

            # 批量添加边
            if edge_operations:
                logger.info(f"批量添加 {len(edge_operations)} 条边...")
                for i in range(0, len(edge_operations), 50):  # 每批50条边
                    batch = edge_operations[i:i+50]
                    for edge_op in batch:
                        try:
                            if self.graph_client.add_edge(
                                edge_op['source_id'],
                                edge_op['target_id'],
                                edge_op['relationship_type'],
                                edge_op['properties']
                            ):
                                edges_added += 1
                        except Exception as e:
                            logger.warning(f"添加边失败 {edge_op['source_id']}-{edge_op['target_id']}: {e}")

            logger.info(f"✅ Neo4j保存完成: {nodes_added} 个节点, {edges_added} 条边")

        except Exception as e:
            logger.error(f"❌ 保存到Neo4j失败: {e}")
            raise

    def _save_batch_results(self, batch_results: Dict):
        """保存批处理结果"""
        try:
            output_dir = Path(self.config['paths']['concept_graph_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"expansion_batch_{self.iteration_count}_{timestamp}.pkl"
            filepath = output_dir / filename

            with open(filepath, 'wb') as f:
                pickle.dump(batch_results, f)

            logger.debug(f"批处理结果已保存: {filepath}")

        except Exception as e:
            logger.error(f"保存批处理结果失败: {e}")

    def _save_iteration_result(self, result: Dict):
        """保存迭代结果"""
        try:
            output_dir = Path(self.config['paths']['concept_graph_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"iteration_{result['iteration']:04d}.json"
            filepath = output_dir / filename

            # 转换数据类为字典
            serializable_result = {}
            for key, value in result.items():
                if key == 'batch_results':
                    serializable_result[key] = [asdict(r) if hasattr(r, '__dict__') else r for r in value]
                else:
                    serializable_result[key] = value

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)

            logger.debug(f"迭代结果已保存: {filepath}")

        except Exception as e:
            logger.error(f"保存迭代结果失败: {e}")

    def _save_final_results(self, iteration_results: List[Dict]):
        """保存最终结果"""
        try:
            output_dir = Path(self.config['paths']['concept_graph_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存概念图
            graph_file = output_dir / "final_concept_graph.json"
            # Convert NetworkX graph to dictionary format
            graph_dict = {}
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                neighbors = list(self.graph.neighbors(node))
                graph_dict[node] = {
                    "attributes": dict(node_data),
                    "neighbors": neighbors
                }

            graph_data = {
                "graph": graph_dict,
                "concept_embeddings": {k: v.tolist() for k, v in self.concept_embeddings.items()},
                "concept_mapping": self.concept_mapping,
                "concept_validity": self.concept_validity,
                "concept_authenticity": {
                    concept: {
                        "is_authentic": result.is_authentic,
                        "confidence": result.confidence,
                        "verification_method": result.verification_method,
                        "details": result.details,
                        "evidence_sources": result.evidence_sources
                    }
                    for concept, result in self.concept_authenticity.items()
                },
                "validity_stats": self.get_validity_statistics(),
                "authenticity_stats": self.get_authenticity_statistics(),
                "metadata": {
                    "total_iterations": len(iteration_results),
                    "final_nodes": len(self.graph.nodes()),
                    "final_edges": len(self.graph.edges()),
                    "timestamp": self._get_timestamp(),
                    "validation_enabled": True
                }
            }

            with open(graph_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)

            # 保存收敛历史
            convergence_file = output_dir / "convergence_history.json"
            with open(convergence_file, 'w', encoding='utf-8') as f:
                json.dump(self.convergence_history, f, ensure_ascii=False, indent=2)

            # 保存迭代摘要
            summary_file = output_dir / "expansion_summary.json"
            summary = {
                "config": self.config,
                "total_iterations": len(iteration_results),
                "final_metrics": self.calculate_metrics(),
                "convergence_achieved": len(self.convergence_history) > 0 and self.convergence_history[-1]["is_converged"],
                "convergence_reason": self.convergence_history[-1]["convergence_reason"] if self.convergence_history else None,
                "timestamp": self._get_timestamp()
            }

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            logger.info(f"最终结果已保存到: {output_dir}")

        except Exception as e:
            logger.error(f"保存最终结果失败: {e}")

    def _validate_new_concepts(self, new_concepts: List[str], center_concept: str) -> List[str]:
        """验证新概念的有效性"""
        if not new_concepts:
            return []

        validated_concepts = []
        context_concepts = list(self.graph.nodes())  # 获取已有概念作为上下文

        # 确保中心概念有embedding缓存
        if center_concept not in self.concept_embeddings:
            logger.warning(f"中心概念 '{center_concept}' 不在embedding缓存中，需要计算")
            try:
                center_embedding = self.embedding_client.encode([center_concept])[0]
                self.concept_embeddings[center_concept] = center_embedding
                logger.info(f"已缓存中心概念 '{center_concept}' 的embedding")
            except Exception as e:
                logger.error(f"计算中心概念 '{center_concept}' 的embedding失败: {e}")
                return []  # 如果中心概念都无法处理，返回空列表

        logger.info(f"开始验证 {len(new_concepts)} 个新概念的有效性（中心概念: {center_concept}）")

        # 批量检查新概念的embedding缓存情况
        uncached_new_concepts = []
        cached_new_concepts = []

        for concept in new_concepts:
            if concept in self.concept_embeddings:
                cached_new_concepts.append(concept)
            else:
                uncached_new_concepts.append(concept)

        logger.info(f"新概念中: {len(cached_new_concepts)} 个已缓存, {len(uncached_new_concepts)} 个需要计算embedding")

        # 检查Qdrant向量数据库中已存在的概念
        if uncached_new_concepts and self.vector_search:
            logger.info(f"检查Qdrant向量数据库中 {len(uncached_new_concepts)} 个概念是否已存在...")
            try:
                existing_check = self.vector_search.vector_client.check_concepts_exist(
                    self.vector_search.collection_name,
                    uncached_new_concepts
                )

                # 从数据库中加载已存在的概念embeddings
                db_existing_concepts = []
                db_missing_concepts = []

                for concept in uncached_new_concepts:
                    if existing_check.get(concept, False):
                        db_existing_concepts.append(concept)
                    else:
                        db_missing_concepts.append(concept)

                logger.info(f"Qdrant检查结果: {len(db_existing_concepts)} 个概念已存在数据库，{len(db_missing_concepts)} 个概念需要新增")

                # 为数据库中已存在的概念创建搜索任务以获取embedding
                if db_existing_concepts:
                    logger.info(f"从Qdrant加载 {len(db_existing_concepts)} 个已存在概念的embedding...")
                    try:
                        # 使用搜索来获取已存在的概念，这样可以得到它们的embedding信息
                        existing_results = self.vector_search.search_concepts(db_existing_concepts, top_k=1)

                        for concept in db_existing_concepts:
                            if concept in existing_results and existing_results[concept]:
                                # 使用向量数据库中的信息，将概念标记为已处理
                                cached_new_concepts.append(concept)
                                # 创建一个虚拟的embedding用于本次计算（因为主要是为了相似度计算）
                                # 实际的embedding会在需要时从数据库获取
                                self.concept_embeddings[concept] = np.ones(1024)  # 使用虚拟embedding避免重复计算
                                logger.debug(f"✅ 概念 '{concept}' 从Qdrant数据库确认存在，跳过重新计算")
                            else:
                                db_missing_concepts.append(concept)

                    except Exception as e:
                        logger.error(f"从Qdrant加载概念embedding失败: {e}")
                        # 如果加载失败，将这些概念视为需要重新计算
                        db_missing_concepts.extend(db_existing_concepts)

                # 更新需要计算embedding的概念列表
                uncached_new_concepts = db_missing_concepts

            except Exception as e:
                logger.error(f"检查Qdrant向量数据库失败: {e}")
                # 如果检查失败，继续正常流程计算所有概念的embedding

        # 批量计算缺失的embeddings
        if uncached_new_concepts:
            logger.info(f"批量计算 {len(uncached_new_concepts)} 个新概念的embedding...")
            try:
                # 分批处理，避免一次请求太多
                batch_size = 20  # 减少batch size避免overload
                total_batches = (len(uncached_new_concepts) + batch_size - 1) // batch_size

                for i in range(0, len(uncached_new_concepts), batch_size):
                    batch = uncached_new_concepts[i:i + batch_size]
                    logger.info(f"  处理第 {i//batch_size + 1}/{total_batches} 批 ({len(batch)} 个概念)")

                    batch_embeddings = self.embedding_client.encode(batch)

                    for concept, embedding in zip(batch, batch_embeddings):
                        self.concept_embeddings[concept] = embedding

                    # 添加小延迟避免overload
                    time.sleep(0.5)  # 500ms延迟

            except Exception as e:
                logger.error(f"批量计算新概念embedding失败: {e}")
                # 如果批量失败，只处理缓存的概念
                uncached_new_concepts = []

        for concept in new_concepts:
            try:
                # 计算概念有效性分数
                validity_score = self._calculate_concept_validity(concept, center_concept)
                self.concept_validity[concept] = validity_score

                # 使用基本有效性阈值判断
                validity_threshold = self.config["concept_expansion"].get("validity_threshold", 0.5)
                
                if validity_score >= validity_threshold:
                    validated_concepts.append(concept)
                    self.validity_stats["valid"] += 1
                    logger.debug(f"概念通过验证: {concept} (有效性:{validity_score:.2f})")
                else:
                    self.validity_stats["invalid"] += 1
                    logger.debug(f"概念有效性不足: {concept} (有效性:{validity_score:.2f})")

            except Exception as e:
                logger.error(f"验证概念 '{concept}' 时出错: {e}")
                continue

        return validated_concepts

    def _calculate_concept_validity(self, concept: str, center_concept: str) -> float:
        """计算概念有效性分数"""
        total_score = 0.0
        weights = {
            'semantic_similarity': 0.25,
            'concept_quality': 0.25,
            'political_theory_relevance': 0.30,
            'linguistic_quality': 0.20
        }

        # 1. 语义相似度检查
        semantic_score = self._check_semantic_similarity(concept, center_concept)
        total_score += semantic_score * weights['semantic_similarity']

        # 2. 概念质量检查
        quality_score = self._check_concept_quality(concept)
        total_score += quality_score * weights['concept_quality']

        # 3. 政治理论相关性检查
        relevance_score = self._check_political_theory_relevance(concept)
        total_score += relevance_score * weights['political_theory_relevance']

        # 4. 语言质量检查
        linguistic_score = self._check_linguistic_quality(concept)
        total_score += linguistic_score * weights['linguistic_quality']

        return min(total_score, 1.0)

    def _check_semantic_similarity(self, concept: str, center_concept: str) -> float:
        """检查概念与中心概念的语义相似度"""
        try:
            # 首先检查是否已经有缓存的embeddings
            cache_status = {
                'total_cached': len(self.concept_embeddings),
                'concept_cached': concept in self.concept_embeddings,
                'center_concept_cached': center_concept in self.concept_embeddings
            }

            if concept in self.concept_embeddings and center_concept in self.concept_embeddings:
                # 使用缓存的embeddings，避免重复计算
                concept_emb = self.concept_embeddings[concept]
                center_emb = self.concept_embeddings[center_concept]
                logger.debug(f"✅ 使用缓存embedding计算相似度: {concept} vs {center_concept}")
            else:
                # 详细分析缓存缺失的原因
                missing_reasons = []
                if concept not in self.concept_embeddings:
                    missing_reasons.append(f"概念 '{concept}' 不在缓存中")
                if center_concept not in self.concept_embeddings:
                    missing_reasons.append(f"中心概念 '{center_concept}' 不在缓存中")

                # 限制缓存缺失警告的频率，避免日志泛滥
                import random
                if random.random() < 0.1:  # 只有10%的概率打印详细警告
                    logger.warning(f"⚠️  embedding缓存缺失 (缓存中有 {cache_status['total_cached']} 个概念)")
                    logger.warning(f"   缺失原因: {'; '.join(missing_reasons)}")
                    if cache_status['total_cached'] > 0:
                        sample_concepts = list(self.concept_embeddings.keys())[:5]
                        logger.warning(f"   缓存样本概念: {sample_concepts}...")

                # 使用现有的embedding_client实例
                embeddings = self.embedding_client.encode([concept, center_concept])
                concept_emb, center_emb = embeddings[0], embeddings[1]

                # 将新计算的embedding缓存起来，避免重复计算
                if concept not in self.concept_embeddings:
                    self.concept_embeddings[concept] = concept_emb
                if center_concept not in self.concept_embeddings:
                    self.concept_embeddings[center_concept] = center_emb

            # 计算余弦相似度
            similarity = np.dot(concept_emb, center_emb) / (np.linalg.norm(concept_emb) * np.linalg.norm(center_emb))

            # 将相似度映射到0-1范围
            return max(0.0, min(1.0, (similarity + 1) / 2))

        except Exception as e:
            logger.warning(f"计算语义相似度失败: {e}")
            return 0.5  # 默认中等分数

    def _check_concept_quality(self, concept: str) -> float:
        """检查概念质量"""
        score = 1.0

        # 长度检查
        if len(concept) < 2:
            score -= 0.5
        elif len(concept) > 20:
            score -= 0.2

        # 特殊字符检查
        if any(char in concept for char in ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=']):
            score -= 0.3

        # 数字检查（政治理论概念通常不以纯数字为主）
        if concept.isdigit():
            score -= 0.8
        elif sum(c.isdigit() for c in concept) / len(concept) > 0.3:
            score -= 0.3

        # 重复字符检查
        if len(set(concept)) / len(concept) < 0.5:
            score -= 0.3

        return max(0.0, score)

    def _check_political_theory_relevance(self, concept: str) -> float:
        """检查政治理论相关性"""
        # 政治理论核心关键词
        political_keywords = [
            '政治', '理论', '思想', '主义', '哲学', '经济', '社会', '文化', '历史',
            '马克思', '毛泽东', '邓小平', '社会主义', '共产主义', '资本主义',
            '民主', '自由', '平等', '公正', '权力', '国家', '政府', '制度',
            '革命', '改革', '发展', '现代化', '阶级', '斗争', '生产', '关系',
            '辩证', '唯物', '唯心', '历史', '认识', '实践', '真理', '价值'
        ]

        concept_lower = concept.lower()
        score = 0.0

        # 直接包含关键词
        for keyword in political_keywords:
            if keyword in concept_lower:
                score += 0.5
                break

        # 相关概念匹配
        related_patterns = [
            '.*论$', '.*学$', '.*派$', '.*观$', '.*主义$', '.*思想$',
            '.*制度$', '.*文化$', '.*经济$', '.*社会$', '.*政治$'
        ]

        import re
        for pattern in related_patterns:
            if re.search(pattern, concept):
                score += 0.3
                break

        # 抽象概念特征
        abstract_indicators = ['性', '度', '化', '力', '关系', '结构', '体系', '模式']
        for indicator in abstract_indicators:
            if concept.endswith(indicator):
                score += 0.2
                break

        return min(1.0, score)

    def _check_linguistic_quality(self, concept: str) -> float:
        """检查语言质量"""
        score = 1.0

        # 检查是否为有效中文词汇
        if not self._is_valid_chinese_term(concept):
            score -= 0.4

        # 检查是否包含常见无意义词汇
        meaningless_words = ['什么', '怎么', '为什么', '因为', '所以', '但是', '然后', '或者']
        if any(word in concept for word in meaningless_words):
            score -= 0.5

        # 检查是否过于口语化
        colloquial_patterns = ['的', '了', '啊', '呢', '吧', '嘛']
        colloquial_count = sum(concept.count(pattern) for pattern in colloquial_patterns)
        if colloquial_count > 1:
            score -= 0.3 * colloquial_count

        return max(0.0, score)

    def _is_valid_chinese_term(self, concept: str) -> bool:
        """检查是否为有效的中文术语"""
        # 简单的中文词汇有效性检查
        if len(concept) < 2:
            return False

        # 检查是否主要由汉字组成
        chinese_char_count = sum('\u4e00' <= char <= '\u9fff' for char in concept)
        if chinese_char_count / len(concept) < 0.5:
            return False

        return True

    def get_validity_statistics(self) -> Dict:
        """获取有效性统计信息"""
        stats = self.validity_stats.copy()

        if stats['verified'] > 0:
            stats['valid_ratio'] = stats['valid'] / stats['verified']
            stats['invalid_ratio'] = stats['invalid'] / stats['verified']
        else:
            stats['valid_ratio'] = 0.0
            stats['invalid_ratio'] = 0.0

        # 计算概念有效性分布
        if self.concept_validity:
            scores = list(self.concept_validity.values())
            stats['avg_validity_score'] = sum(scores) / len(scores)
            stats['max_validity_score'] = max(scores)
            stats['min_validity_score'] = min(scores)
        else:
            stats['avg_validity_score'] = 0.0
            stats['max_validity_score'] = 0.0
            stats['min_validity_score'] = 0.0

        return stats

    def get_authenticity_statistics(self) -> Dict:
        """获取真实性统计信息"""
        stats = self.authenticity_stats.copy()

        if stats['verified'] > 0:
            stats['authentic_ratio'] = stats['authentic'] / stats['verified']
            stats['synthetic_ratio'] = stats['synthetic'] / stats['verified']
            stats['unknown_ratio'] = stats['unknown'] / stats['verified']
        else:
            stats['authentic_ratio'] = 0.0
            stats['synthetic_ratio'] = 0.0
            stats['unknown_ratio'] = 0.0

        # 计算真实性置信度分布
        if self.concept_authenticity:
            confidences = [result.confidence for result in self.concept_authenticity.values()]
            stats['avg_authenticity_confidence'] = sum(confidences) / len(confidences)
            stats['max_authenticity_confidence'] = max(confidences)
            stats['min_authenticity_confidence'] = min(confidences)

            # 统计验证方法
            methods = [result.verification_method for result in self.concept_authenticity.values()]
            method_counts = {}
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            stats['verification_methods'] = method_counts
        else:
            stats['avg_authenticity_confidence'] = 0.0
            stats['max_authenticity_confidence'] = 0.0
            stats['min_authenticity_confidence'] = 0.0
            stats['verification_methods'] = {}

        return stats

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        return datetime.now().isoformat()

    def get_graph_stats(self) -> Dict:
        """获取图统计信息"""
        total_nodes = len(self.graph)
        total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2

        # 计算图的连通性
        G = nx.Graph()
        G.add_nodes_from(self.graph.nodes())
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        connected_components = list(nx.connected_components(G))
        largest_component_size = len(max(connected_components, key=len)) if connected_components else 0

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "connected_components": len(connected_components),
            "largest_component_size": largest_component_size,
            "iterations_completed": self.iteration_count,
            "convergence_history": self.convergence_history[-5:] if self.convergence_history else []
        }

# 便捷函数
def expand_concept_graph(
    seed_concepts_file: str,
    config_file: str = "config/config.yaml"
) -> str:
    """
    完整的概念图扩增流程

    Args:
        seed_concepts_file: 种子概念文件路径
        config_file: 配置文件路径

    Returns:
        结果目录路径
    """
    # 加载种子概念
    with open(seed_concepts_file, 'r', encoding='utf-8') as f:
        if seed_concepts_file.endswith('.json'):
            data = json.load(f)
            if 'seed_concepts' in data:
                seed_concepts = data['seed_concepts']
            else:
                seed_concepts = data
        else:
            seed_concepts = [line.strip() for line in f if line.strip()]

    # 创建概念图并运行扩增
    concept_graph = ConceptGraph(seed_concepts, config_file)
    iteration_results = concept_graph.run_full_expansion()

    # 返回结果目录
    output_dir = Path(concept_graph.config['paths']['concept_graph_dir'])
    return str(output_dir)