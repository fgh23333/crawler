"""
政治理论概念图扩增系统
第三部分：基于种子概念进行迭代扩增，构建完整的概念图谱
"""

import json
import logging
import pickle
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import yaml
import networkx as nx

from .api_client import get_client, APIResponse
from .prompt_templates import PromptTemplates

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

        # 图结构
        self.graph = {}  # 邻接表表示
        self.concept_embeddings = {}  # concept -> embedding
        self.concept_mapping = {}  # 概念映射表，用于去重

        # 统计信息
        self.iteration_count = 0
        self.convergence_history = []

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
        from .embedding_client import get_embedding_client

        embedding_client = get_embedding_client()
        seed_embeddings = embedding_client.encode(cleaned_seeds, show_progress=True)

        # 构建图
        for concept, embedding in zip(cleaned_seeds, seed_embeddings):
            self.graph[concept] = []
            self.concept_embeddings[concept] = embedding
            self.concept_mapping[concept] = concept

        logger.info(f"概念图初始化完成，种子概念数: {len(cleaned_seeds)}")

    def _is_similar_to_existing(self, new_concept: str, new_embedding: np.ndarray) -> Optional[str]:
        """检查新概念是否与已有概念相似"""
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

                logger.debug(f"概念 {center_concept} 扩增成功，新概念数: {len(new_concepts)}")

                return ConceptExpansionResult(
                    concept_id=concept_id,
                    center_concept=center_concept,
                    status="success",
                    new_concepts=new_concepts,
                    returned_center=returned_center,
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
        concepts_to_expand = list(self.graph.keys())
        total_concepts = len(concepts_to_expand)

        logger.info(f"开始批量概念扩增 {total_concepts} 个概念")

        batch_results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_concept = {}

            for idx, concept in enumerate(concepts_to_expand):
                concept_id = f"concept_{idx:06d}"
                neighbors = self.graph.get(concept, [])
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
        from .embedding_client import get_embedding_client

        embedding_client = get_embedding_client()

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
            new_embeddings = embedding_client.encode(unique_concepts, show_progress=True)

            # 逐个处理需要embedding的概念
            total_concepts = len(unique_concepts)
            for idx, (new_concept, new_embedding) in enumerate(zip(unique_concepts, new_embeddings), 1):
                if idx % 500 == 0 or idx == total_concepts:
                    logger.info(f"  处理进度: {idx}/{total_concepts} ({idx/total_concepts*100:.1f}%)")

                # 检查是否与已有概念相似
                similar_concept = self._is_similar_to_existing(new_concept, new_embedding)

                if similar_concept:
                    # 发现相似概念，建立映射
                    self.concept_mapping[new_concept] = similar_concept
                    concept_targets[new_concept] = similar_concept
                else:
                    # 全新概念，添加到图中并建立自映射
                    self.graph[new_concept] = []
                    self.concept_embeddings[new_concept] = new_embedding
                    self.concept_mapping[new_concept] = new_concept
                    concept_targets[new_concept] = new_concept

        # 添加边（连接到所有相关的中心概念）
        edges_added = 0
        for concept in all_new_concepts:
            target_concept = concept_targets[concept]

            for center_concept in concept_to_centers[concept]:
                # 确保中心概念存在于图中
                if center_concept in self.graph:
                    # 添加双向连接
                    if target_concept not in self.graph[center_concept]:
                        self.graph[center_concept].append(target_concept)
                        edges_added += 1

                    if center_concept not in self.graph[target_concept]:
                        self.graph[target_concept].append(center_concept)
                        edges_added += 1

        nodes_added = len([c for c in concept_targets.values() if c not in self.graph.keys() - set(concept_targets.keys())])
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

        iteration_results = []
        previous_metrics = None

        for iteration in range(self.config['concept_expansion']['max_iterations']):
            logger.info(f"\n=== 迭代 {iteration + 1}/{self.config['concept_expansion']['max_iterations']} ===")

            # 运行单轮迭代
            result = self.run_expansion_iteration()
            iteration_results.append(result)

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
            graph_data = {
                "graph": self.graph,
                "concept_embeddings": {k: v.tolist() for k, v in self.concept_embeddings.items()},
                "concept_mapping": self.concept_mapping,
                "metadata": {
                    "total_iterations": len(iteration_results),
                    "final_nodes": len(self.graph),
                    "final_edges": sum(len(neighbors) for neighbors in self.graph.values()) // 2,
                    "timestamp": self._get_timestamp()
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

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        return datetime.now().isoformat()

    def get_graph_stats(self) -> Dict:
        """获取图统计信息"""
        total_nodes = len(self.graph)
        total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2

        # 计算图的连通性
        G = nx.Graph()
        G.add_nodes_from(self.graph.keys())
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