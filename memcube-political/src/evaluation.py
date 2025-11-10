"""
MemCube 评估和验证模块
用于评估概念图和QA数据的质量
"""

import json
import logging
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import yaml
import networkx as nx
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class EvaluationReport:
    """评估报告数据类"""
    concept_graph_metrics: Dict
    qa_quality_metrics: Dict
    coverage_analysis: Dict
    completeness_score: float
    quality_score: float
    overall_score: float
    recommendations: List[str]

class ConceptGraphEvaluator:
    """概念图评估器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化评估器"""
        self.config = self._load_config(config_path)
        self.embedding_model = None

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def load_embedding_model(self):
        """加载embedding模型"""
        if self.embedding_model is None:
            from .embedding_client import get_embedding_client
            logger.info(f"加载embedding模型")
            self.embedding_client = get_embedding_client()

    def evaluate_concept_graph(self, graph_file: str) -> Dict:
        """评估概念图质量"""
        logger.info("开始评估概念图质量")

        try:
            with open(graph_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            graph = graph_data['graph']
            concept_embeddings = graph_data.get('concept_embeddings', {})

            # 基础结构指标
            structure_metrics = self._evaluate_graph_structure(graph)

            # 语义质量指标
            semantic_metrics = self._evaluate_semantic_quality(concept_embeddings)

            # 连接质量指标
            connectivity_metrics = self._evaluate_connectivity_quality(graph)

            # 覆盖度分析
            coverage_metrics = self._evaluate_concept_coverage(graph)

            all_metrics = {
                "structure": structure_metrics,
                "semantic": semantic_metrics,
                "connectivity": connectivity_metrics,
                "coverage": coverage_metrics,
                "total_concepts": len(graph),
                "total_edges": sum(len(neighbors) for neighbors in graph.values()) // 2
            }

            logger.info("概念图评估完成")
            return all_metrics

        except Exception as e:
            logger.error(f"评估概念图失败: {e}")
            raise

    def _evaluate_graph_structure(self, graph: Dict) -> Dict:
        """评估图结构质量"""
        G = nx.Graph()
        G.add_nodes_from(graph.keys())
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        # 基础图指标
        metrics = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "average_clustering": nx.average_clustering(G),
            "is_connected": nx.is_connected(G)
        }

        if G.number_of_nodes() > 0:
            # 连通性指标
            connected_components = list(nx.connected_components(G))
            metrics["connected_components"] = len(connected_components)
            metrics["largest_component_size"] = len(max(connected_components, key=len))
            metrics["largest_component_ratio"] = metrics["largest_component_size"] / G.number_of_nodes()

            # 中心性指标
            degree_centrality = nx.degree_centrality(G)
            metrics["avg_degree_centrality"] = np.mean(list(degree_centrality.values()))
            metrics["max_degree_centrality"] = max(degree_centrality.values())

            # 路径长度指标（仅对连通图或最大连通分量）
            if nx.is_connected(G):
                metrics["avg_shortest_path_length"] = nx.average_shortest_path_length(G)
                metrics["diameter"] = nx.diameter(G)
            elif metrics["largest_component_size"] > 1:
                largest_cc = max(connected_components, key=len)
                subgraph = G.subgraph(largest_cc)
                metrics["avg_shortest_path_length_lcc"] = nx.average_shortest_path_length(subgraph)
                metrics["diameter_lcc"] = nx.diameter(subgraph)

        return metrics

    def _evaluate_semantic_quality(self, concept_embeddings: Dict) -> Dict:
        """评估语义质量"""
        if not concept_embeddings:
            return {"error": "No concept embeddings available"}

        self.load_embedding_model()

        concepts = list(concept_embeddings.keys())
        embeddings = np.array(list(concept_embeddings.values()))

        # 计算语义相似度分布
        similarity_matrix = cosine_similarity(embeddings)

        # 移除对角线（自相似度）
        mask = np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[~mask]

        metrics = {
            "total_concepts": len(concepts),
            "avg_semantic_similarity": np.mean(similarities),
            "std_semantic_similarity": np.std(similarities),
            "min_semantic_similarity": np.min(similarities),
            "max_semantic_similarity": np.max(similarities),
            "high_similarity_pairs": np.sum(similarities > 0.8),
            "low_similarity_pairs": np.sum(similarities < 0.2)
        }

        # 语义多样性评分
        diversity_score = 1 - metrics["avg_semantic_similarity"]
        metrics["semantic_diversity_score"] = diversity_score

        return metrics

    def _evaluate_connectivity_quality(self, graph: Dict) -> Dict:
        """评估连接质量"""
        degrees = [len(neighbors) for neighbors in graph.values()]

        if not degrees:
            return {"error": "Empty graph"}

        metrics = {
            "avg_degree": np.mean(degrees),
            "std_degree": np.std(degrees),
            "min_degree": min(degrees),
            "max_degree": max(degrees),
            "median_degree": np.median(degrees)
        }

        # 度分布分析
        degree_counts = Counter(degrees)
        metrics["unique_degree_values"] = len(degree_counts)
        metrics["most_common_degree"] = degree_counts.most_common(1)[0][0]

        # 连接健康度评估
        isolated_nodes = sum(1 for d in degrees if d == 0)
        low_degree_nodes = sum(1 for d in degrees if d <= 2)
        high_degree_nodes = sum(1 for d in degrees if d >= 10)

        metrics.update({
            "isolated_nodes": isolated_nodes,
            "low_degree_nodes": low_degree_nodes,
            "high_degree_nodes": high_degree_nodes,
            "isolated_node_ratio": isolated_nodes / len(degrees),
            "connectivity_health_score": 1 - (isolated_nodes / len(degrees))
        })

        return metrics

    def _evaluate_concept_coverage(self, graph: Dict) -> Dict:
        """评估概念覆盖度"""
        # 这里可以与外部知识库进行对比
        # 目前基于图内部结构进行评估

        total_concepts = len(graph)
        if total_concepts == 0:
            return {"coverage_score": 0}

        # 计算不同连通分量的覆盖
        G = nx.Graph()
        G.add_nodes_from(graph.keys())
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        connected_components = list(nx.connected_components(G))
        largest_component_size = len(max(connected_components, key=len))

        coverage_metrics = {
            "total_concepts": total_concepts,
            "connected_components": len(connected_components),
            "largest_component_coverage": largest_component_size / total_concepts,
            "fragmentation_score": len(connected_components) / total_concepts
        }

        # 综合覆盖度评分
        coverage_score = coverage_metrics["largest_component_coverage"]
        coverage_metrics["coverage_score"] = coverage_score

        return coverage_metrics

class QAEvaluator:
    """QA数据评估器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化QA评估器"""
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def evaluate_qa_quality(self, qa_file: str) -> Dict:
        """评估QA数据质量"""
        logger.info("开始评估QA数据质量")

        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)

            qa_pairs = qa_data.get('qa_pairs', [])
            if not qa_pairs:
                logger.warning("QA数据为空")
                return {"error": "Empty QA dataset"}

            # 基础质量指标
            basic_metrics = self._evaluate_basic_quality(qa_pairs)

            # 内容多样性指标
            diversity_metrics = self._evaluate_content_diversity(qa_pairs)

            # 概念覆盖指标
            concept_coverage = self._evaluate_concept_coverage_in_qa(qa_pairs)

            # 难度分布指标
            difficulty_metrics = self._evaluate_difficulty_distribution(qa_pairs)

            all_metrics = {
                "basic_quality": basic_metrics,
                "diversity": diversity_metrics,
                "concept_coverage": concept_coverage,
                "difficulty_distribution": difficulty_metrics,
                "total_qa_pairs": len(qa_pairs)
            }

            logger.info("QA质量评估完成")
            return all_metrics

        except Exception as e:
            logger.error(f"评估QA数据失败: {e}")
            raise

    def _evaluate_basic_quality(self, qa_pairs: List[Dict]) -> Dict:
        """评估基础质量"""
        question_lengths = [len(qa.get('question', '').strip()) for qa in qa_pairs]
        answer_lengths = [len(qa.get('answer', '').strip()) for qa in qa_pairs]

        # 基本长度统计
        metrics = {
            "avg_question_length": np.mean(question_lengths) if question_lengths else 0,
            "avg_answer_length": np.mean(answer_lengths) if answer_lengths else 0,
            "min_question_length": min(question_lengths) if question_lengths else 0,
            "max_question_length": max(question_lengths) if question_lengths else 0,
            "min_answer_length": min(answer_lengths) if answer_lengths else 0,
            "max_answer_length": max(answer_lengths) if answer_lengths else 0
        }

        # 质量检查
        valid_questions = sum(1 for length in question_lengths if length >= 10)
        valid_answers = sum(1 for length in answer_lengths if length >= 20)
        questions_with_question_marks = sum(1 for qa in qa_pairs if '?' in qa.get('question', ''))

        metrics.update({
            "valid_question_ratio": valid_questions / len(qa_pairs) if qa_pairs else 0,
            "valid_answer_ratio": valid_answers / len(qa_pairs) if qa_pairs else 0,
            "question_mark_ratio": questions_with_question_marks / len(qa_pairs) if qa_pairs else 0,
            "basic_quality_score": (valid_questions + valid_answers + questions_with_question_marks) / (3 * len(qa_pairs)) if qa_pairs else 0
        })

        return metrics

    def _evaluate_content_diversity(self, qa_pairs: List[Dict]) -> Dict:
        """评估内容多样性"""
        # 分析问题关键词多样性
        question_starts = [qa.get('question', '').strip()[:10] for qa in qa_pairs]
        unique_starts = len(set(question_starts))

        # 分析类型分布
        types = [qa.get('type', 'unknown') for qa in qa_pairs]
        type_distribution = Counter(types)

        # 分析来源分布
        sources = [qa.get('source', 'unknown') for qa in qa_pairs]
        source_distribution = Counter(sources)

        metrics = {
            "unique_question_starts": unique_starts,
            "question_diversity_ratio": unique_starts / len(qa_pairs) if qa_pairs else 0,
            "type_distribution": dict(type_distribution),
            "source_distribution": dict(source_distribution),
            "type_diversity": len(type_distribution),
            "source_diversity": len(source_distribution)
        }

        return metrics

    def _evaluate_concept_coverage_in_qa(self, qa_pairs: List[Dict]) -> Dict:
        """评估QA中的概念覆盖度"""
        single_concept_qa = sum(1 for qa in qa_pairs if qa.get('concept'))
        concept_pair_qa = sum(1 for qa in qa_pairs if qa.get('concept_pair'))

        # 统计覆盖的概念
        covered_concepts = set()
        covered_concept_pairs = set()

        for qa in qa_pairs:
            if qa.get('concept'):
                covered_concepts.add(qa['concept'])
            if qa.get('concept_pair'):
                concept_pair = tuple(sorted(qa['concept_pair']))
                covered_concept_pairs.add(concept_pair)

        metrics = {
            "single_concept_qa": single_concept_qa,
            "concept_pair_qa": concept_pair_qa,
            "covered_concepts": len(covered_concepts),
            "covered_concept_pairs": len(covered_concept_pairs),
            "single_concept_ratio": single_concept_qa / len(qa_pairs) if qa_pairs else 0,
            "concept_pair_ratio": concept_pair_qa / len(qa_pairs) if qa_pairs else 0
        }

        return metrics

    def _evaluate_difficulty_distribution(self, qa_pairs: List[Dict]) -> Dict:
        """评估难度分布"""
        difficulties = [qa.get('difficulty', 'medium') for qa in qa_pairs]
        difficulty_counts = Counter(difficulties)

        metrics = {
            "difficulty_distribution": dict(difficulty_counts),
            "easy_ratio": difficulty_counts.get('easy', 0) / len(qa_pairs) if qa_pairs else 0,
            "medium_ratio": difficulty_counts.get('medium', 0) / len(qa_pairs) if qa_pairs else 0,
            "hard_ratio": difficulty_counts.get('hard', 0) / len(qa_pairs) if qa_pairs else 0,
            "difficulty_diversity": len(difficulty_counts)
        }

        # 理想难度分布评分（平衡分布）
        ideal_ratios = {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}
        difficulty_score = 1 - sum(abs(metrics[f"{level}_ratio"] - ideal_ratios[level])
                                for level in ideal_ratios) / 2
        metrics["difficulty_balance_score"] = difficulty_score

        return metrics

class ComprehensiveEvaluator:
    """综合评估器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化综合评估器"""
        self.config = self._load_config(config_path)
        self.graph_evaluator = ConceptGraphEvaluator(config_path)
        self.qa_evaluator = QAEvaluator(config_path)

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def evaluate_full_system(
        self,
        graph_file: Optional[str] = None,
        qa_file: Optional[str] = None
    ) -> EvaluationReport:
        """评估完整系统"""
        logger.info("开始综合评估")

        concept_graph_metrics = {}
        qa_quality_metrics = {}
        coverage_analysis = {}

        # 评估概念图
        if graph_file and Path(graph_file).exists():
            logger.info("评估概念图...")
            concept_graph_metrics = self.graph_evaluator.evaluate_concept_graph(graph_file)

        # 评估QA质量
        if qa_file and Path(qa_file).exists():
            logger.info("评估QA质量...")
            qa_quality_metrics = self.qa_evaluator.evaluate_qa_quality(qa_file)

        # 覆盖度分析
        if concept_graph_metrics and qa_quality_metrics:
            coverage_analysis = self._analyze_system_coverage(
                concept_graph_metrics, qa_quality_metrics
            )

        # 计算综合评分
        completeness_score = self._calculate_completeness_score(
            concept_graph_metrics, qa_quality_metrics
        )
        quality_score = self._calculate_quality_score(
            concept_graph_metrics, qa_quality_metrics
        )
        overall_score = (completeness_score + quality_score) / 2

        # 生成建议
        recommendations = self._generate_recommendations(
            concept_graph_metrics, qa_quality_metrics, coverage_analysis
        )

        report = EvaluationReport(
            concept_graph_metrics=concept_graph_metrics,
            qa_quality_metrics=qa_quality_metrics,
            coverage_analysis=coverage_analysis,
            completeness_score=completeness_score,
            quality_score=quality_score,
            overall_score=overall_score,
            recommendations=recommendations
        )

        # 保存评估报告
        self._save_evaluation_report(report)

        logger.info(f"综合评估完成，总体评分: {overall_score:.2f}")
        return report

    def _analyze_system_coverage(
        self,
        graph_metrics: Dict,
        qa_metrics: Dict
    ) -> Dict:
        """分析系统覆盖度"""
        coverage = {}

        # 概念图覆盖度
        if 'coverage' in graph_metrics:
            coverage['graph_coverage'] = graph_metrics['coverage']['coverage_score']

        # QA概念覆盖度
        if 'concept_coverage' in qa_metrics:
            total_qa = qa_metrics['total_qa_pairs']
            covered_concepts = qa_metrics['concept_coverage']['covered_concepts']
            coverage['qa_concept_coverage'] = covered_concepts / graph_metrics.get('total_concepts', 1) if graph_metrics.get('total_concepts') else 0

        # 类型覆盖度
        if 'diversity' in qa_metrics:
            coverage['type_diversity'] = qa_metrics['diversity']['type_diversity'] / 5  # 假设最多5种类型

        return coverage

    def _calculate_completeness_score(
        self,
        graph_metrics: Dict,
        qa_metrics: Dict
    ) -> float:
        """计算完整性评分"""
        scores = []

        # 概念图完整性
        if graph_metrics:
            graph_score = 0
            if 'total_concepts' in graph_metrics:
                # 基于概念数量评分
                concept_count = graph_metrics['total_concepts']
                if concept_count >= 1000:
                    graph_score += 0.4
                elif concept_count >= 500:
                    graph_score += 0.3
                elif concept_count >= 100:
                    graph_score += 0.2
                else:
                    graph_score += 0.1

            if 'coverage' in graph_metrics:
                graph_score += graph_metrics['coverage']['coverage_score'] * 0.6

            scores.append(graph_score)

        # QA完整性
        if qa_metrics:
            qa_score = 0
            if 'basic_quality' in qa_metrics:
                qa_score += qa_metrics['basic_quality']['basic_quality_score'] * 0.4

            if 'concept_coverage' in qa_metrics:
                total_qa = qa_metrics['total_qa_pairs']
                if total_qa >= 10000:
                    qa_score += 0.3
                elif total_qa >= 5000:
                    qa_score += 0.25
                elif total_qa >= 1000:
                    qa_score += 0.2
                else:
                    qa_score += 0.1

            if 'diversity' in qa_metrics:
                qa_score += qa_metrics['diversity']['question_diversity_ratio'] * 0.3

            scores.append(qa_score)

        return np.mean(scores) if scores else 0

    def _calculate_quality_score(
        self,
        graph_metrics: Dict,
        qa_metrics: Dict
    ) -> float:
        """计算质量评分"""
        scores = []

        # 概念图质量
        if graph_metrics:
            graph_quality = 0
            if 'connectivity' in graph_metrics:
                graph_quality += graph_metrics['connectivity']['connectivity_health_score'] * 0.5

            if 'semantic' in graph_metrics:
                graph_quality += graph_metrics['semantic']['semantic_diversity_score'] * 0.5

            scores.append(graph_quality)

        # QA质量
        if qa_metrics:
            qa_quality = 0
            if 'basic_quality' in qa_metrics:
                qa_quality += qa_metrics['basic_quality']['basic_quality_score'] * 0.4

            if 'difficulty_distribution' in qa_metrics:
                qa_quality += qa_metrics['difficulty_distribution']['difficulty_balance_score'] * 0.3

            if 'diversity' in qa_metrics:
                qa_quality += qa_metrics['diversity']['type_diversity'] / 5 * 0.3  # 标准化到0-1

            scores.append(qa_quality)

        return np.mean(scores) if scores else 0

    def _generate_recommendations(
        self,
        graph_metrics: Dict,
        qa_metrics: Dict,
        coverage_analysis: Dict
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 概念图改进建议
        if graph_metrics:
            if 'connectivity' in graph_metrics:
                connectivity = graph_metrics['connectivity']
                if connectivity['isolated_node_ratio'] > 0.1:
                    recommendations.append("概念图中孤立节点过多，建议增加概念连接")

                if connectivity['avg_degree'] < 2:
                    recommendations.append("概念平均连接度过低，建议增强概念关联")

            if 'semantic' in graph_metrics:
                semantic = graph_metrics['semantic']
                if semantic['avg_semantic_similarity'] > 0.8:
                    recommendations.append("概念语义相似度过高，可能存在冗余，建议进行去重")

        # QA质量改进建议
        if qa_metrics:
            if 'basic_quality' in qa_metrics:
                basic = qa_metrics['basic_quality']
                if basic['valid_question_ratio'] < 0.9:
                    recommendations.append("部分问题质量不高，建议增加问题长度和完整性检查")

                if basic['valid_answer_ratio'] < 0.9:
                    recommendations.append("部分答案质量不高，建议增加答案长度和详细程度")

            if 'difficulty_distribution' in qa_metrics:
                difficulty = qa_metrics['difficulty_distribution']
                if difficulty['easy_ratio'] > 0.5:
                    recommendations.append("简单题目比例过高，建议增加中等和困难题目")

                if difficulty['hard_ratio'] < 0.1:
                    recommendations.append("困难题目比例过低，建议增加高难度题目")

        # 覆盖度改进建议
        if coverage_analysis:
            if 'qa_concept_coverage' in coverage_analysis:
                if coverage_analysis['qa_concept_coverage'] < 0.5:
                    recommendations.append("QA数据的概念覆盖度不足，建议为更多概念生成QA对")

        if not recommendations:
            recommendations.append("系统质量良好，无明显改进需求")

        return recommendations

    def _save_evaluation_report(self, report: EvaluationReport):
        """保存评估报告"""
        try:
            results_dir = Path(self.config['paths']['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)

            report_file = results_dir / "evaluation_report.json"

            # 转换为可序列化格式
            report_dict = {
                "concept_graph_metrics": report.concept_graph_metrics,
                "qa_quality_metrics": report.qa_quality_metrics,
                "coverage_analysis": report.coverage_analysis,
                "scores": {
                    "completeness": report.completeness_score,
                    "quality": report.quality_score,
                    "overall": report.overall_score
                },
                "recommendations": report.recommendations,
                "timestamp": self._get_timestamp()
            }

            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)

            logger.info(f"评估报告已保存: {report_file}")

        except Exception as e:
            logger.error(f"保存评估报告失败: {e}")

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

# 便捷函数
def evaluate_memcube_system(
    graph_file: Optional[str] = None,
    qa_file: Optional[str] = None,
    config_file: str = "config/config.yaml"
) -> EvaluationReport:
    """
    评估MemCube系统质量

    Args:
        graph_file: 概念图文件路径
        qa_file: QA数据文件路径
        config_file: 配置文件路径

    Returns:
        评估报告
    """
    evaluator = ComprehensiveEvaluator(config_file)
    return evaluator.evaluate_full_system(graph_file, qa_file)