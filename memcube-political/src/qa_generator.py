"""
政治理论QA生成器
第四部分：基于概念图生成高质量的QA知识对
"""

import json
import logging
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import yaml
import random
import itertools

from .api_client import get_client, APIResponse
from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

@dataclass
class QAPair:
    """问答对数据类"""
    question: str
    answer: str
    difficulty: str  # easy, medium, hard
    type: str  # 概念类型
    concept: Optional[str] = None
    concept_pair: Optional[List[str]] = None
    source: Optional[str] = None
    timestamp: Optional[str] = None

class QAGenerator:
    """政治理论QA生成器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化QA生成器"""
        self.config = self._load_config(config_path)
        self.client = get_client()
        self.templates = PromptTemplates()
        self.generated_qa_pairs = []

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def load_concept_graph(self, graph_file: str) -> Dict:
        """加载概念图"""
        try:
            with open(graph_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            logger.info(f"从文件加载了概念图: {graph_file}")
            logger.info(f"概念节点数: {len(graph_data.get('graph', {}))}")

            return graph_data

        except Exception as e:
            logger.error(f"加载概念图失败: {e}")
            raise

    def generate_single_concept_qa(self, concept: str) -> Optional[List[QAPair]]:
        """为单个概念生成QA对"""
        try:
            logger.debug(f"开始为概念 {concept} 生成QA对")

            messages = [
                {
                    "role": "system",
                    "content": self.templates.get_single_concept_qa_system_prompt()
                },
                {
                    "role": "user",
                    "content": self.templates.get_single_concept_qa_user_prompt(concept)
                }
            ]

            response = self.client.json_completion(
                messages=messages,
                model=self.config['api']['model_qa_generator'],
                temperature=0.7,
                max_tokens=3000
            )

            if response.success:
                qa_data = response.content
                qa_pairs = []

                for qa_item in qa_data.get('qa_pairs', []):
                    qa_pair = QAPair(
                        question=qa_item.get('question', ''),
                        answer=qa_item.get('answer', ''),
                        difficulty=qa_item.get('difficulty', 'medium'),
                        type=qa_item.get('type', 'concept_understanding'),
                        concept=concept,
                        source='single_concept',
                        timestamp=self._get_timestamp()
                    )
                    qa_pairs.append(qa_pair)

                logger.debug(f"概念 {concept} 生成了 {len(qa_pairs)} 个QA对")
                return qa_pairs
            else:
                logger.error(f"为概念 {concept} 生成QA失败: {response.error}")
                return None

        except Exception as e:
            logger.error(f"为概念 {concept} 生成QA时出错: {e}")
            return None

    def generate_concept_pair_qa(self, concept1: str, concept2: str) -> Optional[List[QAPair]]:
        """为概念对生成QA对"""
        try:
            logger.debug(f"开始为概念对 ({concept1}, {concept2}) 生成QA对")

            messages = [
                {
                    "role": "system",
                    "content": self.templates.get_concept_pair_qa_system_prompt()
                },
                {
                    "role": "user",
                    "content": self.templates.get_concept_pair_qa_user_prompt(concept1, concept2)
                }
            ]

            response = self.client.json_completion(
                messages=messages,
                model=self.config['api']['model_qa_generator'],
                temperature=0.7,
                max_tokens=3000
            )

            if response.success:
                qa_data = response.content
                qa_pairs = []

                for qa_item in qa_data.get('qa_pairs', []):
                    qa_pair = QAPair(
                        question=qa_item.get('question', ''),
                        answer=qa_item.get('answer', ''),
                        difficulty=qa_item.get('difficulty', 'medium'),
                        type=qa_item.get('focus', 'theoretical_logic'),
                        concept_pair=[concept1, concept2],
                        source='concept_pair',
                        timestamp=self._get_timestamp()
                    )
                    qa_pairs.append(qa_pair)

                logger.debug(f"概念对 ({concept1}, {concept2}) 生成了 {len(qa_pairs)} 个QA对")
                return qa_pairs
            else:
                logger.error(f"为概念对 ({concept1}, {concept2}) 生成QA失败: {response.error}")
                return None

        except Exception as e:
            logger.error(f"为概念对 ({concept1}, {concept2}) 生成QA时出错: {e}")
            return None

    def select_concept_pairs(self, graph: Dict, max_pairs: int = 1000) -> List[Tuple[str, str]]:
        """选择需要生成QA的概念对"""
        concept_pairs = []
        processed_pairs = set()

        # 优先选择有直接连接的概念对
        for concept, neighbors in graph.items():
            for neighbor in neighbors:
                pair = tuple(sorted([concept, neighbor]))
                if pair not in processed_pairs:
                    concept_pairs.append(pair)
                    processed_pairs.add(pair)
                    if len(concept_pairs) >= max_pairs:
                        break
            if len(concept_pairs) >= max_pairs:
                break

        # 如果还不够，随机选择一些概念组合
        if len(concept_pairs) < max_pairs:
            all_concepts = list(graph.keys())
            additional_pairs_needed = max_pairs - len(concept_pairs)

            for _ in range(additional_pairs_needed):
                concept1, concept2 = random.sample(all_concepts, 2)
                pair = tuple(sorted([concept1, concept2]))
                if pair not in processed_pairs:
                    concept_pairs.append(pair)
                    processed_pairs.add(pair)

        logger.info(f"选择了 {len(concept_pairs)} 个概念对进行QA生成")
        return concept_pairs

    def generate_single_concepts_qa(
        self,
        concepts: List[str],
        batch_size: int = 20,
        max_workers: int = 5
    ) -> List[QAPair]:
        """批量生成单个概念的QA对"""
        logger.info(f"开始为 {len(concepts)} 个概念生成QA对")

        all_qa_pairs = []
        total_batches = (len(concepts) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="生成单概念QA批次"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(concepts))
            batch_concepts = concepts[start_idx:end_idx]

            logger.info(f"处理批次 {batch_idx + 1}/{total_batches}，概念数量: {len(batch_concepts)}")

            # 并发处理当前批次
            batch_qa_pairs = self._process_single_concept_batch(batch_concepts, max_workers)
            all_qa_pairs.extend(batch_qa_pairs)

            # 保存中间结果
            if batch_qa_pairs:
                self._save_intermediate_qa(batch_idx, batch_qa_pairs, "single_concept")

            logger.info(f"批次 {batch_idx + 1} 完成，生成QA对: {len(batch_qa_pairs)} 个")

        logger.info(f"单概念QA生成完成，总共生成: {len(all_qa_pairs)} 个QA对")
        return all_qa_pairs

    def generate_concept_pairs_qa(
        self,
        concept_pairs: List[Tuple[str, str]],
        batch_size: int = 10,
        max_workers: int = 3
    ) -> List[QAPair]:
        """批量生成概念对的QA对"""
        logger.info(f"开始为 {len(concept_pairs)} 个概念对生成QA对")

        all_qa_pairs = []
        total_batches = (len(concept_pairs) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="生成概念对QA批次"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(concept_pairs))
            batch_pairs = concept_pairs[start_idx:end_idx]

            logger.info(f"处理批次 {batch_idx + 1}/{total_batches}，概念对数量: {len(batch_pairs)}")

            # 并发处理当前批次
            batch_qa_pairs = self._process_concept_pair_batch(batch_pairs, max_workers)
            all_qa_pairs.extend(batch_qa_pairs)

            # 保存中间结果
            if batch_qa_pairs:
                self._save_intermediate_qa(batch_idx, batch_qa_pairs, "concept_pair")

            logger.info(f"批次 {batch_idx + 1} 完成，生成QA对: {len(batch_qa_pairs)} 个")

        logger.info(f"概念对QA生成完成，总共生成: {len(all_qa_pairs)} 个QA对")
        return all_qa_pairs

    def _process_single_concept_batch(self, concepts: List[str], max_workers: int) -> List[QAPair]:
        """处理单概念QA生成批次"""
        batch_qa_pairs = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_concept = {
                executor.submit(self.generate_single_concept_qa, concept): concept
                for concept in concepts
            }

            # 收集结果
            for future in as_completed(future_to_concept):
                concept = future_to_concept[future]
                try:
                    qa_pairs = future.result()
                    if qa_pairs:
                        batch_qa_pairs.extend(qa_pairs)
                except Exception as e:
                    logger.error(f"处理概念 {concept} 的QA生成时出错: {e}")

        return batch_qa_pairs

    def _process_concept_pair_batch(self, concept_pairs: List[Tuple[str, str]], max_workers: int) -> List[QAPair]:
        """处理概念对QA生成批次"""
        batch_qa_pairs = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_pair = {
                executor.submit(self.generate_concept_pair_qa, concept1, concept2): (concept1, concept2)
                for concept1, concept2 in concept_pairs
            }

            # 收集结果
            for future in as_completed(future_to_pair):
                concept_pair = future_to_pair[future]
                try:
                    qa_pairs = future.result()
                    if qa_pairs:
                        batch_qa_pairs.extend(qa_pairs)
                except Exception as e:
                    logger.error(f"处理概念对 {concept_pair} 的QA生成时出错: {e}")

        return batch_qa_pairs

    def run_full_qa_generation(self, concept_graph_file: str) -> Dict:
        """运行完整的QA生成流程"""
        logger.info("开始完整的QA生成流程")

        # 加载概念图
        graph_data = self.load_concept_graph(concept_graph_file)
        graph = graph_data['graph']

        # 第一阶段：为单个概念生成QA
        logger.info("\n=== 第一阶段：单概念QA生成 ===")
        concepts = list(graph.keys())
        single_concept_qa = self.generate_single_concepts_qa(
            concepts=concepts,
            batch_size=self.config['qa_generation']['concepts_per_batch'],
            max_workers=self.config['qa_generation']['max_workers']
        )

        # 第二阶段：为概念对生成QA
        logger.info("\n=== 第二阶段：概念对QA生成 ===")
        # 选择重要的概念对（基于连接度）
        concept_pairs = self.select_concept_pairs(
            graph=graph,
            max_pairs=min(len(concepts) * 2, 2000)  # 限制概念对数量
        )
        concept_pair_qa = self.generate_concept_pairs_qa(
            concept_pairs=concept_pairs,
            batch_size=self.config['qa_generation']['concepts_per_batch'] // 2,
            max_workers=self.config['qa_generation']['max_workers'] // 2
        )

        # 合并所有QA对
        all_qa_pairs = single_concept_qa + concept_pair_qa
        self.generated_qa_pairs = all_qa_pairs

        # 质量控制和去重
        filtered_qa_pairs = self._filter_and_deduplicate_qa(all_qa_pairs)

        # 保存最终结果
        self._save_final_qa_results(filtered_qa_pairs, graph_data)

        # 生成统计报告
        statistics = self._generate_statistics(filtered_qa_pairs, graph_data)

        result_summary = {
            "total_concepts": len(concepts),
            "selected_concept_pairs": len(concept_pairs),
            "generated_single_concept_qa": len(single_concept_qa),
            "generated_concept_pair_qa": len(concept_pair_qa),
            "total_generated": len(all_qa_pairs),
            "after_filtering": len(filtered_qa_pairs),
            "statistics": statistics,
            "output_files": self._get_output_files()
        }

        logger.info(f"QA生成流程完成:")
        logger.info(f"  - 单概念QA: {len(single_concept_qa)} 个")
        logger.info(f"  - 概念对QA: {len(concept_pair_qa)} 个")
        logger.info(f"  - 总生成数: {len(all_qa_pairs)} 个")
        logger.info(f"  - 质量过滤后: {len(filtered_qa_pairs)} 个")

        return result_summary

    def _filter_and_deduplicate_qa(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """质量控制和去重"""
        logger.info("开始质量控制和去重")

        # 基本过滤
        filtered = []
        for qa in qa_pairs:
            if (len(qa.question.strip()) >= 10 and
                len(qa.answer.strip()) >= 20 and
                qa.question.count('?') >= 1):  # 至少包含一个问号
                filtered.append(qa)

        # 去重（基于问题相似性）
        deduplicated = []
        seen_questions = set()

        for qa in filtered:
            # 简单的去重策略：基于问题的前50个字符
            question_key = qa.question[:50].strip()
            if question_key not in seen_questions:
                deduplicated.append(qa)
                seen_questions.add(question_key)

        logger.info(f"质量控制完成: {len(qa_pairs)} -> {len(filtered)} -> {len(deduplicated)}")
        return deduplicated

    def _save_intermediate_qa(self, batch_idx: int, qa_pairs: List[QAPair], source_type: str):
        """保存中间QA结果"""
        try:
            output_dir = Path(self.config['paths']['results_dir']) / "qa_generation"
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{source_type}_qa_batch_{batch_idx:04d}.json"
            filepath = output_dir / filename

            # 转换为可序列化格式
            serializable_qa = [asdict(qa) for qa in qa_pairs]

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "batch_index": batch_idx,
                    "source_type": source_type,
                    "qa_pairs": serializable_qa,
                    "count": len(serializable_qa),
                    "timestamp": self._get_timestamp()
                }, f, ensure_ascii=False, indent=2)

            logger.debug(f"中间QA结果已保存: {filepath}")

        except Exception as e:
            logger.error(f"保存中间QA结果失败: {e}")

    def _save_final_qa_results(self, qa_pairs: List[QAPair], graph_data: Dict):
        """保存最终QA结果"""
        try:
            results_dir = Path(self.config['paths']['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)

            # 保存完整QA数据集
            qa_file = results_dir / "political_theory_qa_dataset.json"
            qa_dataset = {
                "metadata": {
                    "total_qa_pairs": len(qa_pairs),
                    "generation_model": self.config['api']['model_qa_generator'],
                    "source_concept_graph_nodes": len(graph_data.get('graph', {})),
                    "timestamp": self._get_timestamp(),
                    "config": self.config['qa_generation']
                },
                "qa_pairs": [asdict(qa) for qa in qa_pairs]
            }

            with open(qa_file, 'w', encoding='utf-8') as f:
                json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

            # 保存为训练格式（类似medxpert格式）
            training_file = results_dir / "political_theory_qa_training.jsonl"
            with open(training_file, 'w', encoding='utf-8') as f:
                for qa in qa_pairs:
                    training_item = {
                        "id": f"pol_qa_{hash(qa.question)}",
                        "question": qa.question,
                        "answer": qa.answer,
                        "options": [],  # 可以后续添加选项
                        "subject_name": "政治理论",
                        "question_type": qa.type,
                        "difficulty": qa.difficulty,
                        "source": qa.source,
                        "concept": qa.concept,
                        "concept_pair": qa.concept_pair
                    }
                    f.write(json.dumps(training_item, ensure_ascii=False) + '\n')

            logger.info(f"最终QA结果已保存:")
            logger.info(f"  完整数据集: {qa_file}")
            logger.info(f"  训练格式: {training_file}")

        except Exception as e:
            logger.error(f"保存最终QA结果失败: {e}")

    def _generate_statistics(self, qa_pairs: List[QAPair], graph_data: Dict) -> Dict:
        """生成统计信息"""
        stats = {
            "total_qa_pairs": len(qa_pairs),
            "by_difficulty": {},
            "by_type": {},
            "by_source": {},
            "average_question_length": 0,
            "average_answer_length": 0,
            "concept_coverage": len(set(qa.concept for qa in qa_pairs if qa.concept)),
            "concept_pair_coverage": len(set(tuple(qa.concept_pair) for qa in qa_pairs if qa.concept_pair))
        }

        # 统计各维度分布
        for qa in qa_pairs:
            # 难度分布
            difficulty = qa.difficulty
            stats["by_difficulty"][difficulty] = stats["by_difficulty"].get(difficulty, 0) + 1

            # 类型分布
            qa_type = qa.type
            stats["by_type"][qa_type] = stats["by_type"].get(qa_type, 0) + 1

            # 来源分布
            source = qa.source
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

        # 计算平均长度
        if qa_pairs:
            avg_question_len = sum(len(qa.question) for qa in qa_pairs) / len(qa_pairs)
            avg_answer_len = sum(len(qa.answer) for qa in qa_pairs) / len(qa_pairs)
            stats["average_question_length"] = round(avg_question_len, 2)
            stats["average_answer_length"] = round(avg_answer_len, 2)

        return stats

    def _get_output_files(self) -> List[str]:
        """获取输出文件列表"""
        results_dir = Path(self.config['paths']['results_dir'])
        return [
            str(results_dir / "political_theory_qa_dataset.json"),
            str(results_dir / "political_theory_qa_training.jsonl")
        ]

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        return datetime.now().isoformat()

# 便捷函数
def generate_political_theory_qa(
    concept_graph_file: str,
    config_file: str = "config/config.yaml"
) -> Dict:
    """
    完整的政治理论QA生成流程

    Args:
        concept_graph_file: 概念图文件路径
        config_file: 配置文件路径

    Returns:
        生成结果摘要
    """
    qa_generator = QAGenerator(config_file)
    return qa_generator.run_full_qa_generation(concept_graph_file)