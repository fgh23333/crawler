"""
政治理论概念分析器
第一部分：对种子概念进行深度思考分析
"""

import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import yaml

try:
    from .api_client import get_client, APIResponse
    from .prompt_templates import PromptTemplates
except ImportError:
    from api_client import get_client, APIResponse
    from prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

class ConceptAnalyzer:
    """政治理论概念分析器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化概念分析器"""
        self.config = self._load_config(config_path)
        self.client = get_client()
        self.templates = PromptTemplates()
        self.results = []

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def analyze_single_concept(self, concept: str) -> Optional[Dict]:
        """分析单个概念"""
        try:
            logger.debug(f"开始分析概念: {concept}")

            messages = [
                {
                    "role": "system",
                    "content": self.templates.get_concept_thinking_system_prompt()
                },
                {
                    "role": "user",
                    "content": self.templates.get_concept_thinking_user_prompt(concept)
                }
            ]

            response = self.client.chat_completion(
                messages=messages,
                model=self.config['api']['model_thinker'],
                temperature=0.7,
                max_tokens=4000
            )

            if response.success:
                analysis_result = {
                    "concept": concept,
                    "analysis": response.content,
                    "model": response.model,
                    "usage": response.usage,
                    "timestamp": self._get_timestamp()
                }
                logger.debug(f"概念分析完成: {concept}")
                return analysis_result
            else:
                logger.error(f"概念分析失败 {concept}: {response.error}")
                return None

        except Exception as e:
            logger.error(f"分析概念 {concept} 时出错: {e}")
            return None

    def analyze_concepts_batch(
        self,
        concepts: List[str],
        batch_size: int = 10,
        max_workers: int = 5,
        save_intermediate: bool = True
    ) -> List[Dict]:
        """批量分析概念"""
        logger.info(f"开始批量分析 {len(concepts)} 个概念")

        results = []
        total_batches = (len(concepts) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="分析概念批次"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(concepts))
            batch_concepts = concepts[start_idx:end_idx]

            logger.info(f"处理批次 {batch_idx + 1}/{total_batches}，概念数量: {len(batch_concepts)}")

            # 并发处理当前批次
            batch_results = self._process_batch(batch_concepts, max_workers)
            results.extend(batch_results)

            # 保存中间结果
            if save_intermediate and batch_results:
                self._save_intermediate_results(batch_idx, batch_results)

            logger.info(f"批次 {batch_idx + 1} 完成，成功分析: {len(batch_results)} 个概念")

        self.results = results
        logger.info(f"批量分析完成，总共成功分析: {len(results)} 个概念")
        return results

    def _process_batch(self, concepts: List[str], max_workers: int) -> List[Dict]:
        """处理单个批次的并发分析"""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_concept = {
                executor.submit(self.analyze_single_concept, concept): concept
                for concept in concepts
            }

            # 收集结果
            for future in as_completed(future_to_concept):
                concept = future_to_concept[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"处理概念 {concept} 时出错: {e}")

        return results

    def _save_intermediate_results(self, batch_idx: int, results: List[Dict]):
        """保存中间结果"""
        try:
            output_dir = Path(self.config['paths']['results_dir']) / "concept_analysis"
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"analysis_batch_{batch_idx:04d}.json"
            filepath = output_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.debug(f"中间结果已保存: {filepath}")

        except Exception as e:
            logger.error(f"保存中间结果失败: {e}")

    def save_final_results(self, filename: str = "concept_analysis_results.json"):
        """保存最终结果"""
        try:
            results_dir = Path(self.config['paths']['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)

            filepath = results_dir / filename

            final_data = {
                "metadata": {
                    "total_concepts": len(self.results),
                    "success_rate": len(self.results) / len(self.results) if self.results else 0,
                    "model": self.config['api']['model_thinker'],
                    "timestamp": self._get_timestamp(),
                    "config": {
                        "batch_size": self.config['concept_expansion']['batch_size'],
                        "max_workers": self.config['concept_expansion']['max_workers']
                    }
                },
                "results": self.results
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)

            logger.info(f"最终结果已保存: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"保存最终结果失败: {e}")
            raise

    def load_concepts_from_file(self, filepath: str) -> List[str]:
        """从文件加载概念列表"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.json'):
                    data = json.load(f)
                    if 'seed_concepts' in data:
                        concepts = data['seed_concepts']
                    else:
                        concepts = data
                else:
                    concepts = [line.strip() for line in f if line.strip()]

            logger.info(f"从文件加载了 {len(concepts)} 个概念: {filepath}")
            return concepts

        except Exception as e:
            logger.error(f"加载概念文件失败: {e}")
            raise

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_statistics(self) -> Dict:
        """获取分析统计信息"""
        if not self.results:
            return {"total": 0, "success": 0, "failed": 0}

        total = len(self.results)

        # 计算token使用统计
        total_tokens = 0
        usage_stats = {}
        for result in self.results:
            if result.get('usage'):
                usage = result['usage']
                total_tokens += usage.get('total_tokens', 0)

                model = result.get('model', 'unknown')
                if model not in usage_stats:
                    usage_stats[model] = {"count": 0, "tokens": 0}
                usage_stats[model]["count"] += 1
                usage_stats[model]["tokens"] += usage.get('total_tokens', 0)

        return {
            "total": total,
            "success": total,
            "failed": 0,
            "total_tokens": total_tokens,
            "usage_by_model": usage_stats,
            "average_tokens_per_concept": total_tokens / total if total > 0 else 0
        }

# 便捷函数
def analyze_concepts_from_file(
    concepts_file: str,
    config_file: str = "config/config.yaml",
    batch_size: int = 10,
    max_workers: int = 5
) -> str:
    """
    从文件加载概念并进行分析

    Args:
        concepts_file: 概念文件路径
        config_file: 配置文件路径
        batch_size: 批处理大小
        max_workers: 最大并发数

    Returns:
        结果文件路径
    """
    analyzer = ConceptAnalyzer(config_file)

    # 加载概念
    concepts = analyzer.load_concepts_from_file(concepts_file)

    # 分析概念
    results = analyzer.analyze_concepts_batch(
        concepts=concepts,
        batch_size=batch_size,
        max_workers=max_workers
    )

    # 保存结果
    result_file = analyzer.save_final_results()

    # 输出统计信息
    stats = analyzer.get_statistics()
    logger.info(f"分析统计: {stats}")

    return result_file