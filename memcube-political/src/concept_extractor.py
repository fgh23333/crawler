"""
政治理论概念提取器
第二部分：从LLM思考分析中提取种子概念
"""

import json
import logging
from typing import List, Dict, Set, Optional
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

class ConceptExtractor:
    """政治理论概念提取器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化概念提取器"""
        self.config = self._load_config(config_path)
        self.client = get_client()
        self.templates = PromptTemplates()
        self.extracted_concepts = set()

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def extract_concepts_from_analysis(self, analysis_text: str) -> Optional[List[str]]:
        """从单个分析文本中提取概念"""
        try:
            logger.debug("开始从分析文本中提取概念")

            messages = [
                {
                    "role": "system",
                    "content": self.templates.get_concept_extraction_system_prompt()
                },
                {
                    "role": "user",
                    "content": self.templates.get_concept_extraction_user_prompt(analysis_text)
                }
            ]

            response = self.client.json_completion(
                messages=messages,
                model=self.config['api']['model_extractor'],
                temperature=0.3,
                max_tokens=2000
            )

            if response.success:
                concepts = response.content.get('concepts', [])
                logger.debug(f"成功提取 {len(concepts)} 个概念")
                return concepts
            else:
                logger.error(f"概念提取失败: {response.error}")
                return None

        except Exception as e:
            logger.error(f"从分析文本提取概念时出错: {e}")
            return None

    def extract_from_analysis_results(
        self,
        analysis_results: List[Dict],
        batch_size: int = 20,
        max_workers: int = 5,
        save_intermediate: bool = True
    ) -> List[str]:
        """从分析结果批量提取概念"""
        logger.info(f"开始从 {len(analysis_results)} 个分析结果中提取概念")

        all_concepts = []
        total_batches = (len(analysis_results) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="提取概念批次"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(analysis_results))
            batch_results = analysis_results[start_idx:end_idx]

            logger.info(f"处理提取批次 {batch_idx + 1}/{total_batches}，分析结果数量: {len(batch_results)}")

            # 并发处理当前批次
            batch_concepts = self._process_extraction_batch(batch_results, max_workers)
            all_concepts.extend(batch_concepts)

            # 更新已提取概念集合
            self.extracted_concepts.update(batch_concepts)

            # 保存中间结果
            if save_intermediate and batch_concepts:
                self._save_intermediate_concepts(batch_idx, batch_concepts)

            logger.info(f"提取批次 {batch_idx + 1} 完成，新提取概念: {len(batch_concepts)} 个")

        # 去重和清理
        unique_concepts = self._clean_concepts(all_concepts)
        logger.info(f"批量提取完成，总共提取概念: {len(unique_concepts)} 个")

        return unique_concepts

    def _process_extraction_batch(self, analysis_results: List[Dict], max_workers: int) -> List[str]:
        """处理单个提取批次"""
        batch_concepts = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_result = {
                executor.submit(self._process_single_analysis, result): result
                for result in analysis_results
            }

            # 收集结果
            for future in as_completed(future_to_result):
                analysis_result = future_to_result[future]
                try:
                    concepts = future.result()
                    if concepts:
                        batch_concepts.extend(concepts)
                except Exception as e:
                    logger.error(f"处理分析结果时出错: {e}")

        return batch_concepts

    def _process_single_analysis(self, analysis_result: Dict) -> List[str]:
        """处理单个分析结果"""
        try:
            concept = analysis_result.get('concept', 'unknown')
            analysis_text = analysis_result.get('analysis', '')

            if not analysis_text:
                logger.warning(f"分析结果缺少分析文本: {concept}")
                return []

            concepts = self.extract_concepts_from_analysis(analysis_text)
            if concepts:
                logger.debug(f"从概念 {concept} 的分析中提取了 {len(concepts)} 个概念")
            return concepts or []

        except Exception as e:
            logger.error(f"处理分析结果时出错: {e}")
            return []

    def _clean_concepts(self, concepts: List[str]) -> List[str]:
        """清理和去重概念"""
        # 基本清理
        cleaned = []
        for concept in concepts:
            if concept and isinstance(concept, str):
                concept = concept.strip()
                # 过滤长度
                if 2 <= len(concept) <= 20:
                    cleaned.append(concept)

        # 去重
        unique = list(set(cleaned))

        # 按重要性和长度排序
        unique.sort(key=lambda x: self._get_concept_score(x), reverse=True)

        logger.info(f"概念清理完成: {len(concepts)} -> {len(unique)}")
        return unique

    def _get_concept_score(self, concept: str) -> int:
        """概念重要性评分"""
        length = concept.length if hasattr(concept, 'length') else len(concept)

        # 核心概念通常在4-8个字
        if 4 <= length <= 8:
            return 10
        elif 2 <= length <= 12:
            return 5
        else:
            return 1

    def _save_intermediate_concepts(self, batch_idx: int, concepts: List[str]):
        """保存中间提取的概念"""
        try:
            output_dir = Path(self.config['paths']['results_dir']) / "concept_extraction"
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"extracted_concepts_batch_{batch_idx:04d}.json"
            filepath = output_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "batch_index": batch_idx,
                    "concepts": concepts,
                    "count": len(concepts),
                    "timestamp": self._get_timestamp()
                }, f, ensure_ascii=False, indent=2)

            logger.debug(f"中间提取结果已保存: {filepath}")

        except Exception as e:
            logger.error(f"保存中间提取结果失败: {e}")

    def load_analysis_results(self, filepath: str) -> List[Dict]:
        """加载分析结果"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'results' in data:
                results = data['results']
            else:
                results = data

            logger.info(f"从文件加载了 {len(results)} 个分析结果: {filepath}")
            return results

        except Exception as e:
            logger.error(f"加载分析结果文件失败: {e}")
            raise

    def save_final_concepts(
        self,
        concepts: List[str],
        filename: str = "extracted_concepts_final.json"
    ) -> str:
        """保存最终提取的概念"""
        try:
            results_dir = Path(self.config['paths']['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)

            # 保存JSON格式
            json_filepath = results_dir / filename
            final_data = {
                "metadata": {
                    "total_concepts": len(concepts),
                    "extraction_model": self.config['api']['model_extractor'],
                    "timestamp": self._get_timestamp(),
                    "source_analysis_count": len(self.extracted_concepts)
                },
                "concepts": concepts
            }

            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)

            # 保存文本格式
            txt_filepath = results_dir / filename.replace('.json', '.txt')
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                for concept in concepts:
                    f.write(concept + '\n')

            logger.info(f"最终概念已保存: {json_filepath}")
            logger.info(f"文本格式已保存: {txt_filepath}")

            return str(json_filepath)

        except Exception as e:
            logger.error(f"保存最终概念失败: {e}")
            raise

    def get_extraction_statistics(self) -> Dict:
        """获取提取统计信息"""
        return {
            "total_unique_concepts": len(self.extracted_concepts),
            "extraction_model": self.config['api']['model_extractor']
        }

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

# 便捷函数
def extract_concepts_from_analysis(
    analysis_file: str,
    config_file: str = "config/config.yaml",
    batch_size: int = 20,
    max_workers: int = 5
) -> str:
    """
    从分析结果文件中提取概念

    Args:
        analysis_file: 分析结果文件路径
        config_file: 配置文件路径
        batch_size: 批处理大小
        max_workers: 最大并发数

    Returns:
        提取的概念文件路径
    """
    extractor = ConceptExtractor(config_file)

    # 加载分析结果
    analysis_results = extractor.load_analysis_results(analysis_file)

    # 提取概念
    concepts = extractor.extract_from_analysis_results(
        analysis_results=analysis_results,
        batch_size=batch_size,
        max_workers=max_workers
    )

    # 保存结果
    result_file = extractor.save_final_concepts(concepts)

    # 输出统计信息
    stats = extractor.get_extraction_statistics()
    logger.info(f"提取统计: {stats}")

    return result_file