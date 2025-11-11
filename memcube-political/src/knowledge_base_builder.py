"""
从题库数据构建政治理论知识库
从爬取的题库中提取真实的政治理论概念和理论体系
"""

import json
import re
import logging
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import yaml

logger = logging.getLogger(__name__)

class PoliticalTheoryKnowledgeBaseBuilder:
    """政治理论知识库构建器"""

    def __init__(self, data_path: str = "data/transformed_political_data.json"):
        """初始化构建器"""
        self.data_path = data_path
        self.concepts_db = {}
        self.theoretical_frameworks = {}
        self.authoritative_sources = set()

    def build_knowledge_base(self) -> Dict:
        """从题库数据构建知识库"""
        logger.info("开始从题库数据构建政治理论知识库")

        # 1. 加载题库数据
        qa_data = self._load_qa_data()

        # 2. 提取概念
        concepts = self._extract_concepts_from_qa(qa_data)

        # 3. 构建概念关系
        concept_relations = self._build_concept_relations(qa_data, concepts)

        # 4. 识别理论框架
        frameworks = self._identify_theoretical_frameworks(concepts, concept_relations)

        # 5. 提取权威来源
        sources = self._extract_authoritative_sources(qa_data)

        # 6. 构建知识库
        knowledge_base = self._construct_knowledge_base(concepts, concept_relations, frameworks, sources)

        logger.info(f"知识库构建完成，包含 {len(concepts)} 个概念，{len(frameworks)} 个理论框架")
        return knowledge_base

    def _load_qa_data(self) -> List[Dict]:
        """加载题库数据"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"加载题库数据成功，共 {len(data)} 条记录")
            return data

        except Exception as e:
            logger.error(f"加载题库数据失败: {e}")
            return []

    def _extract_concepts_from_qa(self, qa_data: List[Dict]) -> Dict[str, Dict]:
        """从问答数据中提取概念"""
        concepts = {}
        concept_frequency = Counter()

        for item in qa_data:
            # 从问题和答案中提取概念
            question = item.get('question', '')
            answer = item.get('answer', '')
            subject = item.get('subject', '')

            # 提取候选概念
            candidate_concepts = self._extract_candidate_concepts(question, answer, subject)

            # 统计概念频率
            for concept in candidate_concepts:
                concept_frequency[concept] += 1

            # 为概念收集上下文信息
            for concept in candidate_concepts:
                if concept not in concepts:
                    concepts[concept] = {
                        'frequency': 0,
                        'contexts': [],
                        'related_concepts': set(),
                        'categories': set(),
                        'definitions': [],
                        'sources': set()
                    }

                concepts[concept]['frequency'] += 1
                concepts[concept]['contexts'].append({
                    'question': question,
                    'answer': answer,
                    'subject': subject
                })

                # 添加学科分类
                if subject:
                    concepts[concept]['categories'].add(subject)

        # 过滤低频概念
        min_frequency = 3  # 至少出现3次
        filtered_concepts = {
            concept: info for concept, info in concepts.items()
            if info['frequency'] >= min_frequency
        }

        logger.info(f"提取到 {len(filtered_concepts)} 个候选概念（频率≥{min_frequency}）")
        return filtered_concepts

    def _extract_candidate_concepts(self, question: str, answer: str, subject: str) -> List[str]:
        """从文本中提取候选概念"""
        text = f"{question} {answer}"
        concepts = []

        # 1. 政治理论核心关键词
        political_keywords = [
            '马克思主义', '毛泽东思想', '邓小平理论', '三个代表', '科学发展观',
            '习近平新时代中国特色社会主义思想', '中国特色社会主义', '社会主义', '共产主义',
            '资本主义', '民主', '自由', '平等', '公正', '法治', '人权',
            '人民', '阶级', '革命', '改革', '发展', '现代化', '建设'
        ]

        # 2. 学术术语模式
        academic_patterns = [
            r'[\u4e00-\u9fff]+主义',  # *主义
            r'[\u4e00-\u9fff]+思想',  # *思想
            r'[\u4e00-\u9fff]+理论',  # *理论
            r'[\u4e00-\u9fff]+观',   # *观
            r'[\u4e00-\u9fff]+论',   # *论
            r'[\u4e00-\u9fff]+学',   # *学
            r'[\u4e00-\u9fff]+制度', # *制度
            r'[\u4e00-\u9fff]+体制', # *体制
            r'[\u4e00-\u9fff]+关系', # *关系
            r'[\u4e00-\u9fff]+结构', # *结构
            r'[\u4e00-\u9fff]+体系', # *体系
        ]

        # 3. 重要人物相关概念
        leader_patterns = [
            r'马克思[\u4e00-\u9fff]*', r'恩格斯[\u4e00-\u9fff]*',
            r'列宁[\u4e00-\u9fff]*', r'斯大林[\u4e00-\u9fff]*',
            r'毛泽东[\u4e00-\u9fff]*', r'邓小平[\u4e00-\u9fff]*',
            r'江泽民[\u4e00-\u9fff]*', r'胡锦涛[\u4e00-\u9fff]*',
            r'习近平[\u4e00-\u9fff]*'
        ]

        # 4. 数字+概念模式
        number_patterns = [
            r'一个[\u4e00-\u9fff]+', r'两个[\u4e00-\u9fff]+', r'三个[\u4e00-\u9fff]+',
            r'四个[\u4e00-\u9fff]+', r'五个[\u4e00-\u9fff]+', r'全面[\u4e00-\u9fff]+'
        ]

        # 5. 组合模式
        combined_patterns = [
            r'社会主义[\u4e00-\u9fff]+', r'中国特色社会主义[\u4e00-\u9fff]*',
            r'新民主主义[\u4e00-\u9fff]+', r'人民民主[\u4e00-\u9fff]+',
            r'无产阶级[\u4e00-\u9fff]+', r'资产阶级[\u4e00-\u9fff]+'
        ]

        all_patterns = academic_patterns + leader_patterns + number_patterns + combined_patterns

        # 应用模式匹配
        for pattern in all_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2:  # 至少2个字符
                    concepts.append(match)

        # 检查核心关键词
        for keyword in political_keywords:
            if keyword in text:
                concepts.append(keyword)

        # 去重并过滤
        unique_concepts = list(set(concepts))
        filtered_concepts = [
            concept for concept in unique_concepts
            if self._is_valid_concept(concept)
        ]

        return filtered_concepts

    def _is_valid_concept(self, concept: str) -> bool:
        """验证概念是否有效"""
        # 长度检查
        if len(concept) < 2 or len(concept) > 15:
            return False

        # 排除无意义词汇
        meaningless_words = [
            '什么', '怎么', '为什么', '因为', '所以', '但是', '然后', '或者',
            '这个', '那个', '这些', '那些', '可以', '应该', '必须', '需要'
        ]

        if any(word in concept for word in meaningless_words):
            return False

        # 排除纯数字
        if concept.isdigit():
            return False

        # 排除特殊字符过多
        special_chars = sum(1 for c in concept if not c.isalnum() and c not in '的与和同')
        if special_chars > len(concept) * 0.3:
            return False

        return True

    def _build_concept_relations(self, qa_data: List[Dict], concepts: Dict) -> Dict[str, Set[str]]:
        """构建概念关系"""
        relations = defaultdict(set)

        for item in qa_data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            text = f"{question} {answer}"

            # 找出文本中出现的概念
            concepts_in_text = []
            for concept in concepts.keys():
                if concept in text:
                    concepts_in_text.append(concept)

            # 建立共现关系
            for i, concept1 in enumerate(concepts_in_text):
                for concept2 in concepts_in_text[i+1:]:
                    if concept1 != concept2:
                        relations[concept1].add(concept2)
                        relations[concept2].add(concept1)

        logger.info(f"构建了 {sum(len(rels) for rels in relations.values())} 个概念关系")
        return dict(relations)

    def _identify_theoretical_frameworks(self, concepts: Dict, relations: Dict) -> Dict[str, List[str]]:
        """识别理论框架"""
        frameworks = {}

        # 定义核心框架种子概念
        framework_seeds = {
            "马克思主义理论": ["马克思主义", "历史唯物主义", "辩证唯物主义", "剩余价值"],
            "毛泽东思想": ["毛泽东思想", "实事求是", "群众路线", "独立自主"],
            "邓小平理论": ["邓小平理论", "改革开放", "中国特色社会主义", "社会主义市场经济"],
            "三个代表重要思想": ["三个代表", "先进生产力", "先进文化", "最广大人民"],
            "科学发展观": ["科学发展观", "以人为本", "全面协调可持续", "和谐社会"],
            "习近平新时代中国特色社会主义思想": ["习近平新时代中国特色社会主义思想", "中国梦", "两个一百年", "新时代"],
            "政治制度": ["人民民主专政", "人民代表大会制度", "民主集中制", "政治体制改革"],
            "经济理论": ["社会主义初级阶段", "基本经济制度", "分配制度", "经济发展"]
        }

        # 基于种子概念扩展框架
        for framework_name, seeds in framework_seeds.items():
            framework_concepts = set(seeds)

            # 添加相关的概念
            for seed in seeds:
                if seed in relations:
                    for related in relations[seed]:
                        if related in concepts:
                            framework_concepts.add(related)

            frameworks[framework_name] = list(framework_concepts)

        logger.info(f"识别出 {len(frameworks)} 个理论框架")
        return frameworks

    def _extract_authoritative_sources(self, qa_data: List[Dict]) -> Set[str]:
        """提取权威来源"""
        sources = set()

        # 常见权威来源关键词
        source_keywords = [
            '马克思', '恩格斯', '列宁', '毛泽东', '邓小平', '江泽民', '胡锦涛', '习近平',
            '《资本论》', '《共产党宣言》', '《毛泽东选集》', '《邓小平文选》',
            '党章', '宪法', '全国人大', '党代会', '中央委员会',
            '马克思主义', '毛泽东思想', '邓小平理论'
        ]

        for item in qa_data:
            text = f"{item.get('question', '')} {item.get('answer', '')}"

            for keyword in source_keywords:
                if keyword in text:
                    sources.add(keyword)

        logger.info(f"提取到 {len(sources)} 个权威来源")
        return sources

    def _construct_knowledge_base(self, concepts: Dict, relations: Dict,
                                frameworks: Dict, sources: Set) -> Dict:
        """构建最终知识库"""
        knowledge_base = {
            "concepts": {},
            "frameworks": frameworks,
            "sources": list(sources),
            "metadata": {
                "total_concepts": len(concepts),
                "total_frameworks": len(frameworks),
                "total_sources": len(sources),
                "construction_time": self._get_timestamp()
            }
        }

        # 构建概念详情
        for concept_name, concept_info in concepts.items():
            # 生成定义
            definition = self._generate_concept_definition(concept_name, concept_info)

            # 确定类别
            category = self._determine_concept_category(concept_name, concept_info)

            # 计算真实性分数
            authenticity_score = self._calculate_authenticity_score(concept_name, concept_info)

            knowledge_base["concepts"][concept_name] = {
                "category": category,
                "definition": definition,
                "frequency": concept_info["frequency"],
                "related_concepts": list(relations.get(concept_name, set())),
                "categories": list(concept_info["categories"]),
                "sources": list(sources.intersection({src for context in concept_info["contexts"] for src in [src for src in sources if src in f"{context['question']} {context['answer']}"]})),
                "authenticity_score": authenticity_score,
                "contexts_count": len(concept_info["contexts"])
            }

        return knowledge_base

    def _generate_concept_definition(self, concept: str, concept_info: Dict) -> str:
        """生成概念定义"""
        contexts = concept_info["contexts"]

        # 从上下文中提取定义性语句
        definition_candidates = []

        for context in contexts[:5]:  # 取前5个上下文
            answer = context["answer"]

            # 查找定义性模式
            definition_patterns = [
                f"{concept}是",
                f"{concept}指",
                f"{concept}就是",
                f"所谓{concept}",
                f"关于{concept}"
            ]

            for pattern in definition_patterns:
                if pattern in answer:
                    # 提取定义部分
                    start_idx = answer.find(pattern)
                    if start_idx != -1:
                        # 提取到句号或换行前的内容
                        end_idx = answer.find('。', start_idx)
                        if end_idx == -1:
                            end_idx = answer.find('\n', start_idx)
                        if end_idx == -1:
                            end_idx = start_idx + 100  # 最多取100个字符

                        definition = answer[start_idx:end_idx + 1].strip()
                        if len(definition) > 10:
                            definition_candidates.append(definition)
                        break

        # 返回最合适的定义
        if definition_candidates:
            # 选择最长的定义作为主要定义
            return max(definition_candidates, key=len)
        else:
            # 如果没有找到明确的定义，生成一个通用描述
            frequency = concept_info["frequency"]
            categories = list(concept_info["categories"])
            category_str = f"（{categories[0]}）" if categories else ""
            return f"{concept}是政治理论中的重要概念{category_str}，在题库中出现{frequency}次。"

    def _determine_concept_category(self, concept: str, concept_info: Dict) -> str:
        """确定概念类别"""
        categories = list(concept_info["categories"])

        # 基于概念名称确定类别
        if "主义" in concept:
            return "理论体系"
        elif "思想" in concept:
            return "思想理论"
        elif "制度" in concept or "体制" in concept:
            return "政治制度"
        elif any(word in concept for word in ["经济", "发展", "建设"]):
            return "经济建设"
        elif any(word in concept for word in ["民主", "法治", "权利"]):
            return "政治理论"
        elif any(word in concept for word in ["社会", "文化"]):
            return "社会文化"

        # 基于出现频率最高的学科分类
        if categories:
            category_counter = Counter(categories)
            return category_counter.most_common(1)[0][0]

        return "综合理论"

    def _calculate_authenticity_score(self, concept: str, concept_info: Dict) -> float:
        """计算概念真实性分数"""
        score = 0.0

        # 频率分数 (0-0.3)
        frequency = concept_info["frequency"]
        frequency_score = min(0.3, frequency / 50.0)  # 50次出现为满分
        score += frequency_score

        # 上下文多样性分数 (0-0.3)
        contexts = concept_info["contexts"]
        unique_subjects = len(set(ctx["subject"] for ctx in contexts if ctx["subject"]))
        diversity_score = min(0.3, unique_subjects / 10.0)  # 10个不同学科为满分
        score += diversity_score

        # 概念长度和规范性分数 (0-0.2)
        length_score = 0.1 if 2 <= len(concept) <= 8 else 0.05
        if any(pattern in concept for pattern in ["主义", "思想", "理论", "制度"]):
            length_score += 0.1
        score += min(0.2, length_score)

        # 学科相关性分数 (0-0.2)
        categories = concept_info["categories"]
        if categories:
            relevant_subjects = ["马克思主义", "毛泽东思想", "邓小平理论", "政治学原理"]
            relevance = sum(1 for cat in categories if any(sub in cat for sub in relevant_subjects))
            score += min(0.2, relevance * 0.1)

        return min(1.0, score)

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def save_knowledge_base(self, knowledge_base: Dict, output_path: str = "data/political_theory_knowledge_base.json"):
        """保存知识库"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

            logger.info(f"知识库已保存到: {output_path}")

            # 同时保存为YAML格式便于查看
            yaml_path = output_path.replace('.json', '.yaml')
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(knowledge_base, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"知识库YAML版本已保存到: {yaml_path}")

        except Exception as e:
            logger.error(f"保存知识库失败: {e}")

# 便捷函数
def build_knowledge_base_from_qa(data_path: str = "data/transformed_political_data.json",
                                output_path: str = "data/political_theory_knowledge_base.json") -> Dict:
    """从题库数据构建知识库"""
    builder = PoliticalTheoryKnowledgeBaseBuilder(data_path)
    knowledge_base = builder.build_knowledge_base()
    builder.save_knowledge_base(knowledge_base, output_path)
    return knowledge_base

if __name__ == "__main__":
    # 示例用法
    knowledge_base = build_knowledge_base()
    print(f"知识库构建完成！")
    print(f"概念数量: {knowledge_base['metadata']['total_concepts']}")
    print(f"理论框架: {knowledge_base['metadata']['total_frameworks']}")
    print(f"权威来源: {knowledge_base['metadata']['total_sources']}")