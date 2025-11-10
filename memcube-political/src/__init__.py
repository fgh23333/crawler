"""
MemCube 政治理论概念图扩增系统

基于MemCube框架构建的政治理论领域知识图谱构建工具。
"""

__version__ = "1.0.0"
__author__ = "MemCube Team"
__description__ = "Political Theory Concept Graph Expansion System"

# 导入主要类
from .api_client import OpenAIClient, get_client
from .concept_analyzer import ConceptAnalyzer
from .concept_extractor import ConceptExtractor
from .concept_graph import ConceptGraph, ConceptExpansionResult
from .qa_generator import QAGenerator, QAPair
from .evaluation import (
    ComprehensiveEvaluator,
    ConceptGraphEvaluator,
    QAEvaluator,
    EvaluationReport
)
from .prompt_templates import PromptTemplates

__all__ = [
    # API客户端
    "OpenAIClient",
    "get_client",

    # 核心组件
    "ConceptAnalyzer",
    "ConceptExtractor",
    "ConceptGraph",
    "ConceptExpansionResult",
    "QAGenerator",
    "QAPair",

    # 评估组件
    "ComprehensiveEvaluator",
    "ConceptGraphEvaluator",
    "QAEvaluator",
    "EvaluationReport",

    # 工具类
    "PromptTemplates",
]

# 版本信息
def get_version():
    """获取版本号"""
    return __version__

def get_description():
    """获取描述"""
    return __description__