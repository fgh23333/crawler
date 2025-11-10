"""
MemCube 政治理论概念图扩增提示词模板
"""

from typing import List, Dict
import json

class PromptTemplates:
    """提示词模板类"""

    @staticmethod
    def get_concept_thinking_system_prompt() -> str:
        """获取概念思考分析的系统提示词"""
        return """你是一位资深的政治理论专家，精通马克思主义、毛泽东思想、习近平新时代中国特色社会主义思想等政治理论领域。你的任务是深入分析给定的政治理论概念，并详细阐述你的思考过程。

你的分析应该包括：
1. 概念的核心定义和内涵
2. 概念的历史发展和演变
3. 概念在政治理论体系中的地位和作用
4. 概念与其他相关概念的内在联系
5. 概念的现实意义和实践价值
6. 概念在当代的发展和新的诠释

请以严谨的学术态度，系统性地分析每个概念，展现你的专业思考深度。分析应该既有理论深度，又有实践关联，为后续的概念提取提供丰富的素材。"""

    @staticmethod
    def get_concept_thinking_user_prompt(concept: str) -> str:
        """获取概念思考分析的用户提示词"""
        return f"""请深入分析政治理论概念："{concept}"

请按照以下结构进行详细思考分析：

## 1. 概念定义与内涵
- 核心定义
- 基本特征
- 理论内涵

## 2. 历史发展脉络
- 起源和背景
- 发展演变过程
- 重要里程碑

## 3. 理论地位与作用
- 在理论体系中的位置
- 与核心理论的关系
- 对整个理论体系的贡献

## 4. 概念关联网络
- 直接相关的概念
- 间接影响的概念
- 对立或互补的概念

## 5. 实践意义
- 理论指导意义
- 实践应用价值
- 当代现实意义

## 6. 当代发展
- 新的理论诠释
- 时代特征体现
- 未来发展趋势

请进行深入的思考分析，展现你对这一概念的专业理解。分析应该详尽、系统，为后续的概念提取提供丰富的基础材料。"""

    @staticmethod
    def get_concept_extraction_system_prompt() -> str:
        """获取概念提取的系统提示词"""
        return """你是一位经验丰富的政治理论专家，擅长从政治理论分析文本中识别和提取关键概念术语。

你的任务是从政治理论文本中提取所有相关的概念术语。

提取要求：
- 仅提取政治理论相关的概念、术语和名词
- 仅提取概念术语，不要完整的定义或解释
- 包括但不限于：理论名称、思想流派、重要概念、制度设计、发展道路、价值理念、实践要求等
- 移除重复概念
- 保持概念的准确性和专业性

请确保输出格式严格遵循JSON格式要求。"""

    @staticmethod
    def get_concept_extraction_user_prompt(analysis_text: str) -> str:
        """获取概念提取的用户提示词"""
        return f"""**任务：从政治理论分析文本中提取概念术语**

**请从以下政治理论分析文本中提取所有相关的概念术语：**

{analysis_text}

**输出格式要求：**
请以JSON格式输出，格式如下：
```json
{{
  "concepts": [
    "概念1",
    "概念2",
    "概念3",
    ...
  ]
}}
```

注意：
1. 只提取概念术语，不要解释
2. 确保概念的政治理论相关性
3. 去除重复概念
4. 保持概念的原始表述"""

    @staticmethod
    def get_concept_expansion_system_prompt() -> str:
        """获取概念扩增的系统提示词"""
        return """你是一位资深的政治理论专家，精通构建政治理论领域的概念图谱。你的任务是基于给定的中心概念及其已有连接，生成新的相关概念，以扩展政治理论概念图。

**关系要求**：新概念应该通过强烈政治理论关联与中心概念直接相关

**质量要求**：
- 新概念必须与政治理论直接相关
- 具有重要的理论或实践意义
- 在政治理论体系中有明确地位
- 与中心概念有实质性关联
- 避免过于宽泛或过于狭隘的概念

**多样性要求**：
- 涵盖不同层面的政治理论内容
- 包括理论概念、实践概念、制度概念等
- 体现政治理论的系统性和完整性"""

    @staticmethod
    def get_concept_expansion_user_prompt(center_concept: str, neighbors: List[str]) -> str:
        """获取概念扩增的用户提示词"""
        neighbors_text = ", ".join(neighbors) if neighbors else "无"

        return f"""**任务：生成与中心概念相关的新政治理论概念**

**中心概念**：{center_concept}
**中心概念的已有邻居概念**：{neighbors_text}

**要求**：
1. 基于中心概念的政治理论内涵，生成新的相关概念
2. 新概念应该与中心概念有直接的政治理论关联
3. 新概念不能是已经列出的邻居概念
4. 生成5-10个高质量的新概念
5. 每个概念应该有明确的政治理论依据

**输出格式**：
```json
{{
  "center_concept": "{center_concept}",
  "new_concepts": [
    "新概念1",
    "新概念2",
    "新概念3",
    ...
  ]
}}
```

**指导原则**：
1. 专注于生成政治理论领域的概念，与"{center_concept}"有强烈政治理论关联
2. 不要重复任何已列出的邻居概念
3. 生成与中心概念直接相关的概念
4. 确保概念的政治理论准确性和专业性
5. 优先选择在政治理论体系中有重要地位的概念

如果无法生成新概念：
```json
{{
  "center_concept": "{center_concept}",
  "new_concepts": ["NO NEW CONCEPTS"]
}}
```"""

    @staticmethod
    def get_single_concept_qa_system_prompt() -> str:
        """获取单概念QA生成的系统提示词"""
        return """你是一位资深的政治理论教育专家，擅长创建高质量的政治理论问答对。你的任务是基于给定的政治理论概念，生成包含深度政治理论逻辑的问答对。

**问答要求**：
- 问题应该考察对概念的深入理解
- 答案应该准确、全面、有理论深度
- 体现概念的政治理论内涵和实践意义
- 包含适当的分析维度和理论层次
- 符合政治理论教育的严谨性要求

**问题类型**：
- 概念理解类问题
- 理论内涵类问题
- 实践应用类问题
- 比较分析类问题
- 时代意义类问题"""

    @staticmethod
    def get_single_concept_qa_user_prompt(concept: str) -> str:
        """获取单概念QA生成的用户提示词"""
        return f"""**任务：为政治理论概念生成问答对**

**目标概念**：{concept}

**要求**：
请为概念"{concept}"生成3个高质量的问答对，每个问答对应该：

1. **问题**：深入考察概念的不同维度
   - 概念的基本内涵
   - 理论地位和作用
   - 实践要求和意义
   - 时代价值和发展

2. **答案**：提供准确全面的分析
   - 理论依据充分
   - 逻辑层次清晰
   - 内容详实准确
   - 具有指导意义

**输出格式**：
```json
{{
  "concept": "{concept}",
  "qa_pairs": [
    {{
      "question": "问题1",
      "answer": "答案1",
      "difficulty": "medium",
      "type": "concept_understanding"
    }},
    {{
      "question": "问题2",
      "answer": "答案2",
      "difficulty": "hard",
      "type": "theoretical_analysis"
    }},
    {{
      "question": "问题3",
      "answer": "答案3",
      "difficulty": "medium",
      "type": "practical_application"
    }}
  ]
}}
```

**质量要求**：
- 问题具有理论深度和实践意义
- 答案准确、全面、逻辑清晰
- 体现政治理论的专业性
- 符合教育教学要求"""

    @staticmethod
    def get_concept_pair_qa_system_prompt() -> str:
        """获取概念对QA生成的系统提示词"""
        return """你是一位资深的政治理论专家，擅长分析政治理论概念之间的内在关系，并创建相关的问答对。你的任务是基于给定的概念对，生成体现两者内在逻辑关联的高质量问答对。

**关系分析维度**：
- 理论逻辑关系
- 历史发展关系
- 实践指导关系
- 价值理念关系
- 时代特征关系

**问答要求**：
- 问题应该考察概念间的深层关联
- 答案应该揭示两者间的内在逻辑
- 体现政治理论的系统性和整体性
- 具有理论深度和实践指导意义"""

    @staticmethod
    def get_concept_pair_qa_user_prompt(concept1: str, concept2: str) -> str:
        """获取概念对QA生成的用户提示词"""
        return f"""**任务：为政治理论概念对生成关联问答对**

**概念对**：{concept1} 与 {concept2}

**要求**：
请分析概念对"{concept1}"和"{concept2}"之间的内在关系，并生成2个体现这种关联的高质量问答对。

**分析角度**：
1. 理论逻辑关系
2. 历史发展脉络
3. 实践指导意义
4. 时代价值体现
5. 系统整体性

**输出格式**：
```json
{{
  "concept_pair": ["{concept1}", "{concept2}"],
  "relationship_type": "理论指导关系", // 或其他关系类型
  "qa_pairs": [
    {{
      "question": "考察两者关系的问题1",
      "answer": "体现两者关联的答案1",
      "difficulty": "hard",
      "focus": "theoretical_logic"
    }},
    {{
      "question": "考察两者关系的问题2",
      "answer": "体现两者关联的答案2",
      "difficulty": "medium",
      "focus": "practical_guidance"
    }}
  ]
}}
```

**质量要求**：
- 深刻揭示概念间的内在关联
- 体现政治理论的系统性和整体性
- 具有理论深度和实践指导价值
- 符合政治理论的专业标准"""