---
title: 使用 MemOS 构建生产级知识问答系统
---
## **前言**

在构建特定领域的知识问答（QA）系统时，行业面临一个普遍挑战：大型语言模型（LLM）虽知识广博，但在专业领域的精确性和可靠性上仍有不足；而传统的检索增强生成（RAG）方法，虽能引入外部知识，却受限于文档的离散性和缺乏深度逻辑关联，难以应对复杂的推理问答。

本章的目的是给大家一个如何基于MemOS解决这一问题，并提出并实践一套完整的、生产级的知识增强方案demo。我们的核心目标是证明并实现一个关键论断：通过结构化的知识体系，一个经过精心增强的小尺寸模型，其专业能力可以超越未经增强的大型模型。

为实现这一目标，我们设计并构建了一个名为MemCube的动态知识系统。其构建流程遵循一套严谨的工程方法论：

**挖掘并结构化隐性知识**：我们首先系统性地挖掘大型语言模型（LLM）内部关于特定领域的隐性知识，通过"概念图迭代扩增"的方法，将其转化为一个大规模、高覆盖度的显性概念关系图谱。

**生成结构化知识对**：随后，我们将这个概念图谱作为指导，再次利用LLM，生成大量高质量、包含复杂临床逻辑的问答（QA）知识对，作为知识库的核心内容。

**构建并部署知识库**：最终，我们将这些QA知识对组织并载入图数据库（Neo4j），形成一个可供MemOS系统高效检索的动态知识库（MemCube），以增强小模型的领域能力。

本章将通过心血管医学领域的案例，完整展示从零开始构建MemCube的全过程，并通过量化评估，验证该系统在提升模型专业问答能力上的显著效果，为在实际业务中实现低成本、高精度、可解释的AI知识服务提供一套可复用的标准流程。

---

## **章节引言：构建动态知识体系的蓝图**

本章的核心研究目标是验证通过系统性地构建 MemOS 知识库（即 MemCube），参数规模相对较小（如7B级别）的模型能够在专业领域的问答表现上，达到乃至超越大型模型（如32B+级别）相当的性能水平。

本章将引导您完成一次完整的知识工程实践。我们的最终目标是构建一个专用于心血管病学领域的智能问答系统。为达到此目标，我们将遵循一条清晰、分阶段的路径，每一步都以前一步的产出为基础。

本章的整体脉络如下：

### **第一阶段：构建领域知识的基础结构——概念图谱扩增**

这是所有工作的基础。一个高质量的知识库始于一个全面且结构化的领域概念网络，这个网络是对该领域隐性知识的显性表达。

**目标**：创建一个能广泛覆盖心血管病学领域核心概念及其相互关系的大规模图谱。

**获取种子概念**：我们从专业的医学数据集中筛选出心血管领域的材料，并利用LLM初步提取一批高质量的"种子概念"，作为图谱的起点。

**迭代扩增**：以种子概念为基础，我们通过多轮次的自动化流程，让LLM基于图中已有的概念进行联想，不断生成新的相关概念，并建立连接。

**收敛与评估**：我们会引入严格的收敛控制机制（如相似节点合并、新概念增长率监控），确保图谱在达到足够的知识覆盖度后停止生长。最终，通过与外部知识源的关键词进行比对，来量化评估我们图谱的完整性。

### **第二阶段：生成可应用的知识内容——基于图谱生成QA对**

有了概念图谱这个基础结构，我们接下来需要为其填充可以被AI直接理解和利用的具体知识，即问答对。

**目标**：将图谱中抽象的概念和关系，转化为大量具体的、包含临床逻辑的QA对。

**单概念知识生成**：遍历图谱中的每一个核心概念节点，利用LLM为每个概念生成独立的、有深度的临床问题和答案。

**关联知识生成**：对于图中存在连接的"概念对"，我们让LLM生成能够体现两者内在逻辑的、更复杂的关联性问题。

### **第三阶段：组装并部署知识库——MemCube的构建与挂载**

离散的QA数据需要被组织成一个高效的系统。

**目标**：将所有生成的QA数据结构化，并载入图数据库，形成一个可随时被MemOS调用的知识库。

**流程**：

**数据格式化**：我们将所有QA对统一成标准的JSON格式，并为用于检索的问题文本生成向量嵌入（Embedding）。

**图数据库导入**：编写脚本，将格式化后的节点（概念、QA）和边（关系）数据批量、高效地导入Neo4j数据库。

**MemOS挂载**：最后，通过简单的配置，我们将MemOS系统指向这个Neo4j数据库，正式激活我们的心血管MemCube。

### **第四阶段：验证最终成果——系统评测**

完成构建后，我们需要用客观的数据来证明其价值。

**目标**：量化评估装备了MemCube的较小的大模型，相较于未增强的大模型，在专业问答能力上是否实现了超越。

**流程**：我们将构建一套独立的评测集，通过自动化脚本，对不同配置下的模型进行"同题竞赛"，并由更强的模型作为裁判进行打分，最终用胜率和得分来展示我们系统的实际效果。

通过以上四个阶段，您将清晰地看到一个抽象的业务需求，是如何一步步通过严谨的工程方法，最终落地为一个强大、可评估的AI知识系统的。
------------------------------------------------------------------------------------------------------------------------------

## **领域概念图谱的构建与扩增**

### **目标**

将特定领域的非结构化知识，转化为一个高质量、结构化的种子概念集，这是构建智能MemCube的基石。

### **核心理念**

在专业的问答场景中，大型语言模型（LLM）的训练数据中已经隐含了海量的领域知识。挑战在于，如何系统性地将这些知识“提取”并“组织”起来，从而使一个领域专用的、小型的MemCube具备与大模型相当的知识能力。

直接向LLM提问“请提供心血管领域的全部知识”是低效且不可行的。因此，我们必须建立一系列精确的\*\*“概念锚点”\*\*，以此为基础，系统性地勾勒出LLM内部关于该领域的知识图谱。虽然领域知识的覆盖度难以直接量化，但我们可以通过领域内核心概念和关键词的完备性来间接衡量。此步骤的最终目的是尽可能全面地捕获领域概念，为后续的知识提取提供结构化支持。

---

### **Step 1: 种子概念获取**

图谱的构建始于获取一批高质量的“种子概念”。这些初始概念是图谱迭代扩增的起点，其质量直接影响模型构建领域知识框架的速度和效率。

为确保种子概念的专业性和覆盖度，本实验选用公开的医学专家试题数据集 `medxpert` 作为初始数据源。该数据集包含明确的医学领域分类，便于我们精准筛选心血管领域的相关知识。

由衷感谢：MedXpert公开Benchmark

数据下载地址：https://raw.githubusercontent.com/TsinghuaC3I/MedXpertQA/refs/heads/main/eval/data/medxpertqa/input/medxpertqa_text_input.jsonl

以下代码展示了数据加载与筛选的过程。

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_URL'] = 'https://hf-mirror.com'
os.environ['HF_HUB_BASE_URL'] = 'https://hf-mirror.com'

import glob
import pickle
import requests
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import uuid
import sys
import ijson
from decimal import Decimal
from neo4j import GraphDatabase
from collections import defaultdict
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re
from collections import Counter
import random
from sentence_transformers import SentenceTransformer
from json_repair import repair_json
```

我们约定以下的环境变量：

```python
# api url; api key是您的专属设定
# 对于具体使用的llm模型：
# 1. 在总结归纳seed concepts阶段，我们使用强模型MEDXPERT_THINKER=o3对现有的medxpert心血管领域100+问题进行扩增，抽取其分析问题的思考过程。我们的目的并非让o3正确回答每个问题，而是想在其思考分析每个心血管问题之后，由其思考过程总结心血管种子概念。
# 2. 在提取出o3的思维过程后，我们用MEDXPERT_THINK_EXTRACTOR=gpt-4o对其思考过程抽取心血管概念名词作为种子概念
# 对以上两环节，我们的目的是构建种子概念库；如果您有您的专属领域文档库，您大可以跳过这一阶段，直接由您的领域文档抽取出您感兴趣的种子概念。
# 3. 在种子概念库建立完成后，我们将进行领域概念图扩增。这一环节我们建议根据我们的实验结果和您的成本预算酌情选择您的模型。在我们的试验阶段，我们以CONCEPT_GRAPH_EXTENDER=gpt-4omini为参考。
# 4. 构建完成领域概念图后，我们将对(a)每个概念进行单独的qa生成(b)每个有意义的概念对进行qa生成。由于qa生成依赖模型能力，我们建议您使用强模型。在我们的实验阶段，我们以QA_SYNTHESIZER=gpt-4o为参考。
# 我们对所有使用的embedding模型应用英文embedding开源模EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
api_url="your api url"
api_key="your api key"

#各种可能的model：
MEDXPERT_THINKER = 'o3'
MEDXPERT_THINK_EXTRACTOR = 'gpt-4o'
CONCEPT_GRAPH_EXTENDER = 'gpt-4o-mini'
QA_SYNTHESIZER = 'gpt-4o'
EMBEDDING_MODEL = 'nomic-ai/nomic-embed-text-v1.5'
```

```python
# 从最新的medxpert数据集提取种子概念，其优势在有对应领域的划分
import json
from collections import Counter

data = []
with open("medxpertqa_text_input.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line.strip()))

# 取出心血管领域试题
body_system_counts = Counter(entry["body_system"] for entry in data)
heart_data = []
for i in range(len(data)):
    if data[i]['body_system']=='Cardiovascular':
        heart_data.append(data[i])  
```

为了实现与大型语言模型API的稳定、高效交互，我们构建了一个模块化的API客户端。该客户端集成了连接池、基于指数退避策略的自动重试机制以及请求超时控制，确保了在高并发调用下的鲁棒性。同时，我们定义了标准化的数据结构（`AnalysisResult`）来统一存放请求结果，便于后续处理。

```python

@dataclass
class AnalysisResult:
    """分析结果数据类"""
    status: str  # success, api_error
    question_id: str
    input_data: Dict
    response: Optional[str] = None
    error_details: Optional[str] = None
    processing_time: Optional[float] = None

class APIClient:
    """API调用客户端"""

    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

        # 创建session并配置连接池和重试策略
        self.session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )

        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retry_strategy)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        })

    def call_api(self, messages: List[Dict], timeout: int = 120) -> Dict:
        """API调用"""
        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }

        try:
            response = self.session.post(url=self.api_url, json=data, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return {
                "status": "success",
                "content": result['choices'][0]['message']['content']
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e)
            }
  
```

我们通过一个精心设计的 `PromptTemplate`类来封装和生成与LLM交互的指令。该模板将LLM的角色设定为一位资深的临床医学教授，并要求其以结构化的方式，系统性地拆解和分析医学问题。这种结构化的输出是后续精确提取信息的关键。

```python

class PromptTemplate:
    """提示词模板类"""

    @staticmethod
    def get_system_prompt() -> str:
        return """You are a world-renowned clinical professor at a top teaching hospital with over 20 years of experience. Your thinking is grounded in evidence-based medicine, characterized by rigorous logic and clear reasoning.

Your core mission extends beyond solving clinical problems—you must **teach young doctors and medical students your decision-making process**. Therefore, when analyzing any case, you must:

1. **Systematic Deconstruction**: Begin by breaking down the problem from a macro perspective, identifying core clinical contradictions and key information.
2. **Comprehensive Evaluation**: Provide independent and thorough analysis of all possibilities (including every option), without skipping any.
3. **Clear Reasoning**: Explicitly articulate the "because-therefore" logic behind each judgment, clearly stating which specific clinical indicators, guideline consensus, or pathophysiological principles your decisions are based on.
4. **Principle Extraction**: After analysis, skillfully distill complex individual case decision processes into reusable, instructive core principles.

Your language should combine authority with clarity, enabling listeners to fully replicate your thought process."""

    @staticmethod
    def get_analysis_prompt(question_data: Dict) -> str:
        question = question_data['question']
        options = question_data['options']
  
        options_text = ""
        for opt in options:
            options_text += f"({opt['letter']}) {opt['content']}\n"
        return f"""Analyze this cardiovascular medicine multiple-choice question systematically and select the SINGLE CORRECT ANSWER. Provide a comprehensive analysis that demonstrates expert clinical reasoning. 

    **[Clinical Problem]**
    ---
    {question}

    Answer Choices:
    {options_text}
    ---


    **[Analysis Structure]**

    **Part 1: Clinical Context Analysis**

    Begin by establishing the clinical foundation for this question:

    * **Clinical Scenario Identification**: What is the primary clinical situation being presented? (e.g., diagnostic workup, treatment decision, risk stratification, pathophysiology question, etc.)

    * **Key Clinical Elements**: What are the most important clinical details, patient characteristics, findings, or parameters mentioned in the question stem? Why are these details clinically significant?

    * **Question Focus**: What specific aspect of clinical medicine is this question testing? What clinical knowledge or decision-making skill is being assessed?

    * **Relevant Clinical Framework**: What established clinical guidelines, diagnostic criteria, or treatment algorithms are relevant to answering this question?

    **Part 2: Systematic Option Analysis**

    Now analyze each answer choice methodically:

    **Option (A): **
    * **Clinical Evaluation**: How does this option relate to the clinical scenario? What would be the clinical implications if this were the correct choice?
    * **Evidence-Based Assessment**: Based on current guidelines, evidence, and pathophysiology, is this option clinically appropriate? Why or why not?

    **Option (B): **
    * **Clinical Evaluation**: [Same analysis format]
    * **Evidence-Based Assessment**: [Same analysis format]

    [Continue this systematic analysis for each option through the last one]

    **Part 3: Final Answer and Clinical Synthesis**

    * **Clinical Summary**: Briefly synthesize the key clinical scenario from the question stem and the critical findings from my analysis of each option.

    * **Selected Answer**: Based on my systematic analysis, the correct answer is: **(Letter) [Brief restatement of the correct option]**

    * **Answer Justification**: Concisely explain why this is the best answer, focusing on the most compelling clinical evidence and reasoning.

    * **Option Comparison Summary**: Provide a brief comparative overview of why the chosen option is superior to the other alternatives, highlighting the key clinical distinctions.

    * **Clinical Teaching Point**: Detailedly summarize the essential clinical medicine principle demonstrated by this question as a practical clinical pearl.

    **CRITICAL REQUIREMENT**: End with a clear statement: "**FINAL ANSWER: (a [single] Letter, DO NOT GIVE DETAILED CONTENT IN OPTION)**"

    Begin your analysis now."""
```

为了将上述组件串联起来，我们设计了两个处理器。`CardioAnalysisProcessor` 负责处理单个问题，它组合Prompt、调用API并返回结构化结果。而 `BatchProcessor` 则利用线程池（`ThreadPoolExecutor`）实现了高并发处理，能够将所有筛选出的问题分批、并行地交由 `CardioAnalysisProcessor`处理，并自动保存每批次的结果。这一设计是实现生产级数据处理效率的必要保障。

```python
# 抽取llm思考过程，为后续的种子概念做准备
class CardioAnalysisProcessor:
    """心血管问题分析处理器"""

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.template = PromptTemplate()

    def process_single_question(self, question_data: Dict, question_id: str) -> AnalysisResult:
        """处理单个心血管问题"""
        start_time = time.time()
  
        # 构建消息
        system_prompt = self.template.get_system_prompt()
        user_prompt = self.template.get_analysis_prompt(question_data)
  
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 调用API
        api_result = self.api_client.call_api(messages, timeout=120)
        processing_time = time.time() - start_time

        if api_result["status"] != "success":
            return AnalysisResult(
                status="api_error",
                question_id=question_id,
                input_data=question_data,
                error_details=api_result["error"],
                processing_time=processing_time
            )

        return AnalysisResult(
            status="success",
            question_id=question_id,
            input_data=question_data,
            response=api_result["content"],
            processing_time=processing_time
        )

class BatchProcessor:
    """批量处理器"""

    def __init__(self, processor: CardioAnalysisProcessor, output_dir: str = "cardio_analysis"):
        self.processor = processor
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_questions_list(self, heart_data: List[Dict], max_workers: int = 10, 
                              batch_size: int = 50, batch_delay: int = 1) -> Dict:
        """批量处理心血管问题"""
        total_questions = len(heart_data)
        print(f"开始批量处理 {total_questions} 个心血管临床问题")
        print(f"批次大小: {batch_size}, 最大并发: {max_workers}")

        all_results = {}
        batch_num = 1

        # 分批处理
        for i in range(0, total_questions, batch_size):
            batch_data = heart_data[i:i + batch_size]
            print(f"\n处理批次 {batch_num}: 问题 {i+1}-{min(i+batch_size, total_questions)} ({len(batch_data)} 个)")

            batch_start_time = time.time()
            batch_results = self._process_batch(batch_data, max_workers, i)
            batch_end_time = time.time()

            # 保存批次结果
            self._save_batch_results(batch_results, batch_num, batch_start_time)

            all_results.update(batch_results)

            print(f"批次 {batch_num} 完成，耗时: {batch_end_time - batch_start_time:.2f} 秒")

            batch_num += 1

            # 批次间休息
            if i + batch_size < total_questions:
                print(f"批次间休息 {batch_delay} 秒...")
                time.sleep(batch_delay)

        print(f"\n所有批次处理完成！总计处理 {total_questions} 个问题")
        return all_results

    def _process_batch(self, batch_data: List[Dict], max_workers: int, start_index: int) -> Dict:
        """处理单个批次"""
        batch_results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_question = {}
            for idx, question_data in enumerate(batch_data):
                # 使用原始ID或生成新ID
                question_id = question_data.get('id', f"cardio_{start_index + idx:06d}")
                future = executor.submit(self.processor.process_single_question, question_data, question_id)
                future_to_question[future] = (question_id, question_data)

            # 收集结果
            completed = 0
            for future in as_completed(future_to_question):
                question_id, question_data = future_to_question[future]
                #try:
                result = future.result()
                batch_results[question_id] = result

                # 简单状态显示
                status_symbol = "✓" if result.status == "success" else "✗"
                completed += 1
        
                if completed % 5 == 0 or completed == len(batch_data):
                    success_count = sum(1 for r in batch_results.values() if r.status == "success")
                    print(f"  已完成: {completed}/{len(batch_data)} (成功: {success_count}) {status_symbol}")

        return batch_results

    def _save_batch_results(self, batch_results: Dict, batch_num: int, start_time: float):
        """保存批次结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cardio_analysis_batch_{batch_num:03d}_{timestamp}.pkl"
        filepath = os.path.join(self.output_dir, filename)

        # 统计信息
        total_count = len(batch_results)
        success_count = sum(1 for r in batch_results.values() if r.status == "success")
        error_count = total_count - success_count

        # 构建保存数据
        save_data = {
            "metadata": {
                "batch_num": batch_num,
                "timestamp": timestamp,
                "start_time": start_time,
                "total_questions": total_count,
                "successful_analyses": success_count,
                "failed_analyses": error_count,
                "success_rate": success_count / total_count if total_count > 0 else 0
            },
            "results": batch_results
        }

        # 保存到文件
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"  批次结果已保存: {filename}")
        print(f"  成功: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

        return filepath
```

通过运行上述批量处理流程，我们将所有心血管领域的试题发送给LLM进行深度分析，并将返回的详细分析文本（包含临床情境分析、选项辨析、核心原理总结等）收集起来，为下一步的概念提取做好了数据准备。

**注意**：请在 `process_heart_data`函数中填入您自己的 `api_url`, `api_key`, 和 `model_name`。

```python
def process_heart_data(heart_data: List[Dict], api_url: str, api_key: str, model_name: str, max_workers: int = 10, 
                      batch_size: int = 50, output_dir: str = "cardio_analysis"):
    """处理heart_data的便捷函数"""
    print(f"准备处理 {len(heart_data)} 个心血管临床问题")

    # 初始化API客户端
    api_client = APIClient(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name
    )

    # 初始化处理器
    processor = CardioAnalysisProcessor(api_client)
    batch_processor = BatchProcessor(processor, output_dir=output_dir)

    # 批量处理
    results = batch_processor.process_questions_list(
        heart_data=heart_data,
        max_workers=max_workers,
        batch_size=batch_size,
        batch_delay=1
    )

    return results
```

```python
# 批量抽取每个qa的思考过程
# max_workers: 访问api并发数
# batch_size：多少次api访问完成后进行一次访问结果文件保存

results = process_heart_data(heart_data, max_workers=100, batch_size=400, output_dir="cookbooktest", 
                             api_url = api_url,
                             api_key = api_key,
                             model_name = MEDXPERT_THINKER)
textlist = [results[i].response for i in results.keys()]
```

现在，我们拥有了大量由LLM生成的、关于心血管问题的深度分析文本。下一步任务是从这些非结构化的文本中，提取出所有核心的医学概念。我们再次借助LLM来完成此任务，成功的关键依然在于一个设计精良的Prompt。`PromptTemplate`被重新设计，用于指导LLM扮演心血管专家的角色，遵循一系列严格的提取原则（如提取核心术语、避免描述性组合、输出为标准JSON格式等），确保最终得到的概念列表干净、规范。

```python
class PromptTemplate:

    @staticmethod
    def get_system_prompt() -> str:

        return """You are an experienced cardiovascular specialist, skilled in identifying and extracting medical concept terms from clinical texts.

Your task is to extract all relevant concept terms from cardiovascular clinical texts.

Extraction principles:
- Only extract cardiovascular-related medical concepts, terms, and nouns
- Only concept terms, not complete definitions or explanations
- Prefer single core terms (e.g., "myocardial infarction", "hypertension", "echocardiography")
- Use phrases only when they represent standard medical terminology that cannot be meaningfully separated (e.g., "atrial fibrillation", "coronary artery disease")
- Avoid descriptive combinations (e.g., "severe hypertension" → "hypertension")
- Avoid overly vague terms (e.g., "heart problem")
- Include but not limited to disease names, examination methods, drug treatments, anatomical structures, physiological indicators, clinical manifestations, assessment tools, and all other related concepts
- Remove duplicate concepts
- Sort by importance

Please ensure the output format strictly follows JSON format requirements."""

    @staticmethod
    def get_extraction_prompt(text_content: str) -> str:

        return f"""**Task: Extract concept terms from cardiovascular clinical text**

**Please extract all relevant cardiovascular concept terms from the following text:**

---
{text_content}
---

**Output format (strictly follow JSON format):**
```json
{{
"concepts": [
    "concept1",
    "concept2",
    "concept3",
    "..."
]
}}"""
```

经过上述步骤，我们成功从海量的分析文本中萃取出了一个初步的心血管领域种子概念列表。这份列表为后续概念图的迭代扩增奠定了坚实的基础。

**示例成果:**

```python
seed_concepts = [
    'blood pressure', 'step-up', 'glucagon', 'therapeutic anticoagulation'...
]
```

### **概念图迭代扩增方法**

#### **核心目标**

通过迭代式扩增，实现概念图对目标领域的完整覆盖。我们采用基于LLM的逐步扩展策略，从种子概念集出发，逐轮扩展，最终构建一个全面的领域概念图谱。

#### **迭代流程**

迭代的基本流程如下：

- **输入**: 上一轮迭代生成的概念图。
- **处理**: 对图中的每一个概念节点，将其自身以及已知的相邻概念作为上下文信息提供给LLM，并请求LLM生成更多与该中心概念直接相关的其他概念。
- **输出**: 将LLM返回的新概念经过后处理（如去重），添加到概念图中，作为下一轮迭代的输入。

#### **收敛机制与控制参数**

随着迭代的进行，新生成的概念会越来越多地与图中已有的概念重合，从而使迭代过程自然收敛。我们设计了三个核心参数来精确控制这一过程：

1. **相似节点合并阈值 (`similarity_threshold`)**

   * **机制**: 使用embedding模型计算概念的向量表示，通过余弦相似度判断两个概念的语义相似性。
   * **作用**: 当两个概念的相似度超过设定的阈值时，它们将被合并为图中的单个节点。
   * **影响**: 此参数直接控制了概念图的“粒度”和扩展速度，是平衡图谱完整性与计算成本的关键。
2. **新概念增长率 (`new_concept_rate_threshold`)**

   * **机制**: 计算当前轮次生成的新概念中，在原有图谱中不存在的（即“全新”）概念的比例。
   * **作用**: 当这个比例低于设定的阈值时，可以认为图谱在概念覆盖广度上趋于饱和，从而停止迭代。
3. **新边增长率 (`new_edge_rate_threshold`)**

   * **机制**: 计算在图谱中已有的旧概念之间，本轮新建立的连接（边）数量的增长率。
   * **作用**: 当概念间的关系网络趋于完善，新连接的增长显著放缓时，停止迭代。
   * **意义**: 此指标主要反映概念图内部结构的完整性。

#### **参数选择策略与收敛性分析**

在缺乏外部评估数据集的情况下，如何选择超参数并判断收敛，是实践中的关键问题。

1. **相似节点合并阈值 (`similarity_threshold`)**: **控制迭代速率的核心指标。**
   理论上，可以不进行任何相似节点合并，以构建最全面的概念图，但这会带来巨大的计算成本。因此，设置一个合理的阈值至关重要。最终存在于图中的每个核心节点，可被理解为在该阈值所定义的语义空间内的一个**代表性概念**。此参数是平衡图谱完整性与计算效率的核心调节器。对于需要极致覆盖的学术研究场景，可设置较高阈值（如0.95）；对于注重成本效益的实际应用，可设置较低阈值（如0.80）。
2. **新概念增长率**: **最先收敛但可能停滞的指标。**
   随着迭代的进行，新概念的增长率会率先呈现收敛趋势。然而，实践中发现，当收敛到一定程度后，该指标可能会停滞在一个较低水平，而不会完全趋近于零。其根本原因是LLM在进行概念联想时，可能会逐渐“漂移”出严格的领域边界。因此，不能单独依赖此指标来判断迭代是否完成。
3. **新边增长率**: **最终稳定收敛的指标。**
   当目标领域（如心血管领域）的大部分核心概念已被捕获，且它们之间的主要关系已经建立后，后续新增的节点大多位于领域的“外沿”。这些外沿节点难以与图谱核心区的旧节点建立新的、有意义的连接。这导致了旧概念之间新边的增长率稳定下降并最终收敛。相比新概念增长率，该指标受LLM“超领域扩展”特性的影响较小，是判断概念图**结构完整性**的可靠指标。

**实践策略建议**：在有限成本下，采用平衡的 `similarity_threshold`（如0.80），并结合肘部法则（Elbow Method）观察收敛曲线。当新概念增长率和新边增长率连续1-2轮无明显变化，进入平台期时，即可停止迭代。

---

### **核心代码实现**

我们将上述理论转化为一个能够自我生长的概念图谱系统。

#### **1. 核心数据结构：`ConceptGraph` 类**

系统的核心是 `ConceptGraph`类，它是一个集成了语义理解和动态更新能力的“智能图谱”。

* **初始化 (`__init__`)**: 从一批经过预处理的种子概念开始，计算它们的向量嵌入（embedding），构建图的初始状态。
* **智能去重 (`_is_similar_to_existing`)**: 这是控制图质量和规模的关键。它利用语义向量的余弦相似度来判断一个新概念是否与图中已有的概念在语义上“相近”。只有当相似度超过设定的 `similarity_threshold`时，概念才会被合并。
* **动态更新 (`update_graph`)**: 这是图“生长”的核心动力。该方法接收LLM扩增出的新概念，通过智能去重机制，将真正“全新”的概念作为新节点加入图中，并与源概念建立连接。
* **状态监控 (`calculate_metrics`, `get_graph_stats`)**: 这些方法负责计算我们定义的各项收敛指标（如新概念增长率、新边增长率）和图的统计数据（节点数、边数），从而能够量化地监控每一轮迭代的效果。

```python
class ConceptGraph:
  
    @classmethod
    def from_graph_dict(cls, graph_dict: Dict[str, List[str]], concept_mapping, model, similarity_threshold):
        """
        从保存的图字典重新构建ConceptGraph
        Args:
            graph_dict: 保存的邻接字典
            model: SentenceTransformer模型实例
            similarity_threshold：判为相似概念的threshold
            concept_mapping: 保存的所有见过的相似概念mapping，例如{'hearts':'heart'}
        Returns:
            ConceptGraph实例
        """
        if model is None:
            raise ValueError("模型参数不能为None，请先使用load_embedding_model()加载模型")
  
        # 创建实例但不初始化
        instance = cls.__new__(cls)
        instance.model = model
        instance.graph = graph_dict.copy()
        instance.concept_embeddings = {}
        instance.concept_mapping = concept_mapping  
        instance.similarity_threshold = similarity_threshold
  
        # 重新计算所有概念的embedding并建立自映射
        all_concepts = list(graph_dict.keys())
        if all_concepts:
            print(f"正在为 {len(all_concepts)} 个概念重新计算embedding...")
            all_embeddings = model.encode(all_concepts)
    
            for concept, embedding in zip(all_concepts, all_embeddings):
                instance.concept_embeddings[concept] = embedding
                #instance.concept_mapping[concept] = concept  # 建立自映射
    
            print("ConceptGraph重建完成")
  
        return instance
  
    def __init__(self, seed_concepts: List[str], model, similarity_threshold):
        """
        从种子概念初始化图，并建立embedding库
        Args:
            seed_concepts: 已经过外部去重的种子概念列表
            model: SentenceTransformer模型实例（必需）
            similarity_threshold：判为相似概念的threshold
        """
        if model is None:
            raise ValueError("模型参数不能为None，请先使用load_embedding_model()加载模型")
    
        self.model = model
        self.graph = {}
        self.concept_embeddings = {}  # 维护concept -> embedding的映射
        self.concept_mapping = {}     # 添加概念映射表
        self.similarity_threshold = similarity_threshold  # embedding相似度阈值
  
        # 清理种子概念
        cleaned_seeds = [concept.strip() for concept in seed_concepts if concept.strip()]
  
        print(f"正在为 {len(cleaned_seeds)} 个种子概念计算embedding...")
  
        # 批量计算embedding
        if cleaned_seeds:
            seed_embeddings = self.model.encode(cleaned_seeds)
    
            # 建立初始图、embedding库和映射表
            for concept, embedding in zip(cleaned_seeds, seed_embeddings):
                self.graph[concept] = []
                self.concept_embeddings[concept] = embedding
                self.concept_mapping[concept] = concept  # 建立自映射
  
        print(f"初始化概念图完成，种子概念数: {len(cleaned_seeds)}")
  
    def _get_target_concept(self, concept: str) -> Optional[str]:
        """统一的概念映射查找"""
        return self.concept_mapping.get(concept)
  
    def _is_similar_to_existing(self, new_concept: str, new_embedding: np.ndarray) -> Optional[str]:
        """
        检查新概念是否与已有概念相似
        Returns:
            如果相似，返回相似的已有概念；否则返回None
        """
        if not self.concept_embeddings:
            return None
    
        # 计算与所有已有概念的相似度
        existing_concepts = list(self.concept_embeddings.keys())
        existing_embeddings = np.array([self.concept_embeddings[concept] for concept in existing_concepts])
  
        # 计算cosine相似度
        similarities = self.model.similarity([new_embedding], existing_embeddings)[0]
  
        # 找到最相似的概念
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
  
        if max_similarity >= self.similarity_threshold:
            return existing_concepts[max_similarity_idx]
  
        return None
  
    def get_current_adjacency(self) -> Dict[str, List[str]]:
        """获取当前邻接字典"""
        return self.graph.copy()
  
    def calculate_metrics(self, expansion_results: Dict) -> Dict[str, float]:
        """计算扩增指标"""
        # 收集所有新概念
        all_new_concepts = []
        for result in expansion_results.values():
            if result.status == "success" and result.new_concepts:
                all_new_concepts.extend(result.new_concepts)
  
        if not all_new_concepts:
            return {"connectivity_rate": 0.0}
  
        existing_concepts = set(self.graph.keys())
  
        # 连通度：新生成的边（彼此都是旧节点）/ 前一轮已有总边数
        old_total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
  
        # 统计新生成的边（彼此都是旧节点）
        new_edges_between_old_nodes = 0
        for result in expansion_results.values():
            if result.status == "success" and result.new_concepts:
                center_concept = result.center_concept
                for new_concept in result.new_concepts:
                    if (new_concept in existing_concepts and 
                        center_concept != new_concept and
                        new_concept not in self.graph.get(center_concept, [])):
                        new_edges_between_old_nodes += 1
  
        connectivity_rate = new_edges_between_old_nodes / old_total_edges if old_total_edges > 0 else float('inf') 
  
        return {
            "connectivity_rate": connectivity_rate
        }
  
  
    def update_graph(self, expansion_results: Dict):
        """
        更新图结构 - 使用mapping机制进行去重
        """
        nodes_added = 0
        edges_added = 0
        embedding_duplicates = 0
  
        # 收集所有新概念
        all_new_concepts = []
        concept_to_centers = {}  # 记录每个新概念对应的中心概念
  
        for result in expansion_results.values():
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
            return nodes_added, edges_added, embedding_duplicates
  
        num_all_new_concepts = len(all_new_concepts)
        print(f"收到 {num_all_new_concepts} 个新概念（未去重）")
  
        # 使用mapping快速过滤已知概念
        concepts_need_embedding = []
        concept_targets = {}  # concept -> target_concept 映射
  
        for concept in all_new_concepts:
            target = self._get_target_concept(concept)
            if target is not None:
                # 已知概念，直接使用映射
                concept_targets[concept] = target
                if target != concept:
                    embedding_duplicates += 1
            else:
                # 需要计算embedding的新概念
                concepts_need_embedding.append(concept)
  
        # 只对未知概念计算embedding
        if concepts_need_embedding:
            unique_concepts = list(set(concepts_need_embedding))
            print(f"正在对 {len(unique_concepts)} 个新概念进行embedding去重...")
            new_embeddings = self.model.encode(unique_concepts)
    
            # 逐个处理需要embedding的概念
            total_concepts = len(unique_concepts)
            for idx, (new_concept, new_embedding) in enumerate(zip(unique_concepts, new_embeddings), 1):
                # 每500个输出一次进度
                if idx % 500 == 0 or idx == total_concepts:
                    print(f"  处理进度: {idx}/{total_concepts} ({idx/total_concepts*100:.1f}%)")
                # 检查是否与已有概念相似
                similar_concept = self._is_similar_to_existing(new_concept, new_embedding)
        
                if similar_concept:
                    # 发现相似概念，建立映射
                    self.concept_mapping[new_concept] = similar_concept
                    concept_targets[new_concept] = similar_concept
                    embedding_duplicates += 1
                else:
                    # 全新概念，添加到图中并建立自映射
                    self.graph[new_concept] = []
                    self.concept_embeddings[new_concept] = new_embedding
                    self.concept_mapping[new_concept] = new_concept
                    concept_targets[new_concept] = new_concept
                    nodes_added += 1
  
        # 添加边（连接到所有相关的中心概念）
        for concept in all_new_concepts:
            target_concept = concept_targets[concept]
    
            for center_concept in concept_to_centers[concept]:
                # 确保中心概念存在于图中
                if center_concept in self.graph:
                    # 双向连接
                    if target_concept not in self.graph[center_concept]:
                        self.graph[center_concept].append(target_concept)
                        edges_added += 1
            
                    if center_concept not in self.graph[target_concept]:
                        self.graph[target_concept].append(center_concept)
                        edges_added += 1
  
        print(f"去重完成: 新增节点 {nodes_added}, 新增边 {edges_added//2}, 去重概念 {embedding_duplicates}")
  
        return nodes_added, edges_added // 2, embedding_duplicates, nodes_added / num_all_new_concepts  # 无向图，边数除以2
  
    def get_graph_stats(self) -> Dict[str, int]:
        """获取图统计信息"""
        node_count = len(self.graph)
        edge_count = sum(len(neighbors) for neighbors in self.graph.values()) // 2
        return {"nodes": node_count, "edges": edge_count}
```

#### **2. 完整扩增流程代码**

完整的迭代流程由一个可执行的“单轮概念扩增”流水线构成，它整合了多个组件：

* **实用工具 (`ResponseValidator`, `load_embedding_model`)**: 用于处理工程实践中的常见问题，如修复LLM返回的不规范JSON、加载和管理深度学习模型。
* **LLM交互模块 (`APIClient`, `PromptTemplate`)**: 专门设计的 `PromptTemplate`用于引导LLM基于图中的现有概念进行“联想”和“扩增”。
* **并发处理器 (`ConceptExpander`, `BatchConceptExpander`)**: 负责将图中的每一个概念节点作为独立的任务，并行地向LLM请求扩增，以保证处理效率。
* **迭代总控 (`run_concept_expansion_iteration`)**: 这是一个顶层函数，负责编排以上所有组件，完整地执行一轮“获取当前图 -\> 并发扩增 -\> 更新图 -\> 计算指标”的循环。

```python

# JSON修复库导入
try:
    from json_repair import repair_json
    HAS_JSONREPAIR = True
    print("✓ jsonrepair库已加载，JSON修复功能已启用")
except ImportError:
    HAS_JSONREPAIR = False
    print("⚠ jsonrepair库未安装，将使用基础修复策略。运行 'pip install jsonrepair' 启用高级JSON修复")
    def repair_json(text):
        return text

class ResponseValidator:
    """响应验证器"""
  
    @staticmethod
    def validate_json_response(response_text: str, expected_keys: List[str]) -> Dict:
        """
        验证API返回的内容是否为有效JSON，包含预处理容错
  
        Returns:
            dict: {
                "is_valid_json": bool,
                "parsed_json": dict or None,
                "error_type": str,
                "raw_response": str
            }
        """
        if not response_text or not response_text.strip():
            return {
                "is_valid_json": False,
                "parsed_json": None,
                "error_type": "empty_response",
                "raw_response": response_text
            }
  
        repair_attempts = []
  
        try:
            # 预处理清理
            text = response_text.strip()
    
            # 1. 处理markdown代码块 ```json...``` 或 ```...```
            code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if code_block_match:
                text = code_block_match.group(1).strip()
    
            # 2. 处理引号包装 '...' 或 "..."
            if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
                text = text[1:-1]
    
            # 3. 移除前后的反引号
            text = text.strip('`').strip()
    
            # 4. 查找JSON部分 - 第一个{到最后一个}
            json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
    
            # 第一次尝试：直接解析
            try:
                parsed = json.loads(text)
                repair_attempts.append("direct_parse_success")
            except json.JSONDecodeError as e:
                repair_attempts.append(f"direct_parse_failed: {str(e)}")
        
                # 第二次尝试：使用jsonrepair修复后解析
                if HAS_JSONREPAIR:
                    try:
                        repaired_text = repair_json(text)
                        parsed = json.loads(repaired_text)
                        repair_attempts.append("jsonrepair_success")
                    except Exception as e:
                        repair_attempts.append(f"jsonrepair_failed: {str(e)}")
                        raise
                else:
                    repair_attempts.append("jsonrepair_not_available")
                    raise
    
            # 检查是否符合预期的结构
            if isinstance(parsed, dict) and all(key in parsed for key in expected_keys):
                return {
                    "is_valid_json": True,
                    "parsed_json": parsed,
                    "error_type": None,
                    "raw_response": response_text,
                    "repair_attempts": repair_attempts
                }
            else:
                missing_keys = [key for key in expected_keys if key not in parsed] if isinstance(parsed, dict) else expected_keys
                return {
                    "is_valid_json": False,
                    "parsed_json": parsed,
                    "error_type": f"missing_keys: expected {expected_keys}, missing {missing_keys}",
                    "raw_response": response_text,
                    "repair_attempts": repair_attempts
                }
        
        except json.JSONDecodeError as e:
            return {
                "is_valid_json": False,
                "parsed_json": None,
                "error_type": f"json_decode_error: {str(e)}",
                "raw_response": response_text,
                "repair_attempts": repair_attempts
            }
        except Exception as e:
            return {
                "is_valid_json": False,
                "parsed_json": None,
                "error_type": f"unexpected_error: {str(e)}",
                "raw_response": response_text,
                "repair_attempts": repair_attempts
            }

# 基于种子概念的图扩增迭代代码

@dataclass
class ConceptExpansionResult:
    """概念扩增结果数据类"""
    status: str  # success, api_error, json_error
    concept_id: str
    center_concept: str
    neighbors: List[str]
    response: Optional[str] = None
    error_details: Optional[str] = None
    processing_time: Optional[float] = None
    json_validation: Optional[Dict] = None
    new_concepts: Optional[List[str]] = None
    returned_center: Optional[str] = None  # LLM返回的center_concept

class PromptTemplate:
    """提示词模板类"""

    @staticmethod
    def get_system_prompt() -> str:
        """获取系统提示词"""
        return """You are an experienced cardiovascular specialist, skilled in building comprehensive concept graphs for the cardiovascular domain.

Your task is to expand a cardiovascular concept graph by generating new related concepts based on a given center concept and its existing connections."""

    @staticmethod
    def get_expansion_prompt(center_concept: str, neighbors: List[str]) -> str:
        """生成概念扩增提示词"""
        neighbors_text = ", ".join(neighbors) if neighbors else "None"
  
        return f"""**Task: Generate new cardiovascular concepts related to the center concept**

**Domain**: Cardiovascular medicine
**Relationship requirement**: New concepts should be directly related to the center concept through strong clinical medical associations

**Center concept**: {center_concept}
**Existing neighbor concepts of the center concept**: {neighbors_text}

**Output format (strictly follow JSON format):**

{{
  "center_concept": "{center_concept}",
  "new_concepts": [
    "concept1",
    "concept2",
    "concept3",
    "..."
  ]
}}


If no new concepts can be generated:

{{
  "center_concept": "{center_concept}",
  "new_concepts": ["NO NEW CONCEPTS"]
}}

**Instructions**:

1. Instead of generate general medical concept, focus on generating new cardiovascular-domain concepts that are directly relevant in clinical scenarios to "{center_concept}" with strong clinical medical relation
2. Do not repeat any existing connected concepts listed above
3. Prefer single core terms (e.g., "myocardial infarction", "hypertension", "echocardiography")
4. Use phrases only when they represent standard medical terminology that cannot be meaningfully separated (e.g., "atrial fibrillation", "coronary artery disease")
5. Avoid descriptive combinations (e.g., "severe hypertension" → "hypertension")
6. Avoid overly vague terms (e.g., "heart problem")
7. Generate concepts that are directly related to the center concept
8. Do not repeat any existing connected concepts listed above; Avoid duplicate concepts"""

class ConceptExpander:
    """概念扩增处理器"""

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.template = PromptTemplate()

    def expand_single_concept(self, center_concept: str, neighbors: List[str], concept_id: str) -> ConceptExpansionResult:
        """扩增单个概念"""
        start_time = time.time()
  
        # 构建消息
        system_prompt = self.template.get_system_prompt()
        user_prompt = self.template.get_expansion_prompt(center_concept, neighbors)
  
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 调用API
        api_result = self.api_client.call_api(messages, timeout=120)
        processing_time = time.time() - start_time

        if api_result["status"] != "success":
            return ConceptExpansionResult(
                status="api_error",
                concept_id=concept_id,
                center_concept=center_concept,
                neighbors=neighbors,
                error_details=api_result["error"],
                processing_time=processing_time
            )

        # 验证JSON响应
        expected_keys = ["center_concept", "new_concepts"]
        json_validation = ResponseValidator.validate_json_response(
            api_result["content"], expected_keys
        )
  
        if json_validation["is_valid_json"]:
            returned_center = json_validation["parsed_json"]["center_concept"]
            new_concepts = json_validation["parsed_json"]["new_concepts"]
    
            # 检查是否为"无新增概念"的情况
            if len(new_concepts) == 1 and new_concepts[0].strip() == "NO NEW CONCEPTS":
                return ConceptExpansionResult(
                    status="success",
                    concept_id=concept_id,
                    center_concept=center_concept,
                    neighbors=neighbors,
                    response=api_result["content"],
                    processing_time=processing_time,
                    json_validation=json_validation,
                    new_concepts=[],  # 空列表，表示没有新概念
                    returned_center=returned_center
                )
    
            # 正常处理新概念 - 移除之前的过滤逻辑，交给embedding去重
            new_concepts = [concept.strip() for concept in new_concepts if concept.strip()]
    
            return ConceptExpansionResult(
                status="success",
                concept_id=concept_id,
                center_concept=center_concept,
                neighbors=neighbors,
                response=api_result["content"],
                processing_time=processing_time,
                json_validation=json_validation,
                new_concepts=new_concepts,
                returned_center=returned_center
            )
        else:
            return ConceptExpansionResult(
                status="json_error",
                concept_id=concept_id,
                center_concept=center_concept,
                neighbors=neighbors,
                response=api_result["content"],
                error_details=f"JSON validation failed: {json_validation['error_type']}",
                processing_time=processing_time,
                json_validation=json_validation
            )

class BatchConceptExpander:
    """批量概念扩增处理器"""
    def __init__(self, expander: ConceptExpander, output_dir: str = "concept_expansion"):
        self.expander = expander
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def expand_concepts_batch(self, adjacency_dict: Dict[str, List[str]], max_workers: int = 10) -> Dict:
        """批量扩增概念"""
        concepts_to_expand = list(adjacency_dict.keys())
        total_concepts = len(concepts_to_expand)
        print(f"开始批量概念扩增 {total_concepts} 个概念")
        print(f"最大并发: {max_workers}")

        batch_start_time = time.time()
        batch_results = self._process_batch(concepts_to_expand, adjacency_dict, max_workers)
        batch_end_time = time.time()

        print(f"处理完成，耗时: {batch_end_time - batch_start_time:.2f} 秒")
        return batch_results

    def _process_batch(self, batch_concepts: List[str], adjacency_dict: Dict[str, List[str]], 
                      max_workers: int) -> Dict:
        """处理批次"""
        batch_results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_concept = {}
            for idx, concept in enumerate(batch_concepts):
                concept_id = f"concept_{idx:06d}"
                neighbors = adjacency_dict.get(concept, [])
                future = executor.submit(self.expander.expand_single_concept, concept, neighbors, concept_id)
                future_to_concept[future] = (concept_id, concept)

            # 收集结果
            completed = 0
            for future in as_completed(future_to_concept):
                concept_id, concept = future_to_concept[future]
                try:
                    result = future.result()
                    batch_results[concept_id] = result

                    # 简单状态显示
                    status_symbol = "✓" if result.status == "success" else "✗"
                    completed += 1
            
                    if completed % 1000 == 0 or completed == len(batch_concepts):
                        success_count = sum(1 for r in batch_results.values() if r.status == "success")
                        print(f"  已完成: {completed}/{len(batch_concepts)} (成功: {success_count}) {status_symbol}")

                except Exception as e:
                    batch_results[concept_id] = ConceptExpansionResult(
                        status="exception",
                        concept_id=concept_id,
                        center_concept=concept,
                        neighbors=adjacency_dict.get(concept, []),
                        error_details=str(e)
                    )
                    print(f"  异常: {concept_id} - {str(e)}")

        return batch_results

    def _save_results(self, batch_results: Dict, start_time: float):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"concept_expansion_results_{timestamp}.pkl"
        filepath = os.path.join(self.output_dir, filename)

        # 统计信息
        total_count = len(batch_results)
        success_count = sum(1 for r in batch_results.values() if r.status == "success")
        error_count = total_count - success_count

        # 统计生成的概念数量
        total_new_concepts = sum(len(r.new_concepts) for r in batch_results.values() 
                               if r.status == "success" and r.new_concepts)
  
        # 统计跳过的概念数量（无新增概念）
        skipped_concepts = sum(1 for r in batch_results.values() 
                              if r.status == "success" and r.new_concepts is not None and len(r.new_concepts) == 0)

        # 构建保存数据
        save_data = {
            "metadata": {
                "timestamp": timestamp,
                "start_time": start_time,
                "total_concepts": total_count,
                "successful_expansions": success_count,
                "failed_expansions": error_count,
                "success_rate": success_count / total_count if total_count > 0 else 0,
                "total_new_concepts": total_new_concepts,
                "skipped_concepts": skipped_concepts
            },
            "results": batch_results
        }

        # 保存到文件
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"  结果已保存: {filename}")
        print(f"  成功: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        print(f"  新概念总数: {total_new_concepts}")
        print(f"  跳过概念: {skipped_concepts}")

        return filepath

class ConceptGraph:
  
    @classmethod
    def from_graph_dict(cls, graph_dict: Dict[str, List[str]], concept_mapping, model, similarity_threshold):
        """
        从保存的图字典重新构建ConceptGraph
        Args:
            graph_dict: 保存的邻接字典
            model: SentenceTransformer模型实例
            similarity_threshold：判为相似概念的threshold
            concept_mapping: 保存的所有见过的相似概念mapping，例如{'hearts':'heart'}
        Returns:
            ConceptGraph实例
        """
        if model is None:
            raise ValueError("模型参数不能为None，请先使用load_embedding_model()加载模型")
  
        # 创建实例但不初始化
        instance = cls.__new__(cls)
        instance.model = model
        instance.graph = graph_dict.copy()
        instance.concept_embeddings = {}
        instance.concept_mapping = concept_mapping  
        instance.similarity_threshold = similarity_threshold
  
        # 重新计算所有概念的embedding并建立自映射
        all_concepts = list(graph_dict.keys())
        if all_concepts:
            print(f"正在为 {len(all_concepts)} 个概念重新计算embedding...")
            all_embeddings = model.encode(all_concepts)
    
            for concept, embedding in zip(all_concepts, all_embeddings):
                instance.concept_embeddings[concept] = embedding
                #instance.concept_mapping[concept] = concept  # 建立自映射
    
            print("ConceptGraph重建完成")
  
        return instance
  
    def __init__(self, seed_concepts: List[str], model, similarity_threshold):
        """
        从种子概念初始化图，并建立embedding库
        Args:
            seed_concepts: 已经过外部去重的种子概念列表
            model: SentenceTransformer模型实例（必需）
            similarity_threshold：判为相似概念的threshold
        """
        if model is None:
            raise ValueError("模型参数不能为None，请先使用load_embedding_model()加载模型")
    
        self.model = model
        self.graph = {}
        self.concept_embeddings = {}  # 维护concept -> embedding的映射
        self.concept_mapping = {}     # 添加概念映射表
        self.similarity_threshold = similarity_threshold  # embedding相似度阈值
  
        # 清理种子概念
        cleaned_seeds = [concept.strip() for concept in seed_concepts if concept.strip()]
  
        print(f"正在为 {len(cleaned_seeds)} 个种子概念计算embedding...")
  
        # 批量计算embedding
        if cleaned_seeds:
            seed_embeddings = self.model.encode(cleaned_seeds)
    
            # 建立初始图、embedding库和映射表
            for concept, embedding in zip(cleaned_seeds, seed_embeddings):
                self.graph[concept] = []
                self.concept_embeddings[concept] = embedding
                self.concept_mapping[concept] = concept  # 建立自映射
  
        print(f"初始化概念图完成，种子概念数: {len(cleaned_seeds)}")
  
    def _get_target_concept(self, concept: str) -> Optional[str]:
        """统一的概念映射查找"""
        return self.concept_mapping.get(concept)
  
    def _is_similar_to_existing(self, new_concept: str, new_embedding: np.ndarray) -> Optional[str]:
        """
        检查新概念是否与已有概念相似
        Returns:
            如果相似，返回相似的已有概念；否则返回None
        """
        if not self.concept_embeddings:
            return None
    
        # 计算与所有已有概念的相似度
        existing_concepts = list(self.concept_embeddings.keys())
        existing_embeddings = np.array([self.concept_embeddings[concept] for concept in existing_concepts])
  
        # 计算cosine相似度
        similarities = self.model.similarity([new_embedding], existing_embeddings)[0]
  
        # 找到最相似的概念
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
  
        if max_similarity >= self.similarity_threshold:
            return existing_concepts[max_similarity_idx]
  
        return None
  
    def get_current_adjacency(self) -> Dict[str, List[str]]:
        """获取当前邻接字典"""
        return self.graph.copy()
  
    def calculate_metrics(self, expansion_results: Dict) -> Dict[str, float]:
        """计算扩增指标"""
        # 收集所有新概念
        all_new_concepts = []
        for result in expansion_results.values():
            if result.status == "success" and result.new_concepts:
                all_new_concepts.extend(result.new_concepts)
  
        if not all_new_concepts:
            return {"connectivity_rate": 0.0}
  
        existing_concepts = set(self.graph.keys())
  
        # 连通度：新生成的边（彼此都是旧节点）/ 前一轮已有总边数
        old_total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
  
        # 统计新生成的边（彼此都是旧节点）
        new_edges_between_old_nodes = 0
        for result in expansion_results.values():
            if result.status == "success" and result.new_concepts:
                center_concept = result.center_concept
                for new_concept in result.new_concepts:
                    if (new_concept in existing_concepts and 
                        center_concept != new_concept and
                        new_concept not in self.graph.get(center_concept, [])):
                        new_edges_between_old_nodes += 1
  
        connectivity_rate = new_edges_between_old_nodes / old_total_edges if old_total_edges > 0 else float('inf') 
  
        return {
            "connectivity_rate": connectivity_rate
        }
  
  
    def update_graph(self, expansion_results: Dict):
        """
        更新图结构 - 使用mapping机制进行去重
        """
        nodes_added = 0
        edges_added = 0
        embedding_duplicates = 0
  
        # 收集所有新概念
        all_new_concepts = []
        concept_to_centers = {}  # 记录每个新概念对应的中心概念
  
        for result in expansion_results.values():
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
            return nodes_added, edges_added, embedding_duplicates
  
        num_all_new_concepts = len(all_new_concepts)
        print(f"收到 {num_all_new_concepts} 个新概念（未去重）")
  
        # 使用mapping快速过滤已知概念
        concepts_need_embedding = []
        concept_targets = {}  # concept -> target_concept 映射
  
        for concept in all_new_concepts:
            target = self._get_target_concept(concept)
            if target is not None:
                # 已知概念，直接使用映射
                concept_targets[concept] = target
                if target != concept:
                    embedding_duplicates += 1
            else:
                # 需要计算embedding的新概念
                concepts_need_embedding.append(concept)
  
        # 只对未知概念计算embedding
        if concepts_need_embedding:
            unique_concepts = list(set(concepts_need_embedding))
            print(f"正在对 {len(unique_concepts)} 个新概念进行embedding去重...")
            new_embeddings = self.model.encode(unique_concepts)
    
            # 逐个处理需要embedding的概念
            total_concepts = len(unique_concepts)
            for idx, (new_concept, new_embedding) in enumerate(zip(unique_concepts, new_embeddings), 1):
                # 每500个输出一次进度
                if idx % 500 == 0 or idx == total_concepts:
                    print(f"  处理进度: {idx}/{total_concepts} ({idx/total_concepts*100:.1f}%)")
                # 检查是否与已有概念相似
                similar_concept = self._is_similar_to_existing(new_concept, new_embedding)
        
                if similar_concept:
                    # 发现相似概念，建立映射
                    self.concept_mapping[new_concept] = similar_concept
                    concept_targets[new_concept] = similar_concept
                    embedding_duplicates += 1
                else:
                    # 全新概念，添加到图中并建立自映射
                    self.graph[new_concept] = []
                    self.concept_embeddings[new_concept] = new_embedding
                    self.concept_mapping[new_concept] = new_concept
                    concept_targets[new_concept] = new_concept
                    nodes_added += 1
  
        # 添加边（连接到所有相关的中心概念）
        for concept in all_new_concepts:
            target_concept = concept_targets[concept]
    
            for center_concept in concept_to_centers[concept]:
                # 确保中心概念存在于图中
                if center_concept in self.graph:
                    # 双向连接
                    if target_concept not in self.graph[center_concept]:
                        self.graph[center_concept].append(target_concept)
                        edges_added += 1
            
                    if center_concept not in self.graph[target_concept]:
                        self.graph[target_concept].append(center_concept)
                        edges_added += 1
  
        print(f"去重完成: 新增节点 {nodes_added}, 新增边 {edges_added//2}, 去重概念 {embedding_duplicates}")
  
        return nodes_added, edges_added // 2, embedding_duplicates, nodes_added / num_all_new_concepts  # 无向图，边数除以2
  
    def get_graph_stats(self) -> Dict[str, int]:
        """获取图统计信息"""
        node_count = len(self.graph)
        edge_count = sum(len(neighbors) for neighbors in self.graph.values()) // 2
        return {"nodes": node_count, "edges": edge_count}

def load_embedding_model(model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
    """
    加载embedding模型
    Args:
        model_name: 模型名称
    Returns:
        SentenceTransformer模型实例
    """
    print(f"正在加载embedding模型: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print("模型加载完成")
    return model

def extract_seed_concepts(results):
    """从批量处理结果中提取种子概念"""
    all_concepts = []
  
    for result in results.values():
        if result.status == "success" and result.extracted_concepts:
            all_concepts.extend(result.extracted_concepts)
  
    # 去空格并去重
    seed_concepts = list(set(concept.strip() for concept in all_concepts if concept.strip()))
  
    return seed_concepts

def run_concept_expansion_iteration(api_url: str, api_key: str, model_name: str, concept_graph: ConceptGraph, max_workers: int = 10):
    """运行单次概念扩增迭代"""
    # 初始化API客户端和处理器
    api_client = APIClient(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name
    )
  
    expander = ConceptExpander(api_client)
    batch_expander = BatchConceptExpander(expander)
  
    # 获取当前邻接字典
    current_adjacency = concept_graph.get_current_adjacency()
  
    # 批量扩增概念
    expansion_results = batch_expander.expand_concepts_batch(
        adjacency_dict=current_adjacency,
        max_workers=max_workers
    )
  
    # 计算指标
    metrics = concept_graph.calculate_metrics(expansion_results)
  
    # 更新图 - 使用embedding去重
    nodes_added, edges_added, embedding_duplicates, concept_add_rate = concept_graph.update_graph(expansion_results)

    # 获取更新后的图统计
    graph_stats = concept_graph.get_graph_stats()
  
    # 统计跳过的概念数量
    skipped_count = sum(1 for r in expansion_results.values() 
                       if r.status == "success" and r.new_concepts is not None and len(r.new_concepts) == 0)
  
    # 打印结果
    print(f"\n=== 迭代完成 ===")
    print(f"概念更新度: {concept_add_rate:.3f}")
    print(f"概念连通度: {metrics['connectivity_rate']:.3f}")
    print(f"迭代结束节点数: {graph_stats['nodes']}")
    print(f"迭代结束边数: {graph_stats['edges']}")
    print(f"本轮新增节点: {nodes_added}")
    print(f"本轮新增边数: {edges_added}")
    print(f"跳过的概念: {skipped_count}")
  
    return {
        "concept_add_rate": concept_add_rate,
        "connectivity_rate": metrics['connectivity_rate'],
        "graph_stats": graph_stats,
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "embedding_duplicates": embedding_duplicates,
        "skipped_count": skipped_count,
        "expansion_results": expansion_results
    }
```

### **启动扩增：准备与执行**

在启动大规模迭代之前，我们对初始的 `seed_concepts`列表进行一次内部的语义去重。`deduplicate_seed_concepts`函数会对所有种子概念进行两两语义相似度比较，并丢弃相似度过高的重复概念，以保证初始图谱的纯净性。

完成清洗后，我们使用这份高质量的种子集正式创建 `ConceptGraph`实例，准备进行第一轮扩增。

载入embedding model用于过滤相似的seed concept：

```python
import json
import pickle
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"初始化模型，使用设备: {device}")
model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device=device)
print("模型初始化完成")
```

根据seed concepts初始化concept_graph：

```python

import random
import numpy as np

def deduplicate_seed_concepts(seed_concepts, model, similarity_threshold=0.95):
    """对种子概念进行去重，相似度>阈值的概念对随机保留一个"""
  
    embeddings = model.encode(seed_concepts)
    similarities = model.similarity(embeddings, embeddings)
    similarities = similarities.cpu().numpy()
  
    to_remove = set()
    n = len(seed_concepts)
  
    for i in range(n):
        for j in range(i+1, n):
            if similarities[i][j] > similarity_threshold:
                # 相似度超过阈值，随机选择一个删除
                remove_idx = random.choice([i, j])
                to_remove.add(remove_idx)
                print(f"相似概念对: '{seed_concepts[i]}' vs '{seed_concepts[j]}' (相似度: {similarities[i][j]:.4f})")
                print(f"  -> 删除: '{seed_concepts[remove_idx]}'")
  
    filtered_concepts = [concept for i, concept in enumerate(seed_concepts) if i not in to_remove]
  
    print(f"\n去重结果: {len(seed_concepts)} -> {len(filtered_concepts)} 个概念")
    print(f"删除了 {len(to_remove)} 个相似概念")
  
    return filtered_concepts

# 使用示例
filtered_seed_concepts = deduplicate_seed_concepts(seed_concepts, model, similarity_threshold=0.8)
concept_graph = ConceptGraph(filtered_seed_concepts, model, similarity_threshold=0.8)
```

通过重复执行 `run_concept_expansion_iteration`函数，我们可以完成多轮扩增。在每一轮扩增结束后，我们建议将更新后的图结构（`graph_dict`）和概念映射表（`concept_mapping`）持久化存储为 `.pkl`文件，以防长时间运行的成果因意外中断而丢失。这个过程将持续进行，直到预设的收敛指标达到阈值。

## 图的具体单轮扩增过程参考

```python
# 进行一轮迭代的代码示例

domain = 'Cardio'

# 确保保存目录存在
save_dir = f'cookbooktest/{domain}'
os.makedirs(save_dir, exist_ok=True)

iter_n = 1

results = run_concept_expansion_iteration(model_name=CONCEPT_GRAPH_EXTENDER, concept_graph=concept_graph, max_workers = 100,
                                          api_url=api_url,
                                          api_key=api_key)



# 获取图的邻接表
graph_dict = concept_graph.graph

# 保存为pickle
with open(f'{save_dir}/concept_graph_4omini_{iter_n}_iter.pkl', 'wb') as f:
    pickle.dump(graph_dict, f)

# 获取图的邻接表
concept_mapping = concept_graph.concept_mapping
# 保存为pickle
with open(f'{save_dir}/concept_graph_4omini_{iter_n}_iter_concept_mapping.pkl', 'wb') as f:
    pickle.dump(concept_mapping, f)


# 重复以进行更多轮次···实践经验10次以内应当图规模已经很大、足以收敛。
# 循环代码，仅供参考
# 由于图扩增成本随着轮数增加呈现成本增加的趋势，我们强烈建议您手动逐轮进行迭代，并在每一轮迭代结束后基于您自己业务构建的用于validation的concept库检查当前concept graph的覆盖率以确定终止轮数
# 在没有validation concept库的情况下，您可参考(a)新增概念覆盖度=本轮新增节点/本轮收到的新概念数(b)概念连通度作为收敛终止参考指标
'''
MAX_ITER = 10             # 最大轮数
CONCEPT_ADD_THRESHOLD = 0.05   # 概念新增率下限
CONNECTIVITY_THRESHOLD = 0.2  # 连接率下限

domain = 'Cardio'
save_dir = f'cookbooktest/{domain}'

for iter_n in range(1, MAX_ITER + 1):
    print(f"\n===== Iteration {iter_n} =====")
  
    # 运行一轮概念扩展
    results = run_concept_expansion_iteration(
        model_name=CONCEPT_GRAPH_EXTENDER,
        concept_graph=concept_graph,
        max_workers=100,
        api_url=api_url,
        api_key=api_key
    )

    # 提取指标
    concept_add_rate = results["concept_add_rate"]
    connectivity_rate = results["connectivity_rate"]

    # 保存图邻接表
    graph_dict = concept_graph.graph
    with open(f'{save_dir}/concept_graph_4omini_{iter_n}_iter.pkl', 'wb') as f:
        pickle.dump(graph_dict, f)

    # 保存概念映射
    concept_mapping = concept_graph.concept_mapping
    with open(f'{save_dir}/concept_graph_4omini_{iter_n}_iter_concept_mapping.pkl', 'wb') as f:
        pickle.dump(concept_mapping, f)

    # OR 条件提前终止
    if (concept_add_rate < CONCEPT_ADD_THRESHOLD) or (connectivity_rate < CONNECTIVITY_THRESHOLD):
        print(f"停止迭代：满足停止条件（concept_add_rate<{CONCEPT_ADD_THRESHOLD} 或 connectivity_rate<{CONNECTIVITY_THRESHOLD}）")
        break
'''
```

### **实验数据与评估**

我们在心血管（cardio）和呼吸系统（respiratory）两个医学领域进行了全面的实验验证，使用GPT-4o-mini和GPT-4o两种模型，在不同相似度阈值（0.8, 0.85, 0.9）下进行多轮迭代扩增。验证方式为从维基百科对应领域的医学文章中抽取关键词，计算概念图谱的覆盖率。

#### **实验设置**

- **数据集**: 心血管领域（cardio）、呼吸系统领域（respiratory）
- **模型**: GPT-4o-mini、GPT-4o
- **相似度阈值**: 0.8, 0.85, 0.9
- **评估指标**: 维基百科关键词覆盖率（在0.8, 0.85, 0.9三个阈值下）
- **成本基准**: GPT-4o-mini完成8轮迭代约$20

#### **指标说明**

- **节点数**: 所有见过的概念总数（包括判为相似的概念+图中保留的概念）
- **边数**: 概念图中的边总数
- **新概念率**: 概念图内的新增节点数/全部当轮产生的概念数
- **新边率**: 旧节点间新增的边数/旧节点间新增边前的边数
- **成本估计**：计算当轮和此前轮次产生的概念总数，按8轮迭代的gpt-4o-mini费率估算

#### **心血管领域实验结果**

| 模型        | 阈值 | 轮次 | 节点数 | 边数    | 覆盖率@0.8 | 覆盖率@0.85 | 覆盖率@0.9 | 新概念率 | 新边率 | 成本估计 |
| ----------- | ---- | ---- | ------ | ------- | ---------- | ----------- | ---------- | -------- | ------ | -------- |
| GPT-4o-mini | 0.8  | 1    | 7,797  | 23,284  | 89.67%     | 79.73%      | 66.12%     | 10.05%   | 0.00%  | $0.83    |
| GPT-4o-mini | 0.8  | 2    | 16,066 | 63,712  | 93.79%     | 86.13%      | 76.00%     | 7.20%    | 49.50% | $2.37    |
| GPT-4o-mini | 0.8  | 3    | 28,737 | 132,451 | 95.49%     | 89.67%      | 80.58%     | 5.50%    | 33.40% | $5.06    |
| GPT-4o-mini | 0.8  | 4    | 47,255 | 239,751 | 96.21%     | 92.09%      | 83.52%     | 4.56%    | 25.60% | $9.32    |
| GPT-4o-mini | 0.8  | 5    | 73,359 | 397,872 | 96.73%     | 93.79%      | 86.46%     | 3.79%    | 21.10% | $15.68   |
| GPT-4o      | 0.8  | 1    | 7,232  | 20,577  | 90.91%     | 80.97%      | 68.61%     | 9.68%    | 0.00%  | $7.34    |
| GPT-4o      | 0.8  | 2    | 14,574 | 55,559  | 94.57%     | 88.82%      | 78.68%     | 7.51%    | 46.20% | $20.41   |
| GPT-4o      | 0.8  | 3    | 23,488 | 102,429 | 95.95%     | 91.50%      | 82.01%     | 6.15%    | 23.80% | $37.94   |
| GPT-4o      | 0.8  | 4    | 40,030 | 188,292 | 97.32%     | 94.31%      | 86.20%     | 5.30%    | 23.50% | $70.82   |
| GPT-4o      | 0.8  | 5    | 64,807 | 317,772 | 97.97%     | 95.36%      | 88.82%     | 4.45%    | 18.80% | $120.80  |
| GPT-4o      | 0.85 | 1    | 8,750  | 29,200  | 92.22%     | 84.43%      | 72.73%     | 9.96%    | 0.00%  | $10.34   |
| GPT-4o      | 0.85 | 2    | 17,570 | 79,784  | 95.88%     | 90.97%      | 81.56%     | 9.11%    | 70.90% | $28.58   |
| GPT-4o      | 0.85 | 3    | 32,217 | 172,894 | 96.99%     | 93.66%      | 86.46%     | 7.36%    | 49.40% | $62.47   |
| GPT-4o      | 0.85 | 4    | 55,586 | 328,126 | 97.78%     | 95.29%      | 89.60%     | 6.31%    | 37.90% | $119.19  |

#### **呼吸系统领域实验结果**

| 模型        | 阈值 | 轮次 | 节点数  | 边数    | 覆盖率@0.8 | 覆盖率@0.85 | 覆盖率@0.9 | 新概念率 | 新边率  | 成本估计 |
| ----------- | ---- | ---- | ------- | ------- | ---------- | ----------- | ---------- | -------- | ------- | -------- |
| GPT-4o-mini | 0.8  | 1    | 4,683   | 13,569  | 84.07%     | 74.32%      | 61.11%     | 11.27%   | 0.00%   | $0.49    |
| GPT-4o-mini | 0.8  | 2    | 9,736   | 38,374  | 89.20%     | 82.08%      | 69.71%     | 7.92%    | 56.00%  | $1.42    |
| GPT-4o-mini | 0.8  | 3    | 17,893  | 81,567  | 92.56%     | 87.21%      | 76.10%     | 6.10%    | 36.70%  | $3.11    |
| GPT-4o-mini | 0.8  | 4    | 30,231  | 150,979 | 94.55%     | 89.62%      | 79.98%     | 4.96%    | 28.30%  | $2.76    |
| GPT-4o-mini | 0.8  | 5    | 48,032  | 255,740 | 96.23%     | 91.40%      | 82.08%     | 4.08%    | 23.20%  | $6.96    |
| GPT-4o-mini | 0.8  | 6    | 73,308  | 405,320 | 96.96%     | 92.87%      | 84.38%     | 3.69%    | 19.20%  | $13.02   |
| GPT-4o-mini | 0.8  | 7    | 108,026 | 613,808 | 97.27%     | 93.92%      | 87.42%     | 3.23%    | 16.60%  | $8.49    |
| GPT-4o-mini | 0.8  | 8    | 153,963 | 894,902 | 97.80%     | 94.76%      | 89.10%     | 2.88%    | 14.50%  | $20.00   |
| GPT-4o-mini | 0.85 | 1    | 5,334   | 17,158  | 86.79%     | 77.36%      | 64.47%     | 12.35%   | 0.00%   | $0.61    |
| GPT-4o-mini | 0.85 | 2    | 11,312  | 50,003  | 90.99%     | 84.07%      | 72.54%     | 9.56%    | 71.00%  | $1.75    |
| GPT-4o-mini | 0.85 | 3    | 20,800  | 105,945 | 94.76%     | 88.78%      | 78.41%     | 8.14%    | 44.40%  | $3.69    |
| GPT-4o-mini | 0.85 | 4    | 36,552  | 213,214 | 96.65%     | 92.24%      | 83.23%     | 6.24%    | 41.60%  | $3.99    |
| GPT-4o-mini | 0.85 | 5    | 61,575  | 388,639 | 97.59%     | 94.03%      | 85.32%     | 5.56%    | 34.00%  | $10.08   |
| GPT-4o-mini | 0.9  | 1    | 5,646   | 19,742  | 86.06%     | 75.89%      | 65.30%     | 13.36%   | 0.00%   | $0.70    |
| GPT-4o-mini | 0.9  | 2    | 12,437  | 60,868  | 91.30%     | 84.38%      | 73.79%     | 12.05%   | 124.10% | $2.15    |
| GPT-4o-mini | 0.9  | 3    | 25,008  | 150,025 | 94.23%     | 89.83%      | 81.13%     | 9.63%    | 91.30%  | $5.31    |

#### **关键发现与分析**

##### **1. 收敛性验证：各设定下均趋于收敛**

所有实验设置都展现出清晰的收敛趋势：

- **新概念增长率**：从初始的10-13%稳步下降至3-5%，验证了图谱在概念覆盖上的饱和
- **新边增长率**：从第2轮的高峰（46-71%）逐步收敛至15-20%，证实了概念间关系网络的完善
- **覆盖率平台期**：在4-5轮后覆盖率增长显著放缓，进入收敛状态

##### **2. 模型差异分析：质量vs成本权衡**

**GPT-4o优势**：

- **更高的覆盖率**：在相同轮次下，GPT-4o通常比GPT-4o-mini高出2-4个百分点
- **更快的收敛**：达到相同覆盖率所需轮次更少
- **更好的概念质量**：体现在更高的维基百科关键词匹配率

**成本分析**：

- **GPT-4o成本**: 绝对成本约为GPT-4o-mini的8-15倍
- **性能提升**: 覆盖率提升通常在**2-4%**范围内
- **成本效益**: GPT-4o-mini在大多数实际应用场景下提供更佳的成本效益比

##### **3. 相似度阈值影响：精度与效率的平衡**

- **阈值0.8**: 快速扩展，但可能包含更多相似概念
- **阈值0.85**: 平衡的选择，适合大多数应用场景
- **阈值0.9**: 高精度但扩展速度较慢，适合对概念纯度要求极高的场景

##### **4. 跨领域一致性：方法的普适性**

心血管和呼吸系统两个领域的实验结果高度一致，验证了方法的鲁棒性：

- 收敛模式相似
- 覆盖率趋势一致
- 成本效益比类似

#### **实践建议**

**成本优先场景（推荐）**：

- 使用GPT-4o-mini + 阈值0.8
- 进行4-5轮迭代即可获得95%+的覆盖率
- 成本控制在$3-10范围内

**质量优先场景**：

- 使用GPT-4o + 阈值0.85
- 可获得97%+的高覆盖率
- 需要承担显著的成本增加

**平衡场景**：

- 使用GPT-4o-mini + 阈值0.85
- 达到94-97%的覆盖率
- 成本适中，质量可接受

## **基于概念图谱的知识蒸馏与QA记忆生成**

经过多轮迭代，我们已经构建了一个庞大且结构丰富的概念关系网络。然而，这个网络目前还只是一个“骨架”。为了使其成为能被AI直接利用的知识库，我们需要为其填充具体的内容——即高质量的问答（QA）对。这个过程，我们称之为\*\*“知识蒸馏”\*\*。

我们的策略分为两步：

1. 为图中的每个**单个概念节点**生成独立的问答对，以建立基础知识。
2. 为图中每个具有**临床指导意义的相连概念对**（边）生成关联性问答对，以构建深度知识。

为了实现这一目标，我们构建了 `ConceptDistiller`（概念蒸馏器）及其配套的Prompt。这个Prompt旨在引导一个强大的“教师”模型（如GPT-4o），将一个孤立的医学概念，转化为一个富有临床情景、考验综合推理能力的教学级问题和答案。这些QA对将成为“学生”模型（我们最终要增强的小模型）的记忆内容。

```python

@dataclass
class ConceptDistillationResult:
    """概念QA生成结果数据类"""
    status: str  # success, api_error, json_error
    concept_id: str
    concept_name: str
    response: Optional[str] = None
    error_details: Optional[str] = None
    processing_time: Optional[float] = None
    json_validation: Optional[Dict] = None
    generated_questions: Optional[List[Dict]] = None

class ConceptDistiller:
    """概念QA生成器 - 为每个概念生成问答对"""

    def __init__(self, api_client: APIClient):
        self.api_client = api_client

    def distill_concept(self, concept: str, concept_id: str) -> ConceptDistillationResult:
        """为单个概念生成问答对"""
        start_time = time.time()
      
        # 构建提示词
        system_prompt = self.get_distillation_system_prompt()
        user_prompt = self.get_distillation_prompt(concept)
      
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 调用API
        api_result = self.api_client.call_api(messages, timeout=300)
        processing_time = time.time() - start_time

        if api_result["status"] != "success":
            return ConceptDistillationResult(
                status="api_error",
                concept_id=concept_id,
                concept_name=concept,
                error_details=api_result["error"],
                processing_time=processing_time
            )

        # 验证JSON响应
        expected_keys = ["concept", "questions"]
        json_validation = ResponseValidator.validate_json_response(
            api_result["content"], expected_keys
        )
      
        if json_validation["is_valid_json"]:
            questions = json_validation["parsed_json"]["questions"]
            return ConceptDistillationResult(
                status="success",
                concept_id=concept_id,
                concept_name=concept,
                response=api_result["content"],
                processing_time=processing_time,
                json_validation=json_validation,
                generated_questions=questions
            )
        else:
            return ConceptDistillationResult(
                status="json_error",
                concept_id=concept_id,
                concept_name=concept,
                response=api_result["content"],
                error_details=f"JSON validation failed: {json_validation['error_type']}",
                processing_time=processing_time,
                json_validation=json_validation
            )

    @staticmethod
    def get_distillation_system_prompt() -> str:
        """获取系统提示词"""
        return """You are a world-renowned cardiovascular specialist with 20+ years of clinical experience. Your task is to create high-quality educational content for training junior cardiovascular doctors based on the given cardiovascular concept.

Your generated questions must require clinical reasoning and integration - avoid simple memorization questions."""

    @staticmethod
    def get_distillation_prompt(concept: str) -> str:
        """生成概念QA提示词"""
        return f"""**TARGET CONCEPT: {concept}**

Generate exactly 3 diverse cardiovascular clinical questions about this concept, each with complete learning materials. Follow these requirements:

**QUESTION DESIGN PRINCIPLES:**
1. Realistic cardiovascular clinical scenarios requiring clinical reasoning
2. Every condition mentioned must be CRITICAL to the clinical decision - avoid redundant details
3. Use general descriptors (elderly patient, young adult) rather than specific ages
4. Focus on decision-making situations where this concept is central
5. **AVOID simple factual questions** - require clinical integration and reasoning

**KNOWLEDGE FACTS REQUIREMENTS:**
- Each fact must start with the concept name as the subject
- Focus on core medical properties, mechanisms, clinical significance

**OUTPUT FORMAT (strict JSON):**
{{
  "concept": "{concept}",
  "questions": [
    {{
      "question_id": 1,
      "question": "Clinical scenario question 1...",
      "reasoning_guidance": "Step-by-step clinical thinking process 1...",
      "knowledge_facts": [
        "{concept} fact 1...",
        "{concept} fact 2...",  
        "{concept} fact 3..."
      ],
      "final_answer": "Comprehensive clinical answer..."
    }},
    {{
      "question_id": 2,
      "question": "Clinical scenario question 2...",
      "reasoning_guidance": "Step-by-step clinical thinking process 2...",
      "knowledge_facts": [
        "{concept} fact 1...",
        "{concept} fact 2..."
      ],
      "final_answer": "Comprehensive clinical answer..."
    }},
    {{
      "question_id": 3,
      "question": "Clinical scenario question 3...",
      "reasoning_guidance": "Step-by-step clinical thinking process 3...",
      "knowledge_facts": [
        "{concept} fact 1...",
        "{concept} fact 2..."
      ],
      "final_answer": "Comprehensive clinical answer..."
    }}
  ]
}}

Generate the educational content now."""

class BatchConceptDistiller:
    """批量概念QA生成处理器"""

    def __init__(self, distiller: ConceptDistiller, output_dir: str = "concept_distillation"):
        self.distiller = distiller
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def distill_concept_graph(self, concept_graph_dict: Dict, max_workers: int = 10, 
                            batch_size: int = 20, batch_delay: int = 0) -> Dict:
        """批量生成"""
        concept_list = list(concept_graph_dict.keys())  # 假设dict的key是concept名
        total_concepts = len(concept_list)
        print(f"开始批量处理: {total_concepts} 个概念的QA生成")
        print(f"批次大小: {batch_size}, 最大并发: {max_workers}")

        all_results = {}
        batch_num = 1

        # 分批处理
        for i in range(0, total_concepts, batch_size):
            batch_concepts = concept_list[i:i + batch_size]
            print(f"\n处理批次 {batch_num}: 概念 {i+1}-{min(i+batch_size, total_concepts)} ({len(batch_concepts)} 个)")

            batch_start_time = time.time()
            batch_results = self._process_batch(batch_concepts, max_workers, i)
            batch_end_time = time.time()

            # 保存批次结果
            self._save_batch_results(batch_results, batch_num, batch_start_time)

            all_results.update(batch_results)

            print(f"批次 {batch_num} 完成，耗时: {batch_end_time - batch_start_time:.2f} 秒")

            batch_num += 1

            # 批次间休息
            if i + batch_size < total_concepts:
                #print(f"批次间休息 {batch_delay} 秒...")
                time.sleep(batch_delay)

        print(f"\n所有批次处理完成！总计处理 {total_concepts} 个概念")
        return all_results

    def _process_batch(self, batch_concepts: List[str], max_workers: int, start_index: int) -> Dict:
        """处理单个批次"""
        batch_results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_concept = {}
            for idx, concept in enumerate(batch_concepts):
                concept_id = f"concept_{start_index + idx:06d}"
                future = executor.submit(self.distiller.distill_concept, concept, concept_id)
                future_to_concept[future] = (concept_id, concept)

            # 收集结果
            completed = 0
            for future in as_completed(future_to_concept):
                concept_id, concept = future_to_concept[future]
                try:
                    result = future.result()
                    batch_results[concept_id] = result

                    # 简单状态显示
                    status_symbol = "✓" if result.status == "success" else "✗"
                    completed += 1
                  
                    if completed % 1000 == 0 or completed == len(batch_concepts):
                        success_count = sum(1 for r in batch_results.values() if r.status == "success")
                        print(f"  已完成: {completed}/{len(batch_concepts)} (成功: {success_count}) {status_symbol}")

                except Exception as e:
                    batch_results[concept_id] = ConceptDistillationResult(
                        status="exception",
                        concept_id=concept_id,
                        concept_name=concept,
                        error_details=str(e)
                    )
                    print(f"  异常: {concept_id} - {str(e)}")

        return batch_results

    def _save_batch_results(self, batch_results: Dict, batch_num: int, start_time: float):
        """保存批次结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"concept_distillation_batch_{batch_num:03d}_{timestamp}.pkl"
        filepath = os.path.join(self.output_dir, filename)

        # 统计信息
        total_count = len(batch_results)
        success_count = sum(1 for r in batch_results.values() if r.status == "success")
        total_questions = sum(len(r.generated_questions) for r in batch_results.values() 
                            if r.status == "success" and r.generated_questions)

        save_data = {
            "metadata": {
                "batch_num": batch_num,
                "timestamp": timestamp,
                "start_time": start_time,
                "total_concepts": total_count,
                "successful_distillations": success_count,
                "total_questions_generated": total_questions
            },
            "results": batch_results
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"  批次结果已保存: {filename}")
        print(f"  成功: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        print(f"  生成问题总数: {total_questions}")

        return filepath

# ==================== 便捷使用函数 ====================

def distill_concept_graph(concept_graph_dict: Dict, api_url: str, api_key: str, model_name: str, max_workers: int = 10, 
                        batch_size: int = 20, output_dir: str = "concept_distillation"):
    """蒸馏概念图的便捷函数"""
    print(f"准备蒸馏概念图: {len(concept_graph_dict)} 个概念")

    # 初始化API客户端
    api_client = APIClient(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name
    )

    # 初始化蒸馏器
    distiller = ConceptDistiller(api_client)
    batch_distiller = BatchConceptDistiller(distiller, output_dir=output_dir)

    # 批量蒸馏
    results = batch_distiller.distill_concept_graph(
        concept_graph_dict=concept_graph_dict,
        max_workers=max_workers,
        batch_size=batch_size,
        batch_delay=1
    )

    return results

def test_single_concept_distillation(concept: str, api_url: str, api_key: str, model_name: str, verbose: bool = True):
    """测试单个概念的蒸馏"""
    print("=" * 80)
    print("单个概念蒸馏测试")
    print("=" * 80)

    # 初始化API客户端
    api_client = APIClient(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name
    )

    distiller = ConceptDistiller(api_client)

    print(f"概念: {concept}")
    print()

    # 处理概念
    result = distiller.distill_concept(concept, "test_concept")

    print(f"处理状态: {result.status}")
    print(f"处理耗时: {result.processing_time:.2f}秒")

    if result.status == "success":
        print(f"生成问题数量: {len(result.generated_questions)}")
        print("=" * 80)
        print("生成的数据:")
        print("=" * 80)
        for i, question in enumerate(result.generated_questions, 1):
            print(f"\n问题 {i}:")
            print(f"场景: {question['question']}")
            print(f"推理: {question['reasoning_guidance'][:100]}...")
            print(f"知识点: {len(question['knowledge_facts'])} 条")
            print(f"答案: {question['final_answer'][:100]}...")
        print("=" * 80)
        if verbose:
            print("原始LLM响应:")
            print("=" * 80)
            print(result.response)
        print("=" * 80)
        return {"success": True, "result": result}
    else:
        print(f"处理失败: {result.error_details}")
        return {"success": False, "result": result}

def load_and_analyze_distillation_results(results_dir: str = "concept_distillation"):
    """加载并分析结果"""
    result_files = [f for f in os.listdir(results_dir) 
                   if f.startswith('concept_distillation_batch_') and f.endswith('.pkl')]
    result_files.sort()

    if not result_files:
        print("未找到结果文件")
        return {}

    all_training_data = []
    total_concepts = 0
    total_successful = 0
    total_questions = 0

    print("结果分析:")
    print("=" * 80)

    for file in result_files:
        filepath = os.path.join(results_dir, file)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
      
        metadata = data['metadata']
        results = data['results']

        total_concepts += metadata['total_concepts']
        total_successful += metadata['successful_distillations']
        total_questions += metadata['total_questions_generated']

        print(f"批次 {metadata['batch_num']:3d}: "
              f"总概念 {metadata['total_concepts']:3d}, "
              f"成功 {metadata['successful_distillations']:3d} "
              f"({metadata['successful_distillations']/metadata['total_concepts']*100:.1f}%), "
              f"问题数 {metadata['total_questions_generated']:4d}")

        # 收集数据
        for concept_id, result in results.items():
            if result.status == "success" and result.generated_questions:
                for question in result.generated_questions:
                    training_sample = {
                        "concept": result.concept_name,
                        "concept_id": concept_id,
                        "question_id": question["question_id"],
                        "question": question["question"],
                        "reasoning_guidance": question["reasoning_guidance"],
                        "knowledge_facts": question["knowledge_facts"],
                        "final_answer": question["final_answer"]
                    }
                    all_training_data.append(training_sample)

    print("=" * 80)
    print(f"总计: {total_concepts} 个概念, 成功: {total_successful} ({total_successful/total_concepts*100:.1f}%)")
    print(f"生成QA: {len(all_training_data)} 个 (平均每概念 {len(all_training_data)/total_successful:.1f} 个)")

    return {
        "training_data": all_training_data,
        "statistics": {
            "total_concepts": total_concepts,
            "successful_distillations": total_successful,
            "total_questions": total_questions,
            "training_samples": len(all_training_data)
        }
    }

if __name__ == "__main__":
    print("=" * 80)
    print("概念QA生成系统")
    print("=" * 80)
  
    # 示例使用
    print("使用方法:")
    print("1. test_single_concept_distillation('atrial_fibrillation') - 测试单个概念")
    print("2. distill_concept_graph(concept_graph_dict) - 批量概念QA生成")
    print("3. load_and_analyze_distillation_results() - 分析结果")
    print("\n示例:")
    print("# 批量进行")
    print("# distillation_results = distill_concept_graph(your_concept_graph_dict, max_workers=10, batch_size=20)")
```

在运行大规模批量处理之前，验证Prompt和代码逻辑是否按预期工作是一个好习惯。下面的代码正是这样一个单元测试，它以“心房颤动（atrial_fibrillation）”为例，调用 `test_single_concept_distillation`函数来测试单个概念的QA生成效果。

```python
# 进行单个概念扩增的测试
test_single_concept_distillation('atrial_fibrillation')
```

在实际操作中，我们可能不需要对图谱中的所有概念都生成QA。根据项目目标，我们可以选择一个合适的子集。正如本节开头所述，我们的策略是选取一个中等规模（迭代3轮后的节点）但连接关系更成熟（迭代5轮后的边）的子图作为知识蒸馏的范围。

下面的代码正是为了实现这一目标。它首先加载了两个不同迭代阶段的图谱文件，然后通过 `extract_subgraph`和 `extract_unique_edges`函数，精确地构造出我们用于生成QA的目标图谱 `filtered_graph`。

```python
# 过滤得到目前的目标概念图

import pickle
with open('cookbooktest/Cardio/concept_graph_4omini_5_iter.pkl', 'rb') as f:
    concept_dict = pickle.load(f)

import pickle
with open('cookbooktest/Cardio/concept_graph_4omini_3_iter.pkl', 'rb') as f:
    sub_concept_dict = pickle.load(f)

def extract_subgraph(full_graph_dict, sub_concept_set):
    """提取子图，只保留指定概念及其之间的边"""
  
    # 如果sub_concept_set是dict，取其keys；如果是list/set，直接用
    if isinstance(sub_concept_set, dict):
        valid_concepts = set(sub_concept_set.keys())
    else:
        valid_concepts = set(sub_concept_set)
  
    subgraph = {}
  
    for concept, neighbors in full_graph_dict.items():
        # 只处理在子集中的概念
        if concept in valid_concepts:
            # 只保留邻居中也在子集中的概念
            filtered_neighbors = [n for n in neighbors if n in valid_concepts]
            if filtered_neighbors:  # 只保留有邻居的节点
                subgraph[concept] = filtered_neighbors
  
    print(f"原图: {len(full_graph_dict)} 个节点")
    print(f"子图: {len(subgraph)} 个节点")
  
    # 统计边数
    total_edges = sum(len(neighbors) for neighbors in subgraph.values())
    print(f"子图边数: {total_edges}")
  
    return subgraph

filtered_graph = extract_subgraph(concept_dict, sub_concept_dict)

# 去除重复边
def extract_unique_edges(graph_dict):
    """从图中提取唯一边对，双向边只保留一条"""
  
    processed_pairs = set()
    unique_edges = []
  
    for concept_a, neighbors in graph_dict.items():
        for concept_b in neighbors:
            # 排序确保(A,B)和(B,A)被视为同一对
            edge = tuple(sorted([concept_a, concept_b]))
    
            if edge not in processed_pairs:
                processed_pairs.add(edge)
                unique_edges.append(edge)
  
    print(f"总边数: {sum(len(neighbors) for neighbors in graph_dict.values())}")
    print(f"去重后边数: {len(unique_edges)}")
  
    return unique_edges

# 使用
unique_edges = extract_unique_edges(filtered_graph)

# 查看前几条边
print("前5条边:")
for i, (a, b) in enumerate(unique_edges[:5]):
    print(f"{i+1}. {a} <-> {b}")
```

```python
# 批量进行单个concept的qa提取
results = distill_concept_graph(
    concept_graph_dict=example_concept_dict,
    max_workers=100,        # 并发数
    batch_size=1000,        # 批次大小
    output_dir="cookbooktest/Cardio",  # 保存位置
    api_url=api_url, 
    api_key=api_key, 
    model_name=QA_SYNTHESIZER
)
```

在实际操作中，我们选择了一个中等规模（迭代3轮后的节点）但连接关系更成熟（迭代5轮后的边）的子图作为知识蒸馏的范围，以平衡知识的广度与深度。对于概念对的知识蒸馏，我们设计了一个更为复杂的“评估-生成”两段式Prompt。LLM首先扮演“过滤器”的角色，严格评估概念对的临床关联性和教学价值，只有通过评估的“黄金组合”才会进入第二阶段，生成一个同时涵盖两个概念、逻辑更复杂的QA对。

```python
# 对于concept pair的问题生成提取，按下不表，我们提供prompt作为启示：
    @staticmethod
    def get_pair_system_prompt() -> str:
        """获取概念对评估系统提示词"""
        return """You are a world-renowned cardiovascular specialist with 20+ years of clinical experience. Your task is to rigorously evaluate concept pairs and generate high-quality educational content. You must act as a **strict filter**, approving only pairs with a **direct, critical, and undeniable link** in clinical practice and training."""

    @staticmethod
    def get_pair_prompt(concept_pairs: List[tuple]) -> str:
        """生成概念对评估提示词"""
        pairs_text = ""
        for i, (concept_a, concept_b) in enumerate(concept_pairs, 1):
            pairs_text += f"{i}. {concept_a} <-> {concept_b}\n"
  
        return f"""**CONCEPT PAIRS TO EVALUATE:**
{pairs_text}

For each pair, you must strictly evaluate the following two criteria. **BOTH must be strongly true** to proceed.

1.  **Direct Clinical Relevance**: Is there a **direct causal, pathophysiological, diagnostic, or therapeutic link** between the two concepts? The connection should not just a weak, coincidental, or indirect association. One concept must frequently and directly influence the consideration of the other in **critical clinical decision-making**.

2.  **Essential Educational Value**: Does understanding this specific link teach a **crucial, non-obvious clinical reasoning skill**? The relationship should highlight a common point of confusion to be clarified, a key differential diagnosis, or a pivotal management decision. It must be more than a simple factual association.

**EXAMPLE OF A PAIR TO REJECT:**
- `"Hypertension" <-> "Stethoscope"`: While a stethoscope is used in the diagnosis of hypertension, this is a basic procedural fact.

For each pair that meet the stringent criteria:
1. Generate 1 clinical question covering BOTH concepts.
2. Every condition mentioned must be CRITICAL to the clinical decision - avoid redundant details
3. Use general descriptors (elderly patient, young adult) rather than specific ages
4. Focus on decision-making situations where simultaneously considering the concept pairs is central
5. **AVOID simple factual questions** - require clinical integration and reasoning

**OUTPUT FORMAT (strict JSON):**
{{
  "evaluated_pairs": [
    {{
      "concept_pair": ["concept_a", "concept_b"],
      "is_clinically_relevant": true,
      "is_instructionally_meaningful": true,
      "question": {{
        "question": "Clinical scenario covering both concepts...",
        "reasoning_guidance": "Step-by-step clinical thinking...",
        "knowledge_facts": [
          "Concept_a fact 1...",
          "Concept_b fact 1...",
          "Concept_a fact 2..."
        ],
        "final_answer": "Comprehensive answer..."
      }}
    }},
    {{
      "concept_pair": ["concept_x", "concept_y"],
      "is_clinically_relevant": false,
      "is_instructionally_meaningful": false,
      "question": null
    }}
  ]
}}

Generate the evaluation and content now."""
```

---

## **QA数据结构示例**

所有经过知识蒸馏流程生成的QA数据，都会被整理成统一、标准的JSON对象格式，以便于后续的程序读取和处理。该结构包含以下关键字段：

- `concept`: 知识来源，可以是单个概念（字符串）或概念对（列表）。
- `question`: 核心的临床问题。
- `reasoning_guidance`: 解决该问题的临床思维路径。
- `knowledge_facts`: 解答该问题所需的核心知识点。
- `final_answer`: 对问题的综合性、权威性答案。

我们将所有格式化后的QA数据集合命名为 `qa_collection`。

1. **单点概念 QA 示例**

```python
{'concept': 'ankylosing spondylitis',
 'question': 'A young adult patient with a 5-year history of ankylosing spondylitis presents with unexplained fatigue and palpitations. Laboratory tests reveal anemia and elevated acute phase reactants. In the context of ankylosing spondylitis, what cardiovascular complication should be explored, and what is the likely mechanism of the heart condition related to this systemic inflammatory disease?',
 'reasoning_guidance': 'Identify the common systemic manifestations of ankylosing spondylitis including inflammation and anemia. Consider the cardiovascular implications of chronic inflammation and anemia on cardiac function. Explore the mechanism by which systemic diseases like ankylosing spondylitis can result in heart conditions such as myocardial fibrosis or dysfunction.',
 'knowledge_facts': ['ankylosing spondylitis can cause systemic inflammation, contributing to cardiovascular complications like myocardial fibrosis.',
  'ankylosing spondylitis-associated inflammation can lead to chronic anemia, affecting cardiovascular health.',
  'ankylosing spondylitis may lead to cardiac conduction system involvement, resulting in palpitations.'],
 'final_answer': "Given the patient's symptoms and laboratory findings, myocardial fibrosis due to systemic inflammation related to ankylosing spondylitis should be explored. The fatigue and palpitations may be due, in part, to anemia exacerbating cardiac stress, and inflammation leading to fibrosis, altering cardiac conduction and function."}

```

2. **概念对 QA 示例**

```python
{'concept': ['apical hypertrophy of the lv', 'myocardial ischaemia'],
 'question': 'A middle-aged adult with a history of hypertension presents with exertional chest pain. Echocardiography reveals apical hypertrophy of the left ventricle. How would you differentiate between hypertrophic cardiomyopathy and myocardial ischaemia as the cause of the symptoms?',
 'reasoning_guidance': 'Consider the role of diagnostic imaging and stress testing in distinguishing between structural heart changes and ischemic heart conditions. Evaluate the characteristic findings of apical hypertrophy and myocardial ischemia.',
 'knowledge_facts': ['Apical hypertrophy can mimic signs of myocardial ischaemia.',
  'Myocardial ischaemia is often indicated by ST-segment changes during stress.',
  'Hypertrophic cardiomyopathy may present with specific echocardiographic patterns of ventricular thickening.'],
 'final_answer': 'To differentiate hypertrophic cardiomyopathy from myocardial ischaemia, perform a stress test to assess for changes indicative of ischemia and use advanced imaging modalities like cardiac MRI, which can provide detailed myocardial characterization.'}

```

---

## **最后一步：构建并导出MemCube**

至此，所有数据准备工作均已完成。我们现在要将这些独立的“知识单元”(`qa_collection`)组装成一个强大、互联的知识网络——**MemCube**。

构建流程如下：

1. **概念为骨架**：概念图谱中的每一个“概念”，都将成为MemCube中的一个独立的**节点**。
2. **QA为血肉**：每一个“QA对”，也将成为一个独立的**节点**，并与它所来源的一个或两个概念节点相连。
3. **问题为索引**：我们将每个QA节点中的**问题（question）文本进行向量化**，作为其在记忆网络中的语义“地址”，用于快速检索。

下面的Python脚本一站式地完成了这个转换过程。它加载 `qa_collection`，提取并创建所有概念节点和QA节点，然后根据预设的逻辑在节点之间建立关系边，最后将所有节点和边组装成一个完整的、符合MemOS格式的JSON对象，并导出为文件。

```python
import os
from sentence_transformers import SentenceTransformer
import torch
model = SentenceTransformer(
    EMBEDDING_MODEL,
    trust_remote_code=True
)
# =============================================================================
# Cell 1: 导入库和辅助函数
# =============================================================================
import pickle
import uuid
import json
from datetime import datetime
from collections import defaultdict
import numpy as np

# 数据加载
with open("cookbooktest/Cardio/qa_collection.pkl", 'rb') as f:
    qa_collection = pickle.load(f)

print(f"✅ 加载了 {len(qa_collection)} 个QA数据")

def generate_real_embedding_batch(texts, batch_size=50):
    """批量生成embedding向量"""
    if isinstance(texts, str):
        # 单个文本，直接处理
        embedding = model.encode(texts, convert_to_tensor=False)
        return embedding.tolist()
  
    # 批量处理
    all_embeddings = []
    total = len(texts)
  
    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        batch_texts = texts[i:batch_end]
  
        print(f"  Embedding批次 {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch_texts)} 个文本)")
  
        # 批量编码
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=False, show_progress_bar=False)
  
        # 转换为list并添加到结果
        for emb in batch_embeddings:
            all_embeddings.append(emb.tolist())
  
    return all_embeddings

# =============================================================================
# Cell 2: 数据验证和概念提取
# =============================================================================
def extract_unique_concepts(qa_collection):
    """从QA数据中提取所有唯一概念，并验证数据格式"""
    unique_concepts = set()
    invalid_data = []
    valid_concept_qa = 0
    valid_relation_qa = 0
  
    for i, qa_data in enumerate(qa_collection):
        if isinstance(qa_data['concept'], str):
            # Concept QA - 单个概念
            unique_concepts.add(qa_data['concept'])
            valid_concept_qa += 1
        elif isinstance(qa_data['concept'], list):
            # Relation QA - 应该是2个概念的配对
            if len(qa_data['concept']) == 2:
                unique_concepts.update(qa_data['concept'])
                valid_relation_qa += 1
            else:
                # 数据异常：不是2个概念
                invalid_data.append({
                    'index': i,
                    'concept': qa_data['concept'],
                    'length': len(qa_data['concept']),
                    'question': qa_data['question'][:100] + "..."
                })
        else:
            # 数据异常：concept既不是str也不是list
            invalid_data.append({
                'index': i,
                'concept': qa_data['concept'],
                'type': type(qa_data['concept']),
                'question': qa_data['question'][:100] + "..."
            })
  
    # 报告数据验证结果
    print(f"📊 数据验证结果:")
    print(f"   - 有效 Concept QA: {valid_concept_qa}")
    print(f"   - 有效 Relation QA: {valid_relation_qa}")
    print(f"   - 异常数据: {len(invalid_data)}")
    print(f"   - 提取的唯一概念: {len(unique_concepts)}")
  
    if invalid_data:
        print(f"\n⚠️ 异常数据详情:")
        for item in invalid_data[:3]:  # 只显示前3个
            print(f"   索引{item['index']}: concept={item['concept']}")
            print(f"     问题: {item['question']}")
        if len(invalid_data) > 3:
            print(f"   ... 还有 {len(invalid_data) - 3} 个异常数据")
  
    return list(unique_concepts), invalid_data, valid_concept_qa, valid_relation_qa

# 执行数据验证
print("🔍 开始数据验证...")
unique_concepts, invalid_data, valid_concept_qa, valid_relation_qa = extract_unique_concepts(qa_collection)

print(f"\n✅ 概念列表示例: {list(unique_concepts)[:5]}...")

# =============================================================================
# Cell 3: 创建概念节点
# =============================================================================
def create_concept_nodes(unique_concepts):
    """创建所有概念节点 - 使用概念名称作为memory和embedding"""
    concept_nodes = {}
  
    print(f"开始为 {len(unique_concepts)} 个概念生成embedding...")
  
    # 批量生成概念embedding
    concept_embeddings = generate_real_embedding_batch(unique_concepts, batch_size=100)
  
    for i, (concept, embedding) in enumerate(zip(unique_concepts, concept_embeddings)):
        concept_id = str(uuid.uuid4())
  
        node = {
            "id": concept_id,
            "memory": concept,  # 概念名称作为memory
            "metadata": {
                "type": "fact",
                "memory_type": "UserMemory",
                "status": "activated",
                "entities": [concept],
                "tags": [concept],
                "embedding": embedding,  # 概念名称的embedding
                "created_at": datetime.now().isoformat(),
                "usage": [],
                "background": ""
            }
        }
  
        concept_nodes[concept] = {
            "id": concept_id,
            "node": node
        }
  
        if (i + 1) % 20 == 0:
            print(f"  已完成 {i + 1}/{len(unique_concepts)} 个概念")
  
    print(f"✅ 创建了 {len(concept_nodes)} 个概念节点")
    return concept_nodes

# 执行概念节点创建
print("🏗️ 创建概念节点...")
concept_nodes = create_concept_nodes(unique_concepts)
print("概念节点示例ID:", list(concept_nodes.keys())[0], "->", concept_nodes[list(concept_nodes.keys())[0]]["id"])

# =============================================================================
# Cell 4: 创建QA节点
# =============================================================================
def create_qa_nodes(qa_collection, concept_nodes):
    """创建所有QA节点 - 批量embedding优化"""
  
    # 1. 先收集所有问题文本和元数据
    all_questions = []
    all_metadata = []
    skipped_count = 0
  
    for qa_data in qa_collection:
        question = qa_data['question']
  
        # 构建完整memory内容
        memory_content = f"""Question: {qa_data['question']}

Reasoning Guidance: {qa_data['reasoning_guidance']}

Knowledge Facts: {'; '.join(qa_data['knowledge_facts'])}

Answer: {qa_data['final_answer']}"""
  
        # 判断QA类型并准备metadata
        if isinstance(qa_data['concept'], str):
            # Concept QA
            concept_name = qa_data['concept']
            if concept_name not in concept_nodes:
                print(f"  警告: 概念 '{concept_name}' 不存在，跳过该QA")
                skipped_count += 1
                continue
        
            qa_type = "concept_qa"
            entities = [concept_name]
            tags = [concept_name]
            related_concept_ids = [concept_nodes[concept_name]["id"]]
    
        elif isinstance(qa_data['concept'], list) and len(qa_data['concept']) == 2:
            # Relation QA
            concept_names = qa_data['concept']
    
            # 检查所有概念是否存在
            missing_concepts = [name for name in concept_names if name not in concept_nodes]
            if missing_concepts:
                print(f"  警告: 概念 {missing_concepts} 不存在，跳过该QA")
                skipped_count += 1
                continue
        
            qa_type = "relation_qa"
            entities = concept_names
            tags = concept_names
            related_concept_ids = [concept_nodes[name]["id"] for name in concept_names]
    
        else:
            # 跳过异常数据
            skipped_count += 1
            continue
  
        all_questions.append(question)
        all_metadata.append({
            'memory_content': memory_content,
            'qa_type': qa_type,
            'entities': entities,
            'tags': tags,
            'related_concept_ids': related_concept_ids
        })
  
    print(f"收集了 {len(all_questions)} 个有效问题（跳过 {skipped_count} 个），开始批量生成embedding...")
  
    # 2. 批量生成所有问题的embedding
    all_embeddings = generate_real_embedding_batch(all_questions, batch_size=100)
  
    # 3. 创建QA节点
    qa_nodes = []
    concept_qa_count = 0
    relation_qa_count = 0
  
    for i, (question, metadata, embedding) in enumerate(zip(all_questions, all_metadata, all_embeddings)):
        qa_id = str(uuid.uuid4())
  
        node = {
            "id": qa_id,
            "memory": metadata['memory_content'],
            "metadata": {
                "type": "fact",
                "memory_type": "UserMemory",
                "status": "activated",
                "entities": metadata['entities'],
                "tags": metadata['tags'],
                "embedding": embedding,  # 问题的embedding
                "created_at": datetime.now().isoformat(),
                "usage": [],
                "background": "",
                # 临时字段，用于创建边关系
                "qa_type": metadata['qa_type'],
                "related_concept_ids": metadata['related_concept_ids']
            }
        }
  
        qa_nodes.append(node)
  
        if metadata['qa_type'] == "concept_qa":
            concept_qa_count += 1
        else:
            relation_qa_count += 1
  
        if (i + 1) % 50 == 0:
            print(f"  已创建 {i + 1}/{len(all_questions)} 个QA节点")
  
    print(f"✅ 创建了 {len(qa_nodes)} 个QA节点")
    print(f"   - Concept QA: {concept_qa_count}")
    print(f"   - Relation QA: {relation_qa_count}")
  
    return qa_nodes

# 执行QA节点创建
print("🏗️ 创建QA节点...")
qa_nodes = create_qa_nodes(qa_collection, concept_nodes)
if qa_nodes:
    print(f"QA节点示例: {qa_nodes[0]['metadata']['qa_type']}")
```

```python

# =============================================================================
# Cell 5: 创建关系边
# =============================================================================
def create_edges(concept_nodes, qa_nodes, qa_collection):
    """创建节点间的关系边"""
    edges = []
    edge_set = set()  # 用于去重边
  
    # 1. 概念↔概念 RELATE_TO 关系 (从Relation QA推导)
    concept_relations = set()
    for qa_data in qa_collection:
        if isinstance(qa_data['concept'], list) and len(qa_data['concept']) == 2:
            # Relation QA表示两个概念间有临床相关关系
            concept_A, concept_B = qa_data['concept']
            if concept_A in concept_nodes and concept_B in concept_nodes:
                relation_key = tuple(sorted([concept_A, concept_B]))
                concept_relations.add(relation_key)
  
    relate_count = 0
    for concept_A, concept_B in concept_relations:
        concept_A_id = concept_nodes[concept_A]["id"]
        concept_B_id = concept_nodes[concept_B]["id"]
  
        edge_key = tuple(sorted([concept_A_id, concept_B_id]))
        if edge_key not in edge_set:
            edges.append({
                "source": concept_A_id,
                "target": concept_B_id,
                "type": "RELATE_TO"
            })
            edge_set.add(edge_key)
            relate_count += 1
  
    print(f"✅ 创建了 {relate_count} 条概念间 RELATE_TO 关系")
  
    # 2. 概念 PARENT QA 关系 (Concept QA)
    parent_count = 0
    for qa_node in qa_nodes:
        if qa_node['metadata']['qa_type'] == "concept_qa":
            concept_id = qa_node['metadata']['related_concept_ids'][0]
    
            edges.append({
                "source": concept_id,
                "target": qa_node['id'],
                "type": "PARENT"
            })
            parent_count += 1
  
    print(f"✅ 创建了 {parent_count} 条概念→QA PARENT 关系")
  
    # 3. 概念 PARENT QA 关系 (Relation QA - 桥接问题)
    relation_parent_count = 0
    for qa_node in qa_nodes:
        if qa_node['metadata']['qa_type'] == "relation_qa":
            qa_id = qa_node['id']
    
            # 确保related_concept_ids有效
            if 'related_concept_ids' in qa_node['metadata']:
                for concept_id in qa_node['metadata']['related_concept_ids']:
                    edges.append({
                        "source": concept_id,   # 概念作为父节点
                        "target": qa_id,        # 桥接问题作为子节点
                        "type": "PARENT"
                    })
                    relation_parent_count += 1
  
    print(f"✅ 创建了 {relation_parent_count} 条概念→桥接QA PARENT 关系")
    print(f"📊 总关系数: {len(edges)}")
  
    return edges

# 执行边关系创建
print("🔗 创建关系边...")
edges = create_edges(concept_nodes, qa_nodes, qa_collection)

# =============================================================================
# Cell 6: 组装和保存最终JSON
# =============================================================================
def assemble_final_json(concept_nodes, qa_nodes, edges):
    """组装最终的TextualMemoryItem格式JSON"""
  
    # 合并所有节点
    all_nodes = []
  
    # 添加概念节点
    for concept_data in concept_nodes.values():
        all_nodes.append(concept_data["node"])
  
    # 添加QA节点，清理临时字段
    for qa_node in qa_nodes:
        # 深拷贝节点，避免修改原数据
        clean_node = {
            "id": qa_node["id"],
            "memory": qa_node["memory"],
            "metadata": qa_node["metadata"].copy()
        }
  
        # 移除临时字段
        if "qa_type" in clean_node["metadata"]:
            del clean_node["metadata"]["qa_type"]
        if "related_concept_ids" in clean_node["metadata"]:
            del clean_node["metadata"]["related_concept_ids"]
  
        all_nodes.append(clean_node)
  
    # 构建最终结构
    result = {
        "nodes": all_nodes,
        "edges": edges
    }
  
    print(f"✅ 最终JSON包含:")
    print(f"   - 节点数: {len(all_nodes)}")
    print(f"   - 边数: {len(edges)}")
    print(f"   - 概念节点: {len(concept_nodes)}")
    print(f"   - QA节点: {len(qa_nodes)}")
    print(f"✅ 已清理临时字段")
  
    return result

# 执行最终组装
print("📦 组装最终JSON...")
final_json = assemble_final_json(concept_nodes, qa_nodes, edges)
```

```python
def save_final_json(result, filename="cardio_textual_memory_graph.json"):
    """保存最终JSON到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
  
    print(f"✅ 已保存到文件: {filename}")
    return filename


# 保存结果
filename = save_final_json(final_json, "cookbooktest/Cardio/cardio_textual_memory_graph.json")

print("\n🎉 转换完成！")
print(f"📄 输出文件: {filename}")
print(f"📋 最终统计:")
print(f"   - 总节点: {len(final_json['nodes'])}")
print(f"   - 总边: {len(final_json['edges'])}")

# 显示一些示例数据验证
if final_json['nodes']:
    sample_node = final_json['nodes'][0]
    print(f"\n📝 示例节点:")
    print(f"   ID: {sample_node['id']}")
    print(f"   Memory: {sample_node['memory'][:50]}...")
    print(f"   Type: {sample_node['metadata']['type']}")
    print(f"   Entities: {sample_node['metadata']['entities']}")

if final_json['edges']:
    sample_edge = final_json['edges'][0]
    print(f"\n🔗 示例边:")
    print(f"   {sample_edge['source']} --{sample_edge['type']}--> {sample_edge['target']}")
```

### **加载MemCube**

现在，是时候将这份“数字蓝图”注入一个高性能的持久化存储中，使其成为一个可供MemOS实时访问的MemCube。

我们提供了一个为性能优化的批量导入工具脚本，它能够绕过逐条添加的瓶颈，直接、高效地将整个MemCube加载，同时确保其数据结构与MemOS完全兼容。该脚本的核心任务包括：创建数据库约束、批量导入节点和边、创建向量索引（这是实现毫秒级语义搜索的核心）以及兼容性验证。

```python
# 加载memcube进入neo4j

#!/usr/bin/env python3
import sys
import os
import ijson
import json
import time
from datetime import datetime
from decimal import Decimal
from neo4j import GraphDatabase

# ===================== 配置信息 - 请修改以下信息 =====================
NEO4J_URI = 'bolt://localhost:7687'
NEO4J_USERNAME = 'your neo4j username'
NEO4J_PASSWORD = 'your neo4j password'
NEO4J_DATABASE = 'neo4j'
JSON_FILE_PATH = 'cookbooktest/Cardio/cardio_textual_memory_graph.json'
# ===================================================================

# 全局驱动实例
driver = None

def get_driver():
    """获取Neo4j驱动实例"""
    global driver
    if not driver:
        try:
            driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )
        except Exception as e:
            print(f"❌ 创建驱动失败: {e}")
            sys.exit(1)
    return driver

def close_driver():
    """关闭驱动连接"""
    global driver
    if driver:
        driver.close()
        driver = None

def test_neo4j_connection():
    """测试Neo4j连接"""
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run("RETURN 'Connection OK' AS message")
            print(f"✅ Neo4j连接成功: {result.single()['message']}")
        return True
    except Exception as e:
        print(f"❌ Neo4j连接失败: {e}")
        return False

def create_memos_compatible_schema():
    """创建MemOS兼容的schema和索引"""
    print("创建MemOS兼容的数据结构...")
  
    try:
        driver = get_driver()
        with driver.session() as session:
            # 创建MemOS兼容的约束
            session.run("""
                CREATE CONSTRAINT memory_id_unique IF NOT EXISTS
                FOR (n:Memory) REQUIRE n.id IS UNIQUE
            """)
            print("✅ 创建Memory节点ID唯一约束")
        return True
  
    except Exception as e:
        print(f"❌ Schema创建失败: {e}")
        return False

def bulk_import_nodes():
    """批量导入节点 - Neo4j原生方式"""
    print("\n" + "=" * 50)
    print("开始Neo4j原生批量导入节点")
    print("=" * 50)
  
    driver = config.get_driver()
    start_time = time.time()
    success_count = 0
    batch_size = 5000  # 大批次以获得最佳性能
    batch = []
  
    try:
        with open(config.json_file_path, 'rb') as f:
            nodes = ijson.items(f, 'nodes.item')
    
            for node in nodes:
                # 准备MemOS兼容的节点数据
                node_data = prepare_memos_node(node)
                batch.append(node_data)
        
                # 执行批量导入
                if len(batch) >= batch_size:
                    batch_success = execute_node_batch(driver, batch)
                    success_count += batch_success
                    batch = []
            
                    # 显示进度
                    elapsed = time.time() - start_time
                    rate = success_count / elapsed
                    eta_minutes = (200000 - success_count) / rate / 60
            
                    print(f"  已导入: {success_count:,}/200,000 ({success_count/200000*100:.1f}%) | "
                          f"速度: {rate:.1f}节点/秒 | "
                          f"预计剩余: {eta_minutes:.1f}分钟")
    
            # 处理剩余批次
            if batch:
                batch_success = execute_node_batch(driver, batch)
                success_count += batch_success
  
        total_time = time.time() - start_time
        print(f"\n✅ 节点批量导入完成:")
        print(f"  导入数量: {success_count:,}")
        print(f"  总时间: {total_time/60:.1f}分钟")
        print(f"  平均速度: {success_count/total_time:.1f}节点/秒")
        return success_count
  
    except Exception as e:
        print(f"❌ 批量导入失败: {e}")
        return success_count


def clean_data_types(obj):
    """清理数据类型，确保Neo4j兼容"""
    if isinstance(obj, dict):
        return {k: clean_data_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data_types(item) for item in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    elif obj is None:
        return None
    else:
        return obj

def prepare_memos_node(node):
    """准备MemOS兼容的节点数据"""
    # 先清理数据类型
    node = clean_data_types(node)
    metadata = node.get('metadata', {}).copy()
  
    # 确保必要字段
    if 'created_at' not in metadata:
        metadata['created_at'] = datetime.now().isoformat()
    if 'updated_at' not in metadata:
        metadata['updated_at'] = datetime.now().isoformat()
  
    return {
        'id': node.get('id'),
        'memory': node.get('memory', ''),
        'metadata': clean_data_types(metadata)
    }

def execute_node_batch(driver, batch):
    """执行节点批次导入"""
    cypher_query = """
    UNWIND $batch AS nodeData
    MERGE (n:Memory {id: nodeData.id})
    SET n.memory = nodeData.memory,
        n.created_at = datetime(nodeData.metadata.created_at),
        n.updated_at = datetime(nodeData.metadata.updated_at),
        n += nodeData.metadata
    RETURN count(n) as imported
    """
  
    try:
        with driver.session() as session:
            result = session.run(cypher_query, batch=batch)
            return result.single()['imported']
    except Exception as e:
        print(f"  批次导入错误: {e}")
        return 0

def bulk_import_edges():
    """批量导入边"""
    print("\n" + "=" * 50)
    print("开始Neo4j原生批量导入边")
    print("=" * 50)
  
    driver = config.get_driver()
    start_time = time.time()
    success_count = 0
    batch_size = 10000  # 边可以用更大的批次
    batch = []
  
    try:
        with open(config.json_file_path, 'rb') as f:
            edges = ijson.items(f, 'edges.item')
    
            for edge in edges:
                # 清理边数据类型
                edge_clean = clean_data_types(edge)
                batch.append({
                    'source': edge_clean.get('source'),
                    'target': edge_clean.get('target'),
                    'type': edge_clean.get('type')
                })
        
                if len(batch) >= batch_size:
                    batch_success = execute_edge_batch(driver, batch)
                    success_count += batch_success
                    batch = []
            
                    elapsed = time.time() - start_time
                    rate = success_count / elapsed
                    eta_minutes = (500000 - success_count) / rate / 60
            
                    if success_count % 50000 == 0:  # 每5万条显示
                        print(f"  已导入: {success_count:,}/500,000 ({success_count/500000*100:.1f}%) | "
                              f"速度: {rate:.1f}边/秒 | "
                              f"预计剩余: {eta_minutes:.1f}分钟")
    
            # 处理剩余批次
            if batch:
                batch_success = execute_edge_batch(driver, batch)
                success_count += batch_success
  
        total_time = time.time() - start_time
        print(f"\n✅ 边批量导入完成:")
        print(f"  导入数量: {success_count:,}")
        print(f"  总时间: {total_time/60:.1f}分钟")
        print(f"  平均速度: {success_count/total_time:.1f}边/秒")
        return success_count
  
    except Exception as e:
        print(f"❌ 边导入失败: {e}")
        return success_count

def execute_edge_batch(driver, batch):
    """执行边批次导入"""
    cypher_query = """
    UNWIND $batch AS edgeData
    MATCH (source:Memory {id: edgeData.source})
    MATCH (target:Memory {id: edgeData.target})
    MERGE (source)-[r:PARENT]->(target)
    RETURN count(r) as imported
    """
  
    try:
        with driver.session() as session:
            result = session.run(cypher_query, batch=batch)
            return result.single()['imported']
    except Exception as e:
        print(f"  边批次导入错误: {e}")
        return 0

def create_memos_indexes():
    """创建MemOS需要的索引"""
    print("\n" + "=" * 50)
    print("创建MemOS兼容索引")
    print("=" * 50)
  
    try:
        driver = config.get_driver()
        with driver.session() as session:
            # MemOS常用索引
            indexes = [
                "CREATE INDEX memory_type_idx IF NOT EXISTS FOR (n:Memory) ON (n.memory_type)",
                "CREATE INDEX memory_status_idx IF NOT EXISTS FOR (n:Memory) ON (n.status)",
                "CREATE INDEX memory_created_at_idx IF NOT EXISTS FOR (n:Memory) ON (n.created_at)",
                "CREATE INDEX memory_updated_at_idx IF NOT EXISTS FOR (n:Memory) ON (n.updated_at)",
                "CREATE INDEX memory_user_name_index IF NOT EXISTS FOR (n:Memory) ON (n.user_name)"
            ]
    
            for index_query in indexes:
                session.run(index_query)
                print(f"✅ 索引创建: {index_query.split()[-7]}")  # 提取索引名
    
            # 创建向量索引 - MemOS向量搜索必需
            try:
                session.run("""
                    CREATE VECTOR INDEX memory_vector_index IF NOT EXISTS
                    FOR (n:Memory) ON (n.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                print("✅ 向量索引创建: memory_vector_index (768维)")
            except Exception as ve:
                print(f"⚠️  向量索引创建失败: {ve}")
                print("   向量搜索功能将不可用")
        print("✅ 所有MemOS兼容索引创建完成")
  
    except Exception as e:
        print(f"❌ 索引创建失败: {e}")

def verify_memos_compatibility():
    """验证MemOS兼容性"""
    print("\n" + "=" * 50)
    print("验证MemOS兼容性")
    print("=" * 50)
  
    try:
        # 添加MemOS路径
        sys.path.append('./MemOS/src')
        from memos.configs.graph_db import GraphDBConfigFactory
        from memos.graph_dbs.factory import GraphStoreFactory
  
        # 创建MemOS配置
        graph_config = GraphDBConfigFactory(
            backend="neo4j",
            config={
                "uri": config.uri,
                "user": config.username,
                "password": config.password,
                "db_name": config.database,
                "auto_create": False,
                "embedding_dimension": 768,
            }
        )
  
        graph_store = GraphStoreFactory.from_config(graph_config)
  
        # 测试基本功能
        try:
            node_count = graph_store.count_nodes("UserMemory")
            print(f"✅ MemOS节点统计: {node_count:,} 个UserMemory节点")
        except:
            print("⚠️  节点统计功能需要微调")
  
        # 测试导出功能
        try:
            exported = graph_store.export_graph()
            print(f"✅ MemOS图导出: {len(exported.get('nodes', []))} 节点, {len(exported.get('edges', []))} 边")
        except Exception as e:
            print(f"⚠️  图导出功能: {e}")
  
        print("✅ MemOS兼容性验证完成")
        return True
  
    except Exception as e:
        print(f"❌ MemOS兼容性验证失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 Neo4j 批量导入工具")
    print("=" * 50)
  
    try:
        # 1. 获取用户配置 - 一次性输入所有信息
        config.get_user_input()
      
        # 2. 测试连接
        if not test_neo4j_connection():
            return
  
        # 3. 创建兼容schema
        if not create_memos_compatible_schema():
            return
  
        # 4. 显示预估
        print(f"\n直接Neo4j批量导入预估:")
        print(f"  节点数: 200,000")
        print(f"  边数: 500,000")
        print(f"  批次大小: 5,000节点/批次, 10,000边/批次")
        print(f"  预期速度: 1000+节点/秒, 5000+边/秒")
        print(f"  预计时间: 15-25分钟")
  
        confirm = input("\n是否开始直接批量导入? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 用户取消导入")
            return
  
        # 5. 执行导入
        total_start = time.time()
  
        # 导入节点
        node_count = bulk_import_nodes()
  
        # 导入边
        edge_count = bulk_import_edges()
  
        # 创建索引
        create_memos_indexes()
  
        # 验证兼容性
        compatible = verify_memos_compatibility()
  
        # 总结
        total_time = time.time() - total_start
        print("\n" + "=" * 50)
        print("直接批量导入完成总结")
        print("=" * 50)
        print(f"✅ 总耗时: {total_time/60:.1f}分钟")
        print(f"📊 导入统计:")
        print(f"  节点: {node_count:,}")
        print(f"  边: {edge_count:,}")
        print(f"  MemOS兼容性: {'✅ 完全兼容' if compatible else '⚠️ 需要调整'}")
  
        if node_count > 0:
            print("\n💡 现在可以使用MemOS的所有功能:")
            print("  - 语义搜索")
            print("  - 图查询")
            print("  - 记忆推理")
            print("  - 可视化")
          
    except KeyboardInterrupt:
        print("\n❌ 用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序执行错误: {e}")
    finally:
        # 确保关闭数据库连接
        config.close_driver()
        print("🔒 数据库连接已关闭")

if __name__ == "__main__":
    main()
```

#### **在MemOS中挂载MemCube**

当数据成功导入后，我们的心血管MemCube便正式“上线”。在应用程序中，只需通过一个指向数据库的配置文件来初始化MemOS的 `TreeTextMemory`。之后，我们就可以通过这个 `tree_memory`对象与庞大的知识库进行交互，为AI赋予专业的领域记忆。

```python
#挂载memcube
from memos.configs.memory import TreeTextMemoryConfig
from memos.memories.textual.tree import TreeTextMemory

# 1. 参考挂载memcube所需的config文件示例
config_data = {
    "extractor_llm": {
        "backend": "huggingface",
        "config": {
            "model_name_or_path": "/mnt/public/model/huggingface/Qwen2.5-14B",
            "temperature": 0.1,
            "remove_think_prefix": True,
            "max_tokens": 8192
        }
    },
    "dispatcher_llm": {
        "backend": "huggingface",
        "config": {
            "model_name_or_path": "/mnt/public/model/huggingface/Qwen3-0.6B",
            "temperature": 0.1,
            "remove_think_prefix": True,
            "max_tokens": 8192
        }
    },
    "embedder": {
        "backend": "sentence_transformer",
        "config": {
            "model_name_or_path": "your embedding model path"
        }
    },
    "graph_db": {
        "backend": "neo4j",
        "config": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "yourpassword",
            "db_name": "neo4j",
            "auto_create": False,
            "embedding_dimension": 768
        }
    }
}

# 2. 写入 JSON 文件
json_path = "cookbooktest/tree_config.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(config_data, f, indent=2, ensure_ascii=False)

print(f"配置文件已生成: {json_path}")

# 3. 读取配置并初始化 TreeTextMemory
config = TreeTextMemoryConfig.from_json_file(json_path)
tree_memory = TreeTextMemory(config)

```

---

## **效果评估报告：MemCube记忆增强框架性能验证**

### **评估方法**

为了量化心血管MemCube带来的性能提升，我们构建了一套自动化的评估流程。该流程的核心是：利用一个强大的第三方模型（如Gemini-2.5-Pro）作为中立的“考官”，来检验装备了MemCube的模型在解决专业问题能力上的提升。

#### 评测考题示例

```python
# questions (partial in 200 questions)
 'A 65-year-old male patient presents to the emergency department with severe lower abdominal pain, inability to urinate, and is complaining of lightheadedness and palpitations. His past medical history includes hypertension controlled with lisinopril, benign prostatic hyperplasia for which he has been taking tamsulosin, and moderate alcohol use. On examination, his heart rate is elevated at 105 beats per minute, blood pressure is 140/90 mmHg, and he appears uncomfortable. Palpation reveals a distended bladder. An ECG shows sinus tachycardia without ischemic changes. You suspect bladder distension might be causing autonomic reflex changes affecting cardiac function. Considering this scenario, explain the physiological mechanism by which bladder distension might result in cardiac symptoms, and outline your approach to managing this patient to resolve both the urinary and cardiovascular concerns.',
 'In an elderly patient with poorly controlled diabetes, how do advanced glycation end products (AGEs) contribute to the pathophysiology of endothelial injury, and what implications does this have for the management of cardiovascular risks?',
 'A middle-aged patient with venous insufficiency is monitored using transcutaneous oxygen measurements to assess tissue perfusion. How does venous insufficiency affect transcutaneous oxygen levels, and how should these results influence treatment decisions for skin ulcers?',
 'A young adult with confirmed tubular acidosis is presenting with significant metabolic acidosis. How would sodium bicarbonate therapy be utilized in this case, and what are the considerations for its dosages and effects?'
```

```python
 
#### 核心检索流程实现
# 下面的脚本展示了基于MemCube的检索步骤：用`tree_memory.search()`方法。精准地从我们庞大的MemCube中，找出与当前考题语义最相似的知识片段。您也可后续配置chat模型，从而直接实现对话的功能。
# question_list: 用户提供的待检索相似问题的列表。例如：['What are the signs of a myocardial infarction?', 'What are the potential dangers of high blood pressure?']
question_list = ['What are the signs of a myocardial infarction?', 'What are the potential dangers of high blood pressure?']
search_results_dict = {}

for i, question in enumerate(question_list):
    print(i)
    results = tree_memory.search(question, top_k=15)
    # 排除长度短的纯concept节点
    filtered_results = [node.memory for node in results if len(node.memory) > 100]
    search_results_dict[i] = {
        'question': question,
        'results': filtered_results
    }
```

---

# 🧠 MemCube记忆操作系统医学AI效果评估报告

## 📋 执行摘要

基于强模型客观评判的200个医学案例对比分析，本报告全面评估了MemCube在医学AI应用中的效果。MemCube通过MemOS构建的领域知识pipeline，显著提升了AI模型的医学推理能力。

### **核心对比结果与分析**

#### **直接胜负统计**

| 对比配置                             | MemCube增强版胜 | Baseline基础版胜 | 平局 |
| ------------------------------------ | :-------------: | :--------------: | :--: |
| **7B模型内部对比**             |  **47**  |   **3**   | 150 |
| **32B模型内部对比**            |  **92**  |   **0**   | 108 |
| **7B+MemCube vs 32B Baseline** |  **57**  |   **3**   | 140 |

#### **关键洞察分析**

1. **对于大模型，精确的领域知识依然至关重要**：结果表明，为32B参数模型配备MemCube后，其性能表现出压倒性优势（92胜0败）。这证实了即使对于已有较强基础能力的大模型，一个结构化的外部领域知识库依然能带来决定性的性能飞跃，特别是在要求知识精确性的专业领域。
2. **记忆系统实现“以小博大”**：在7B模型与32B基础模型的跨级对比中，配备了MemCube的7B模型取得了57胜3败的显著优势。这清晰地证明，一个设计精良的记忆系统能够有效弥补小模型在知识储备上的不足，使其在特定领域的表现超越比自己大数倍的通用模型。
3. **记忆系统具有普遍有效性**：无论是7B还是32B模型，在装备MemCube后，其回答的专业性和准确性均获得显著提升。这表明本章提出的“概念图谱-知识蒸馏-记忆增强”框架是一种具有普遍适用性的、高效的AI能力提升方案。

### **代表性案例深入分析**

为了更直观地理解MemCube的作用机制，我们对部分评测案例进行深入分析。

#### **案例分析 1 (7B模型, ID: 9): Retinopathy of Prematurity (ROP) 案例**

**临床问题**: 涉及一名患有早产儿视网膜病变（ROP）的6岁儿童，因其眼底微血管变化，被转诊至心血管科进行全身性评估。问题要求评估应进行哪些心血管检查，以及眼底发现如何影响心血管评估策略。

**临床问题**:

```
A 6-year-old child is referred to your cardiovascular clinic for evaluation after being diagnosed with retinopathy of prematurity (ROP). The ophthalmologist has noted significant retinal vessel changes and has expressed concerns regarding the child's cardiovascular health given the potential systemic implications of microvascular changes observed in ROP. As you assess the child, you consider possible cardiovascular complications that may arise. What cardiovascular assessments would be pertinent in this case, and how might the existing retinal findings influence your cardiovascular evaluation strategy, particularly concerning microvascular health and its potential systemic effects?
```

**MemCube增强回答**:

```
{'model': 'qwen2.5-7b-instruct', 'version': 'v2', 'question_id': 9, 'prompt': "As an experienced medical expert, provide comprehensive analysis using evidence-based medicine principles.\n\n**CLINICAL CASE:**\nA 6-year-old child is referred to your cardiovascular clinic for evaluation after being diagnosed with retinopathy of prematurity (ROP). The ophthalmologist has noted significant retinal vessel changes and has expressed concerns regarding the child's cardiovascular health given the potential systemic implications of microvascular changes observed in ROP. As you assess the child, you consider possible cardiovascular complications that may arise. What cardiovascular assessments would be pertinent in this case, and how might the existing retinal findings influence your cardiovascular evaluation strategy, particularly concerning microvascular health and its potential systemic effects?\n\n**KEY EVIDENCE:**\n• Question: A young child presents with suspected retinopathy of prematurity, potentially linked to a congenital heart defect that has led to inconsistent oxygen delivery. As a cardiovascular specialist, how would you approach the management of this child's systemic condition to optimize retinal health?\n\nReasoning Guidance: Evaluate the impact of the congenital heart defect on systemic oxygenation. Consider the role of oxygen supplementation and monitoring. Integrate cardiovascular management strategies with ophthalmologic treatment to optimize retinal health.\n\nKnowledge Facts: Pediatric retinal disorders often involve insufficient retinal vascular development.; Pediatric retinal disorders can be exacerbated by systemic oxygen imbalances, common in congenital heart defects.; Effective management of pediatric retinal disorder requires collaboration with ophthalmology and cardiology.\n\nAnswer: The management involves stabilizing systemic oxygen levels through correction of the heart defect, if feasible, and careful use of supplemental oxygen. Coordination with an ophthalmologist to monitor retinal changes and implement laser therapy or surgical interventions may be required.\n• Question: During a cardiovascular examination, a pediatric patient with coexisting retinal and cardiovascular disorders seems to have poor growth despite appropriate medical interventions. What could be the systemic implications of these concurrent conditions, and how should clinical decision-making address these concerns?\n\nReasoning Guidance: Integrate understanding of pediatric retinal disorder with the potential cardiovascular inefficiencies causing poor systemic circulation and growth delays. Consider multidisciplinary approaches for these intertwined issues, promoting comprehensive care strategies.\n\nKnowledge Facts: Pediatric retinal disorders and cardiac anomalies can have overlapping pathogenic mechanisms affecting systemic development.; A comprehensive clinical approach involves assessing the interplay between circulatory efficiency and ocular vascular health.; Addressing underlying cardiovascular inefficiencies may relieve secondary complications impacting systemic development.\n\nAnswer: The clinical approach should prioritize optimization of cardiovascular function to improve circulation efficiencies, potentially benefiting retinal health and promoting growth. Collaboration across specialties, including cardiology, ophthalmology, and pediatrics, is crucial for comprehensive systemic management.\n• Question: A young adult with a history of pediatric retinal disorder secondary to Kawasaki disease is undergoing cardiovascular follow-up for potential coronary artery complications. How can ongoing retinal issues influence cardiovascular management?\n\nReasoning Guidance: Assess how retinal issues, such as impaired visual acuity or peripheral vision loss, might affect compliance or understanding of cardiovascular treatment plans. Consider unique strategies in communication or monitoring to enhance treatment adherence.\n\nKnowledge Facts: Pediatric retinal disorders can result from inflammatory conditions like Kawasaki disease impacting vasculature.; In patients with persistent retinal disorder consequences, visual limitations could obstruct understanding of complex medical instructions.; Cardiovascular management must adapt to individual patient's sensory and learning capabilities, ensuring thorough education about treatment and prevention strategies.\n\nAnswer: Management may require tailored communication strategies, ensuring the patient fully understands and is capable of following the prescribed cardiovascular treatment. Regular check-ups should integrate assessments of retinal status to anticipate any complications affecting routine cardiovascular care.\n• Question: A premature infant is undergoing ophthalmic screening for retinopathy of prematurity (ROP). What complications, if untreated, might arise related to the retina?\n\nReasoning Guidance: Understand how untreated ROP can lead to tractional retinal detachment in premature infants.\n\nKnowledge Facts: Retinopathy of prematurity involves abnormal retinal vessel development.; Advanced ROP can result in retinal traction and detachment.; Screening and early treatment are critical in preventing detachment.\n\nAnswer: Regular screening and timely intervention, such as laser photocoagulation, are essential to prevent retinal detachment.\n\nPlease provide evidence-based medical analysis and management recommendations.", 'response': "### Comprehensive Analysis and Management Recommendations for a Child with Retinopathy of Prematurity (ROP) and Potential Cardiovascular Complications\n\n#### Clinical Case Overview\nA 6-year-old child has been referred to a cardiovascular clinic for evaluation due to a diagnosis of retinopathy of prematurity (ROP). The ophthalmologist has noted significant retinal vessel changes and expressed concerns about potential systemic implications, particularly regarding microvascular health and its potential systemic effects.\n\n#### Key Evidence and Reasoning\n\n1. **Impact of Congenital Heart Defect on Systemic Oxygenation:**\n   - **Congenital Heart Defects and ROP:** Congenital heart defects can lead to inconsistent oxygen delivery, which can affect retinal vascular development. The retina is highly sensitive to oxygen levels, and inadequate oxygenation can result in ROP.\n   - **Management Strategy:**\n     - **Correcting the Heart Defect:** If the congenital heart defect is correctable, surgical intervention should be considered to improve systemic oxygenation.\n     - **Supplemental Oxygen:** In the interim, supplemental oxygen can be used to optimize oxygen levels, but it should be carefully monitored to avoid hyperoxia, which can also be detrimental.\n     - **Collaboration with Ophthalmology:** Regular ophthalmologic follow-up is essential to monitor retinal changes and intervene with laser therapy or surgical interventions if necessary.\n\n2. **Systemic Implications of Concurrent Conditions:**\n   - **Poor Growth and Circulatory Efficiency:** Poor growth in a child with both retinal and cardiovascular disorders may be due to systemic circulatory inefficiencies. This can be exacerbated by underlying cardiovascular anomalies.\n   - **Multidisciplinary Approach:**\n     - **Cardiovascular Optimization:** Focus on optimizing cardiovascular function to improve circulation efficiency. This may include pharmacological interventions, lifestyle modifications, and, if necessary, surgical interventions.\n     - **Comprehensive Care:** Collaboration between cardiology, ophthalmology, and pediatrics is essential to address the interplay between circulatory efficiency and ocular vascular health.\n\n3. **Ongoing Retinal Issues and Cardiovascular Management:**\n   - **Visual Limitations and Treatment Adherence:** Retinal issues, such as impaired visual acuity or peripheral vision loss, can affect a patient's ability to understand and adhere to complex medical instructions.\n   - **Tailored Communication Strategies:**\n     - **Simplified Instructions:** Use clear and simple language to ensure the patient or caregiver fully understands the treatment plan.\n     - **Visual Aids and Educational Materials:** Provide visual aids and educational materials to enhance understanding and compliance.\n     - **Regular Follow-ups:** Schedule regular check-ups to monitor both cardiovascular and retinal health, ensuring timely intervention if any complications arise.\n\n4. **Complications of Untreated ROP:**\n   - **Tractional Retinal Detachment:** Advanced ROP can lead to retinal traction and detachment, which can result in severe visual impairment or blindness.\n   - **Screening and Early Intervention:**\n     - **Regular Ophthalmic Screening:** Implement regular ophthalmic screening for premature infants to detect ROP early.\n     - **Timely Intervention:** Promptly intervene with laser photocoagulation or other treatments to prevent retinal detachment.\n\n### Evidence-Based Medical Analysis and Management Recommendations\n\n1. **Initial Assessment:**\n   - **Cardiovascular Examination:** Perform a thorough cardiovascular examination, including echocardiography, to assess the presence and severity of any congenital heart defects.\n   - **Ophthalmologic Evaluation:** Conduct a comprehensive ophthalmologic examination to assess the extent of retinal changes and determine the stage of ROP.\n\n2. **Management Plan:**\n   - **Correct Congenital Heart Defects:** If the heart defect is correctable, arrange for surgical intervention to improve systemic oxygenation.\n   - **Supplemental Oxygen Therapy:** Use supplemental oxygen cautiously to optimize oxygen levels, but monitor for hyperoxia.\n   - **Collaborative Care:** Establish a multidisciplinary care team including cardiology, ophthalmology, and pediatrics to coordinate treatment and monitoring.\n   - **Growth Monitoring:** Regularly monitor the child's growth and development to address any systemic circulatory inefficiencies.\n   - **Communication Strategies:** Use simplified instructions and visual aids to enhance treatment adherence, especially if the child has visual limitations.\n   - **Ophthalmic Screening:** Implement regular ophthalmic screening for premature infants to detect ROP early and intervene promptly.\n\n3. **Monitoring and Follow-ups:**\n   - **Regular Follow-ups:** Schedule regular follow-ups with both the cardiologist and ophthalmologist to monitor the child's cardiovascular and retinal health.\n   - **Adjustments to Treatment:** Be prepared to adjust the treatment plan based on the child's response to interventions and any new findings.\n\nBy following these evidence-based recommendations, the healthcare team can optimize the child's cardiovascular and retinal health, ensuring the best possible outcomes for the patient.", 'length': 5313, 'api_time': 89.44595098495483, 'timestamp': 1753351860.232497, 'error': False}
```

**Baseline回答**:

```
{'model': 'qwen2.5-7b-instruct', 'version': 'baseline', 'question_id': 9, 'prompt': "Analyze this clinical case as a senior medical expert:\n\nA 6-year-old child is referred to your cardiovascular clinic for evaluation after being diagnosed with retinopathy of prematurity (ROP). The ophthalmologist has noted significant retinal vessel changes and has expressed concerns regarding the child's cardiovascular health given the potential systemic implications of microvascular changes observed in ROP. As you assess the child, you consider possible cardiovascular complications that may arise. What cardiovascular assessments would be pertinent in this case, and how might the existing retinal findings influence your cardiovascular evaluation strategy, particularly concerning microvascular health and its potential systemic effects?\n\nPlease provide detailed medical analysis and recommendations.", 'response': "In this clinical case, a 6-year-old child with retinopathy of prematurity (ROP) presents with significant retinal vessel changes, raising concerns about potential systemic implications. As a senior medical expert, I would recommend a comprehensive cardiovascular evaluation to assess for any underlying cardiovascular complications that may be related to the microvascular changes observed in the retina. Here is a detailed analysis and a recommended evaluation strategy:\n\n### 1. **Medical History and Physical Examination**\n- **Medical History:**\n  - **Perinatal History:** Obtain detailed information about the child's birth, including gestational age, birth weight, and any complications during delivery.\n  - **Family History:** Inquire about any cardiovascular or ocular diseases in the family.\n  - **Past Medical History:** Review any previous medical conditions, surgeries, or treatments.\n  - **Current Medications:** Document any medications the child is currently taking.\n- **Physical Examination:**\n  - **General Appearance:** Assess for signs of prematurity or other systemic conditions.\n  - **Cardiovascular Examination:**\n    - **Heart Sounds:** Auscultate for any murmurs, gallops, or rubs.\n    - **Peripheral Vascular Signs:** Check for signs of hypertension, such as brachial artery blood pressure differences, or signs of coarctation of the aorta.\n    - **Capillary Refill Time:** Evaluate for any signs of poor perfusion.\n\n### 2. **Cardiovascular Assessments**\nGiven the potential systemic implications of microvascular changes observed in ROP, the following cardiovascular assessments are pertinent:\n\n#### **A. Blood Pressure Measurement**\n- **Non-Invasive Measurement:** Measure blood pressure in both arms to detect any differences that might suggest coarctation of the aorta.\n- **Ambulatory Blood Pressure Monitoring (ABPM):** Consider this if there are any signs of hypertension or if the child is symptomatic.\n\n#### **B. Echocardiography**\n- **Echocardiogram:** This is the gold standard for evaluating cardiac structure and function. It can help identify any congenital heart defects, valvular abnormalities, or other structural heart issues that might be related to the microvascular changes.\n- **Fetal Echocardiography:** If the child was born prematurely, a fetal echocardiogram might have been performed, which could provide additional information.\n\n#### **C. Cardiac MRI or CT**\n- **Imaging Studies:** If there are specific concerns about complex cardiac anatomy or if echocardiography is inconclusive, cardiac MRI or CT can provide detailed images of the heart and great vessels.\n\n#### **D. Cardiovascular Risk Factors**\n- **Fasting Lipid Profile:** Assess for dyslipidemia, which can be a risk factor for cardiovascular disease.\n- **Fasting Blood Glucose:** Evaluate for diabetes or impaired glucose tolerance, which can also contribute to cardiovascular risk.\n- **Anthropometric Measurements:** Measure height, weight, and body mass index (BMI) to assess for obesity, which is a significant risk factor for cardiovascular disease.\n\n### 3. **Ophthalmologic and Microvascular Health Considerations**\n- **Retinal Changes:** The significant retinal vessel changes in ROP suggest microvascular abnormalities. These changes are often associated with systemic conditions such as hypertension, diabetes, and other vascular disorders.\n- **Systemic Evaluation:** Given the systemic implications, a thorough evaluation for other microvascular diseases should be considered, including:\n  - **Fundoscopy:** Regular follow-up fundoscopy to monitor for progression of ROP.\n  - **Retinal Vessel Fluorescein Angiography:** To assess the integrity of the retinal vasculature and identify any areas of leakage or ischemia.\n  - **Systemic Blood Pressure Monitoring:** Regular monitoring of blood pressure to detect any hypertension.\n  - **Glucose Tolerance Testing:** To screen for diabetes or impaired glucose tolerance.\n\n### 4. **Recommendations**\n- **Referral to a Pediatric Cardiologist:** Given the potential systemic implications, it is crucial to refer the child to a pediatric cardiologist for a comprehensive cardiovascular evaluation.\n- **Regular Follow-Up:** Schedule regular follow-up visits to monitor the child's cardiovascular health and retinal status.\n- **Lifestyle Modifications:** If any risk factors are identified, implement lifestyle modifications such as a healthy diet, regular exercise, and weight management.\n- **Genetic Counseling:** Consider genetic counseling if there is a family history of cardiovascular or ocular diseases.\n\n### Conclusion\nThe microvascular changes observed in ROP suggest a potential systemic vascular disease. A comprehensive cardiovascular evaluation, including echocardiography, blood pressure monitoring, and other relevant tests, is essential to identify any underlying cardiovascular complications. Regular follow-up and monitoring are crucial to ensure early detection and management of any potential issues.", 'length': 4977, 'api_time': 92.66847014427185, 'timestamp': 1753351863.4531698, 'error': False}
```

**案例解析**:
在此案例中，基础模型（回答B）给出了一个常规、标准的临床评估流程，涵盖了病史询问、体格检查和各类心脏检查。然而，其回答未能深入探讨ROP与心血管系统之间内在的病理生理联系。

相比之下，MemCube增强后的模型（回答A）展现了更高层次的临床推理能力。它能够调用并整合多个相关的知识片段（这些知识来源于我们为“ROP”、“先天性心脏病”、“系统性氧合”等概念蒸馏的QA记忆），例如：

* 明确指出了“先天性心脏病可能导致不稳定的氧气输送，从而影响视网膜血管发育”。
* 强调了优化“系统性循环效率”对改善眼部血管健康和全身发育的潜在益处。

这体现了MemCube的核心优势：它不仅仅是简单的信息检索，而是将离散的知识点在推理时进行有效关联，形成了一个多角度、更具深度的分析框架。这使得增强后的模型能够像资深专家一样，从病因层面探讨问题，并提出跨学科的综合管理策略，而不仅仅是罗列检查项目。

---

### 32B模型MemCube效果评估

#### 案例 1: 医学问答对比 (ID: 146)

**临床问题**: 针对一名已确诊为特发性扩张型心肌病（DCM）并出现心悸、头晕症状的成年患者，询问心律失常的存在如何影响其诊断和治疗策略。
**临床问题**:

```
An adult patient with known idiopathic dilated cardiomyopathy presents with palpitations and dizziness. How does the presence of arrhythmias influence your diagnostic and therapeutic approach, especially in the context of managing dilated cardiomyopathy?
```

**MemCube增强回答**:

```
{'model': 'qwen2.5-32b-instruct', 'version': 'v2', 'question_id': 146, 'prompt': 'As an experienced medical expert, provide comprehensive analysis using evidence-based medicine principles.\n\n**CLINICAL CASE:**\nAn adult patient with known idiopathic dilated cardiomyopathy presents with palpitations and dizziness. How does the presence of arrhythmias influence your diagnostic and therapeutic approach, especially in the context of managing dilated cardiomyopathy?\n\n**KEY EVIDENCE:**\n• Question: A middle-aged patient diagnosed with idiopathic dilated cardiomyopathy presents with palpitations and dizziness. Considering the risk of proarrhythmia, what diagnostic strategies and management plans should be considered?\n\nReasoning Guidance: Evaluate the role of idiopathic dilated cardiomyopathy in altering cardiac electrophysiology, leading to arrhythmic complications. Discuss the impact of heart failure medications on arrhythmia risk and selection of antiarrhythmic drugs fostering minimal proarrhythmic potential.\n\nKnowledge Facts: Idiopathic dilated cardiomyopathy can lead to heart chamber enlargement affecting electrical conduction.; Proarrhythmia refers to the increased risk of arrhythmias caused by medications or cardiac conditions.; Monitoring with ECG and considering beta-blocker or anticoagulant therapy are key in management.\n\nAnswer: Given the history of idiopathic dilated cardiomyopathy, the patient should be monitored closely with ECG for arrhythmic patterns. Opt for rhythm-stabilizing medications like beta-blockers while avoiding drugs with high proarrhythmic potential.\n• Question: An adult presents with palpitations and a recent diagnosis of idiopathic dilated cardiomyopathy. How should the presence of frequent atrial premature beats influence the clinical management of this patient?\n\nReasoning Guidance: Evaluate how atrial arrhythmias can exacerbate heart failure symptoms and potential management strategies to mitigate this risk.\n\nKnowledge Facts: Idiopathic dilated cardiomyopathy can lead to heart failure symptoms.; Frequent atrial premature beats can worsen cardiac function.; Managing arrhythmias may improve heart failure control.\n\nAnswer: Focus on optimizing heart failure management and consider treatment options for arrhythmias, such as beta-blockers or antiarrhythmic drugs.\n• Question: A young adult has been diagnosed with idiopathic dilated cardiomyopathy and is experiencing palpitations. Analyze how idiopathic dilated cardiomyopathy can cause palpitations and determine an appropriate treatment strategy.\n\nReasoning Guidance: Palpitations in dilated cardiomyopathy could indicate arrhythmias. Evaluate cardiac function and rhythm, using diagnostics to determine arrhythmia presence and guide treatment such as antiarrhythmics or device therapy.\n\nKnowledge Facts: Idiopathic dilated cardiomyopathy can lead to arrhythmias causing palpitations.; Palpitations may reflect arrhythmic risks in cardiomyopathy due to ventricular dysfunction.; Evaluating arrhythmia type and severity is essential in managing palpitations and potential heart failure exacerbations.\n\nAnswer: Investigate arrhythmia presence using ECG, consider beta-blockers or anti-arrhythmic medication, and evaluate need for device therapy like ICD.\n• Question: A young adult with idiopathic dilated cardiomyopathy presents with arrhythmias. How might disruptions in cardiac action potentials contribute to this presentation, and what treatment options should be considered?\n\nReasoning Guidance: Understand the mechanics of cardiac action potentials and how disruptions can lead to arrhythmias in dilated cardiomyopathy. Analyze therapeutic strategies targeting electrophysiological abnormalities.\n\nKnowledge Facts: Action potentials are crucial for coordinated cardiac contraction.; Idiopathic dilated cardiomyopathy can alter cardiac electrical activity.; Arrhythmias in cardiomyopathy may result from ion channel dysfunction or structural changes.\n\nAnswer: Altered cardiac action potentials due to cardiomyopathy can cause electrical instability, resulting in arrhythmias. Management includes antiarrhythmic drugs, device therapy, and addressing underlying structural heart issues.\n\nPlease provide evidence-based medical analysis and management recommendations.', 'response': '### Clinical Case Analysis: Idiopathic Dilated Cardiomyopathy with Palpitations and Dizziness\n\n#### Background\nIdiopathic dilated cardiomyopathy (IDC) is characterized by ventricular dilation and impaired systolic function, often leading to heart failure. The presence of arrhythmias in IDC is common and can exacerbate symptoms such as palpitations and dizziness, potentially leading to more severe complications like syncope or sudden cardiac death.\n\n#### Diagnostic Approach\n1. **Electrocardiogram (ECG) Monitoring:**\n   - **Initial ECG:** To identify any immediate arrhythmias, such as atrial fibrillation, ventricular tachycardia, or frequent premature beats.\n   - **Holter Monitoring:** For 24-48 hours to capture intermittent arrhythmias.\n   - **Event Monitor:** For patients with infrequent symptoms, to record arrhythmias as they occur.\n   - **Implantable Loop Recorder:** For patients with recurrent symptoms and negative initial monitoring.\n\n2. **Echocardiography:**\n   - To assess ventricular function, chamber dimensions, and wall motion abnormalities.\n   - To evaluate for any thrombus formation, which could be a source of embolic events.\n\n3. **Blood Tests:**\n   - Cardiac biomarkers (e.g., BNP, NT-proBNP) to assess heart failure severity.\n   - Thyroid function tests, as hyperthyroidism can cause palpitations.\n   - Electrolyte levels, particularly potassium and magnesium, which can affect cardiac rhythm.\n\n#### Therapeutic Approach\n1. **Optimization of Heart Failure Therapy:**\n   - **Beta-Blockers:** Proven to reduce mortality and hospitalization in IDC. They also stabilize the heart rhythm, reducing the risk of arrhythmias.\n   - **ACE Inhibitors/ARBs:** To reduce afterload and improve ventricular function.\n   - **Diuretics:** To manage fluid overload and alleviate symptoms of heart failure.\n   - **SGLT2 Inhibitors:** Emerging evidence supports their use in reducing cardiovascular events in heart failure patients.\n\n2. **Antiarrhythmic Therapy:**\n   - **Class I Antiarrhythmics (e.g., flecainide, propafenone):** Used cautiously due to potential proarrhythmic effects.\n   - **Class III Antiarrhythmics (e.g., amiodarone, sotalol):** Amiodarone is effective but has significant side effects; sotalol is a beta-blocker with antiarrhythmic properties.\n   - **Calcium Channel Blockers (e.g., verapamil, diltiazem):** Useful for certain arrhythmias like atrial fibrillation, but use with caution due to negative inotropic effects.\n\n3. **Device Therapy:**\n   - **Implantable Cardioverter Defibrillator (ICD):** Indicated in patients with IDC and reduced ejection fraction (EF <35%) to prevent sudden cardiac death.\n   - **Pacemakers:** Considered in patients with bradyarrhythmias or those requiring resynchronization therapy (CRT).\n\n4. **Lifestyle Modifications:**\n   - Smoking cessation, alcohol moderation, and a low-sodium diet.\n   - Regular, moderate-intensity exercise as tolerated.\n   - Stress management techniques to reduce arrhythmia triggers.\n\n#### Proarrhythmic Risk and Management\n- **Proarrhythmic Potential:** Antiarrhythmic drugs can paradoxically increase the risk of arrhythmias, particularly in patients with structural heart disease. Careful selection and monitoring are essential.\n- **Risk Stratification:** Use risk stratification tools like the CHA2DS2-VASc score for thromboembolic risk and the European Society of Cardiology guidelines for ICD implantation.\n\n#### Conclusion\nIn managing a patient with IDC and arrhythmias, a comprehensive approach is necessary, combining optimal heart failure therapy with targeted antiarrhythmic treatment and device therapy as indicated. Close monitoring and patient education are crucial to prevent complications and improve quality of life.', 'length': 3725, 'api_time': 18.48910903930664, 'timestamp': 1753352210.820965, 'error': False}
```

**Baseline回答**:

```
{'model': 'qwen2.5-32b-instruct', 'version': 'baseline', 'question_id': 146, 'prompt': 'Analyze this clinical case as a senior medical expert:\n\nAn adult patient with known idiopathic dilated cardiomyopathy presents with palpitations and dizziness. How does the presence of arrhythmias influence your diagnostic and therapeutic approach, especially in the context of managing dilated cardiomyopathy?\n\nPlease provide detailed medical analysis and recommendations.', 'response': "In the context of an adult patient with known idiopathic dilated cardiomyopathy (DCM) presenting with palpitations and dizziness, the presence of arrhythmias significantly influences the diagnostic and therapeutic approach. Here is a detailed analysis and recommendations based on current medical practices:\n\n### Clinical Context and Initial Assessment\n- **Idiopathic Dilated Cardiomyopathy**: DCM is characterized by ventricular dilation and impaired systolic function, often leading to heart failure and arrhythmias.\n- **Symptoms**: Palpitations and dizziness are common symptoms that can be indicative of arrhythmias, which are frequent complications in DCM.\n- **Risk Factors**: The patient's history of DCM places them at higher risk for arrhythmias, particularly atrial fibrillation (AF), ventricular tachycardia (VT), and bradyarrhythmias.\n\n### Diagnostic Approach\n1. **History and Physical Examination**: Detailed history to understand the onset, duration, and triggers of palpitations and dizziness. Physical examination should focus on signs of heart failure, such as jugular venous distension, rales, and peripheral edema.\n2. **Electrocardiogram (ECG)**: Essential for detecting arrhythmias. Can identify AF, VT, or other conduction abnormalities.\n3. **Holter Monitoring**: Useful for patients with intermittent symptoms to capture arrhythmias that may not be evident on a standard ECG.\n4. **Echocardiography**: To assess ventricular function, size, and potential thrombus formation, especially if AF is suspected.\n5. **Cardiac MRI**: Provides detailed images of the heart structure and function, which can be crucial in assessing the extent of DCM and ruling out other causes of cardiomyopathy.\n6. **Blood Tests**: Including electrolytes, thyroid function tests, and markers of heart failure (BNP/NT-proBNP).\n\n### Therapeutic Approach\n1. **Management of Arrhythmias**:\n   - **Atrial Fibrillation**: If diagnosed, rate control or rhythm control strategies should be considered. Rate control can be achieved with beta-blockers or non-dihydropyridine calcium channel blockers. Rhythm control might involve antiarrhythmic drugs or catheter ablation.\n   - **Ventricular Tachycardia**: If VT is suspected or confirmed, antiarrhythmic drugs (e.g., amiodarone) or implantable cardioverter-defibrillator (ICD) may be necessary. Catheter ablation can also be considered.\n   - **Bradyarrhythmias**: If bradyarrhythmias are present, a pacemaker may be indicated.\n\n2. **Management of Dilated Cardiomyopathy**:\n   - **Heart Failure Medications**: Continue or initiate medications such as ACE inhibitors/ARBs, beta-blockers, aldosterone antagonists, and diuretics as needed.\n   - **Lifestyle Modifications**: Encourage a heart-healthy diet, regular physical activity, and smoking cessation.\n   - **Monitoring**: Regular follow-up with echocardiography and clinical assessment to monitor disease progression and response to therapy.\n\n3. **Anticoagulation**:\n   - If AF is present, anticoagulation therapy is necessary to prevent thromboembolic events. The choice of anticoagulant (e.g., warfarin, direct oral anticoagulants) should be based on the patient's risk factors and comorbidities.\n\n### Prognosis and Long-Term Management\n- **Prognosis**: The presence of arrhythmias in DCM can worsen prognosis. Regular monitoring and timely intervention are crucial.\n- **Long-Term Management**: Focus on preventing arrhythmias and heart failure exacerbations. Consider multidisciplinary care involving cardiologists, electrophysiologists, and heart failure specialists.\n\n### Conclusion\nThe presence of arrhythmias in a patient with idiopathic dilated cardiomyopathy requires a comprehensive approach, including thorough diagnostic evaluation and tailored therapeutic interventions. The goal is to manage both the underlying cardiomyopathy and the arrhythmias to improve the patient's quality of life and prognosis.", 'length': 3902, 'api_time': 13.865102052688599, 'timestamp': 1753352205.427208, 'error': False}
```

**案例分析**:
基础的32B模型已经能够给出一个相当全面和准确的回答，覆盖了诊断流程、治疗方案和预后管理。

然而，MemCube增强后的模型在回答的**结构性、逻辑性和细节深度**上更胜一筹。通过检索和整合MemCube中关于“DCM”、“心律失常”、“致心律失常风险”等概念的QA记忆，回答A呈现出以下特点：

* **逻辑层次更清晰**：回答A将诊断和治疗方法分门别类，如将抗心律失常药物按类别（Class I, Class III等）进行阐述，并明确指出了在DCM背景下使用它们的注意事项（如致心律失常风险）。
* **关键知识点突出**：回答A明确地将“优化心衰治疗”作为处理心律失常的基础，并列举了β受体阻滞剂等药物在稳定心律方面的双重作用。这些都是来源于MemCube中强调临床实践关键点的QA记忆。
* **更强的风险意识**：回答A专门设有“致心律失常风险和管理”一节，这显示出模型不仅仅在陈述知识，更在模仿专家的风险评估思维。

这个案例表明，即使对于强大的32B模型，MemCube依然能起到"知识教练"的作用，帮助模型将庞杂的内部知识以更结构化、更符合临床逻辑的方式组织和表达出来，从而提供更具指导价值的专业建议。

---

## 🚀 实践体验：试用心血管医学 MemCube 演示

本章所展示的心血管医学知识问答系统已经构建完成，并提供了一个包含 **211,315 个记忆条目** 和 **522,368 个语义关联** 的完整 MemCube 演示版本。

### 📦 演示系统特点

- **🫀 专业领域**：心血管医学知识体系
- **📊 数据规模**：211,315 个高质量记忆条目
- **🔗 关联网络**：522,368 个概念间语义连接  
- **💾 数据大小**：约 5.0GB 结构化医学知识
- **🤖 AI 支持**：支持多种 LLM 模型（GPT-4o、Claude、本地模型等）
- **🌐 部署就绪**：基于 Neo4j + MemOS 的生产级架构

### 🔍 立即体验

想要亲手体验本章介绍的完整构建流程和最终效果？您可以访问我们的演示项目：

**👉 [Cardio MemCube Demo - Hugging Face](https://huggingface.co/datasets/MemCube/cardio-memcube-demo)**

该演示项目提供：
- ✅ **完整的安装指南**：一键部署心血管 MemCube 系统
- ✅ **可运行的代码示例**：直接体验知识问答功能
- ✅ **详细的技术文档**：了解构建方法论和最佳实践
- ✅ **多语言模型支持**：灵活配置不同的 AI 模型后端

### ⚠️ 重要说明

- **🏥 医学免责声明**：该演示仅用于技术展示和教育目的，不应作为医学诊断或治疗的依据
- **🌐 语言支持**：当前版本使用英文优化的嵌入模型，中文查询需要翻译或更换多语言嵌入模型
- **🔧 技术框架**：这是一个可适用于任何专业领域的技术参考实现

通过实际体验这个演示系统，您将更好地理解如何将本章的理论方法转化为实际的生产级应用，并为构建您自己领域的 MemCube 系统积累宝贵经验。