"""
MemCube 政治理论概念图扩增系统 - 主程序入口
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from loguru import logger

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent))

from concept_analyzer import analyze_concepts_from_file
from concept_extractor import extract_concepts_from_analysis
from concept_graph import expand_concept_graph
from qa_generator import generate_political_theory_qa

def setup_logging(config):
    """设置日志"""
    from loguru import logger

    # 移除默认处理器
    logger.remove()

    # 添加控制台输出
    logger.add(
        sys.stdout,
        format=config['logging']['format'],
        level=config['logging']['level'],
        colorize=True
    )

    # 添加文件输出
    logs_dir = Path(config['paths']['logs_dir'])
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        logs_dir / "memcube_{time:YYYY-MM-DD}.log",
        format=config['logging']['format'],
        level=config['logging']['level'],
        rotation=config['logging']['rotation'],
        retention=config['logging']['retention'],
        encoding='utf-8'
    )

def validate_api_config():
    """验证API配置"""
    api_config_file = Path("config/api_keys.yaml")
    if not api_config_file.exists():
        logger.error("API配置文件不存在！")
        logger.info("请复制 config/api_keys.yaml.example 为 config/api_keys.yaml")
        logger.info("然后填入你的OpenAI API密钥")
        return False

    # 检查是否为示例文件
    with open(api_config_file, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'your-openai-api-key-here' in content:
            logger.error("请先配置API密钥！")
            logger.info("编辑 config/api_keys.yaml 文件，填入真实的API密钥")
            return False

    return True

def run_stage_concept_analysis(config):
    """运行第一阶段：概念分析"""
    logger.info("=" * 60)
    logger.info("开始第一阶段：种子概念思考分析")
    logger.info("=" * 60)

    seed_concepts_file = config['paths']['seed_concepts']
    if not Path(seed_concepts_file).exists():
        logger.error(f"种子概念文件不存在: {seed_concepts_file}")
        return None

    # 运行概念分析
    analysis_file = analyze_concepts_from_file(
        concepts_file=seed_concepts_file,
        config_file="config/config.yaml",
        batch_size=config['concept_expansion']['batch_size'],
        max_workers=config['concept_expansion']['max_workers']
    )

    logger.success(f"概念分析完成，结果文件: {analysis_file}")
    return analysis_file

def run_stage_concept_extraction(config):
    """运行第一阶段：概念提取"""
    logger.info("=" * 60)
    logger.info("开始第一阶段：从思考分析中提取概念")
    logger.info("=" * 60)

    # 查找最新的分析结果文件
    results_dir = Path(config['paths']['results_dir']) / "concept_analysis"
    analysis_files = list(results_dir.glob("concept_analysis_results.json"))

    if not analysis_files:
        logger.error("未找到概念分析结果文件！")
        logger.info("请先运行 --stage concept-analysis")
        return None

    analysis_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"使用分析结果文件: {analysis_file}")

    # 运行概念提取
    concepts_file = extract_concepts_from_analysis(
        analysis_file=str(analysis_file),
        config_file="config/config.yaml",
        batch_size=config['concept_expansion']['batch_size'],
        max_workers=config['concept_expansion']['max_workers']
    )

    logger.success(f"概念提取完成，结果文件: {concepts_file}")
    return concepts_file

def run_stage_concept_expansion(config):
    """运行第二阶段：概念图扩增"""
    logger.info("=" * 60)
    logger.info("开始第二阶段：概念图迭代扩增")
    logger.info("=" * 60)

    seed_concepts_file = config['paths']['seed_concepts']
    if not Path(seed_concepts_file).exists():
        logger.error(f"种子概念文件不存在: {seed_concepts_file}")
        return None

    # 运行概念图扩增
    graph_dir = expand_concept_graph(
        seed_concepts_file=seed_concepts_file,
        config_file="config/config.yaml"
    )

    logger.success(f"概念图扩增完成，结果目录: {graph_dir}")

    # 返回概念图文件路径
    graph_file = Path(graph_dir) / "final_concept_graph.json"
    return str(graph_file) if graph_file.exists() else None

def run_stage_qa_generation(config, graph_file=None):
    """运行第三阶段：QA生成"""
    logger.info("=" * 60)
    logger.info("开始第三阶段：QA知识生成")
    logger.info("=" * 60)

    if not graph_file:
        # 查找最新的概念图文件
        concept_graph_dir = Path(config['paths']['concept_graph_dir'])
        graph_files = list(concept_graph_dir.glob("final_concept_graph.json"))

        if not graph_files:
            logger.error("未找到概念图文件！")
            logger.info("请先运行 --stage concept-expansion")
            return None

        graph_file = max(graph_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"使用概念图文件: {graph_file}")

    # 运行QA生成
    result_summary = generate_political_theory_qa(
        concept_graph_file=graph_file,
        config_file="config/config.yaml"
    )

    logger.success("QA生成完成！")
    logger.info(f"生成统计:")
    logger.info(f"  - 总QA对数: {result_summary['after_filtering']}")
    logger.info(f"  - 单概念QA: {result_summary['generated_single_concept_qa']}")
    logger.info(f"  - 概念对QA: {result_summary['generated_concept_pair_qa']}")

    return result_summary

def run_full_pipeline(config):
    """运行完整流程"""
    logger.info("=" * 80)
    logger.info("开始 MemCube 政治理论概念图扩增完整流程")
    logger.info("=" * 80)

    # 验证API配置
    if not validate_api_config():
        return

    # 阶段1：概念图扩增（如果种子概念已经准备好，可以直接跳到概念扩增）
    logger.info("🚀 阶段1：概念图迭代扩增")
    graph_file = run_stage_concept_expansion(config)

    if not graph_file:
        logger.error("概念图扩增失败，流程终止")
        return

    # 阶段2：QA生成
    logger.info("🚀 阶段2：QA知识生成")
    qa_result = run_stage_qa_generation(config, graph_file)

    if not qa_result:
        logger.error("QA生成失败")
        return

    # 完成总结
    logger.info("=" * 80)
    logger.success("🎉 MemCube 完整流程执行完成！")
    logger.info("=" * 80)
    logger.info("生成的文件:")
    for output_file in qa_result.get('output_files', []):
        logger.info(f"  - {output_file}")

    logger.info("\n下一步:")
    logger.info("1. 检查生成的QA数据质量")
    logger.info("2. 可以将数据导入图数据库进行管理")
    logger.info("3. 基于生成的知识构建应用系统")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MemCube 政治理论概念图扩增系统")

    parser.add_argument(
        "--stage",
        choices=[
            "concept-analysis",     # 概念思考分析
            "concept-extraction",   # 概念提取
            "concept-expansion",     # 概念图扩增
            "qa-generation",         # QA生成
            "all"                   # 完整流程
        ],
        default="all",
        help="选择要运行的阶段"
    )

    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="配置文件路径"
    )

    args = parser.parse_args()

    # 加载配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return

    # 设置日志
    setup_logging(config)

    logger.info("MemCube 政治理论概念图扩增系统启动")
    logger.info(f"运行阶段: {args.stage}")
    logger.info(f"配置文件: {args.config}")

    # 验证API配置
    if args.stage != "concept-analysis":  # concept-analysis阶段可能不需要API
        if not validate_api_config():
            return

    # 根据选择的阶段运行
    try:
        if args.stage == "concept-analysis":
            run_stage_concept_analysis(config)
        elif args.stage == "concept-extraction":
            run_stage_concept_extraction(config)
        elif args.stage == "concept-expansion":
            run_stage_concept_expansion(config)
        elif args.stage == "qa-generation":
            run_stage_qa_generation(config)
        elif args.stage == "all":
            run_full_pipeline(config)

    except KeyboardInterrupt:
        logger.warning("用户中断执行")
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        logger.exception("详细错误信息:")

    logger.info("程序执行结束")

if __name__ == "__main__":
    main()