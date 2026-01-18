#!/usr/bin/env python3
"""
配置概念扩增参数
"""

import os
import yaml

def configure_expansion():
    """配置概念扩增参数"""
    print("=== 概念扩增参数配置 ===")
    print("\n当前可选配置:")
    print("1. 测试模式 - 只扩增1轮，保存到Neo4j")
    print("2. 快速模式 - 扩增3轮，保存到Neo4j")
    print("3. 标准模式 - 扩增10轮，收敛后停止")
    print("4. 自定义模式 - 自定义参数")
    print("5. 查看当前配置")
    print("0. 退出")

    while True:
        try:
            choice = input("\n请选择配置模式 (0-5): ").strip()

            if choice == "0":
                print("退出配置")
                return
            elif choice == "1":
                configure_test_mode()
                break
            elif choice == "2":
                configure_quick_mode()
                break
            elif choice == "3":
                configure_standard_mode()
                break
            elif choice == "4":
                configure_custom_mode()
                break
            elif choice == "5":
                show_current_config()
                continue
            else:
                print("无效选择，请输入0-5之间的数字")
        except KeyboardInterrupt:
            print("\n退出配置")
            return
        except Exception as e:
            print(f"配置失败: {e}")

def configure_test_mode():
    """测试模式配置"""
    config = {
        'max_iterations': 1,
        'batch_size': 5,
        'max_workers': 1,
        'auto_save_after_iteration': True,
        'save_to_neo4j_after_each_iteration': True,
        'stop_after_first_iteration': True
    }
    apply_config("测试模式", config)

def configure_quick_mode():
    """快速模式配置"""
    config = {
        'max_iterations': 3,
        'batch_size': 8,
        'max_workers': 2,
        'auto_save_after_iteration': True,
        'save_to_neo4j_after_each_iteration': True,
        'stop_after_first_iteration': False
    }
    apply_config("快速模式", config)

def configure_standard_mode():
    """标准模式配置"""
    config = {
        'max_iterations': 10,
        'batch_size': 15,
        'max_workers': 3,
        'auto_save_after_iteration': False,
        'save_to_neo4j_after_each_iteration': False,
        'stop_after_first_iteration': False
    }
    apply_config("标准模式", config)

def configure_custom_mode():
    """自定义模式配置"""
    print("\n=== 自定义配置 ===")

    config = {}

    # 获取迭代次数
    while True:
        try:
            max_iterations = int(input("最大迭代次数 (1-50): "))
            if 1 <= max_iterations <= 50:
                config['max_iterations'] = max_iterations
                break
            else:
                print("请输入1-50之间的数字")
        except ValueError:
            print("请输入有效的数字")

    # 获取批处理大小
    while True:
        try:
            batch_size = int(input("批处理大小 (1-100): "))
            if 1 <= batch_size <= 100:
                config['batch_size'] = batch_size
                break
            else:
                print("请输入1-100之间的数字")
        except ValueError:
            print("请输入有效的数字")

    # 获取并发数
    while True:
        try:
            max_workers = int(input("并发工作数 (1-10): "))
            if 1 <= max_workers <= 10:
                config['max_workers'] = max_workers
                break
            else:
                print("请输入1-10之间的数字")
        except ValueError:
            print("请输入有效的数字")

    # 其他配置
    config['auto_save_after_iteration'] = get_yes_no("每轮迭代后自动保存")
    config['save_to_neo4j_after_each_iteration'] = get_yes_no("每轮迭代后保存到Neo4j")
    config['stop_after_first_iteration'] = get_yes_no("第一轮后停止")

    apply_config("自定义模式", config)

def get_yes_no(prompt):
    """获取yes/no输入"""
    while True:
        answer = input(f"{prompt} (y/n): ").strip().lower()
        if answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False
        else:
            print("请输入 y/n 或 yes/no")

def apply_config(mode_name, config):
    """应用配置"""
    config_path = "config/config.yaml"

    # 备份原配置
    backup_path = config_path + f".backup_{int(time.time())}"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"✓ 原配置已备份到: {backup_path}")

    # 读取并修改配置
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    # 应用概念扩增配置
    if 'concept_expansion' not in full_config:
        full_config['concept_expansion'] = {}

    for key, value in config.items():
        old_value = full_config['concept_expansion'].get(key, "未设置")
        full_config['concept_expansion'][key] = value
        print(f"  {key}: {old_value} → {value}")

    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✅ {mode_name}配置已应用到: {config_path}")

    # 显示配置摘要
    print("\n配置摘要:")
    print(f"- 最大迭代次数: {config['max_iterations']}")
    print(f"- 批处理大小: {config['batch_size']}")
    print(f"- 并发工作数: {config['max_workers']}")
    print(f"- 自动保存: {config['auto_save_after_iteration']}")
    print(f"- 保存到Neo4j: {config['save_to_neo4j_after_each_iteration']}")
    if config.get('stop_after_first_iteration'):
        print("- 第一轮后停止: 是")

def show_current_config():
    """显示当前配置"""
    config_path = "config/config.yaml"

    if not os.path.exists(config_path):
        print("❌ 配置文件不存在")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    concept_config = config.get('concept_expansion', {})

    print("\n=== 当前概念扩增配置 ===")
    print(f"最大迭代次数: {concept_config.get('max_iterations', '未设置')}")
    print(f"批处理大小: {concept_config.get('batch_size', '未设置')}")
    print(f"并发工作数: {concept_config.get('max_workers', '未设置')}")
    print(f"自动保存: {concept_config.get('auto_save_after_iteration', '未设置')}")
    print(f"保存到Neo4j: {concept_config.get('save_to_neo4j_after_each_iteration', '未设置')}")
    print(f"第一轮后停止: {concept_config.get('stop_after_first_iteration', '未设置')}")

    # 其他相关配置
    print(f"\n=== 相关配置 ===")
    api_config = config.get('api', {})
    print(f"最大tokens: {api_config.get('max_tokens', '未设置')}")
    print(f"最大重试: {api_config.get('max_retries', '未设置')}")
    print(f"超时时间: {api_config.get('timeout', '未设置')}")

    embedding_config = config.get('embedding', {})
    print(f"Embedding批处理大小: {embedding_config.get('batch_size', '未设置')}")
    print(f"请求延迟: {embedding_config.get('request_delay', '未设置')}")

if __name__ == "__main__":
    import time
    configure_expansion()