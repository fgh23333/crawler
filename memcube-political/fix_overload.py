#!/usr/bin/env python3
"""
快速修复模型过载问题
"""

import os
import yaml

def main():
    """主函数"""
    print("=== 模型过载快速修复 ===")

    config_path = "config/config.yaml"

    # 备份原配置文件
    if os.path.exists(config_path):
        backup_path = config_path + ".backup"
        with open(config_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"✓ 原配置文件已备份到: {backup_path}")

    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 应用过载修复设置
    fixes = {
        "api": {
            "max_tokens": 2048,
            "max_retries": 5,
            "timeout": 180,
            "rate_limit_delay": 5.0,
            "enable_rate_limiting": True
        },
        "concept_expansion": {
            "batch_size": 5,
            "max_workers": 1
        },
        "qa_generation": {
            "concepts_per_batch": 3,
            "max_workers": 1,
            "qa_pairs_per_concept": 1,
            "qa_pairs_per_concept_pair": 1
        },
        "embedding": {
            "batch_size": 2,
            "request_delay": 3.0
        }
    }

    # 应用修复
    for section, settings in fixes.items():
        if section not in config:
            config[section] = {}
        for key, value in settings.items():
            old_value = config[section].get(key, "未设置")
            config[section][key] = value
            print(f"  {section}.{key}: {old_value} → {value}")

    # 保存修复后的配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✓ 配置文件已修复: {config_path}")
    print("\n修复内容:")
    print("- 大幅降低批处理大小和并发数")
    print("- 增加请求间隔时间")
    print("- 增加重试次数和超时时间")
    print("- 启用速率限制")

    print("\n使用方法:")
    print("1. 重启你的程序")
    print("2. 监控系统资源使用情况")
    print("3. 如果仍然过载，可以进一步调整参数")

    print("\n建议:")
    print("- 监控CPU和内存使用率")
    print("- 考虑关闭其他应用程序")
    print("- 如果问题持续，考虑使用更轻量的模型")

if __name__ == "__main__":
    main()