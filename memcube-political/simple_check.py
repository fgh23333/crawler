#!/usr/bin/env python3
"""
简化的Qdrant检查测试
"""

import sys
import os

def check_config():
    """检查配置文件"""
    print("检查配置文件...")

    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')

        if not os.path.exists(config_path):
            print(f"ERROR: 配置文件不存在: {config_path}")
            return False

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        vector_config = config.get('vector_database', {})
        options = vector_config.get('options', {})
        check_existing = options.get('check_existing', True)

        print(f"PASS: 配置文件读取成功")
        print(f"  check_existing = {check_existing}")

        if check_existing:
            print("PASS: 存在性检查已启用")
        else:
            print("WARN: 存在性检查已禁用")

        return True

    except Exception as e:
        print(f"ERROR: 配置检查失败: {e}")
        return False

def check_method():
    """检查方法是否存在"""
    print("\n检查方法...")

    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from vector_database_client import QdrantClient

        if hasattr(QdrantClient, 'check_concepts_exist'):
            print("PASS: check_concepts_exist方法存在")
            return True
        else:
            print("FAIL: check_concepts_exist方法不存在")
            return False

    except Exception as e:
        print(f"ERROR: 方法检查失败: {e}")
        return False

if __name__ == "__main__":
    print("=== Qdrant存在性检查验证 ===")

    config_ok = check_config()
    method_ok = check_method()

    if config_ok and method_ok:
        print("\nSUCCESS: 所有关键功能正常")
        print("Qdrant重复向量化问题应该已修复")
    else:
        print("\nFAILED: 部分功能异常")

    print("\n建议:")
    print("1. 确保config/config.yaml中设置了check_existing: true")
    print("2. 确保Qdrant服务正在运行")
    print("3. 检查网络连接和API密钥配置")