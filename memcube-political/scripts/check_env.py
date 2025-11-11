#!/usr/bin/env python3
"""
环境验证脚本
检查Ollama、Python包和配置是否正确设置
"""

import sys
import os
import requests
import yaml
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Python版本过低，需要3.8+")
        return False
    else:
        print("Python版本满足要求")
        return True

def check_packages():
    """检查必需的Python包"""
    print("\n检查Python包...")

    required_packages = [
        'openai',
        'numpy',
        'pandas',
        'pyyaml',
        'tqdm',
        'requests',
        'jsonlines',
        'loguru',
        'networkx'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'sentence-transformers':
                import sentence_transformers
            elif package == 'python-dotenv':
                import dotenv
            elif package == 'pyyaml':
                import yaml
            else:
                __import__(package.replace('-', '_'))
            print(f"通过: {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"缺失: {package}")

    if missing_packages:
        print(f"\n缺少包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("所有必需包已安装")
        return True

def check_ollama():
    """检查Ollama服务"""
    print("\n检查Ollama服务...")

    try:
        # 测试Ollama连接
        response = requests.get("http://localhost:11434/api/tags", timeout=5)

        if response.status_code == 200:
            data = response.json()
            models = [model.get('name', '') for model in data.get('models', [])]

            print("Ollama服务运行正常")
            print(f"可用模型: {models}")

            # 检查BGE-M3模型
            bge_m3_available = any('bge-m3' in model for model in models)
            if bge_m3_available:
                print("BGE-M3模型已安装")
                return True
            else:
                print("BGE-M3模型未安装")
                print("请运行: ollama pull bge-m3")
                return False
        else:
            print(f"Ollama服务响应异常: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("无法连接到Ollama服务")
        print("请确保Ollama正在运行: ollama serve")
        return False
    except Exception as e:
        print(f"检查Ollama时出错: {e}")
        return False

def test_embedding():
    """测试embedding功能"""
    print("\n测试Embedding功能...")

    try:
        # 使用项目中的embedding客户端
        sys.path.append('src')
        from embedding_client import get_embedding_client

        client = get_embedding_client()

        # 测试单个文本
        test_text = "这是一个测试文本"
        embedding = client.encode([test_text])

        if len(embedding) > 0:
            dimension = len(embedding[0])
            print(f"Embedding功能正常，维度: {dimension}")
            return True
        else:
            print("Embedding返回空结果")
            return False

    except Exception as e:
        print(f"Embedding测试失败: {e}")
        return False

def check_config():
    """检查配置文件"""
    print("\n检查配置文件...")

    config_files = [
        'config/config.yaml'
    ]

    all_exist = True
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"通过: {config_file}")
        else:
            print(f"缺失: {config_file}")
            all_exist = False

    # 检查API密钥配置
    api_config_file = Path('config/api_keys.yaml')
    if api_config_file.exists():
        try:
            with open(api_config_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查Gemini API密钥
            if 'your-gemini-api-key-here' in content:
                print("警告: Gemini API密钥尚未配置")
                print("请编辑 config/api_keys.yaml 文件")
                all_exist = False
            else:
                print("通过: Gemini API密钥已配置")

          # 检查API密钥是否已配置
            if 'your-gemini-api-key-here' in content:
                print("警告: API密钥尚未配置")
                all_exist = False
            else:
                print("通过: API密钥已配置")
        except Exception as e:
            print(f"错误: 读取API配置失败: {e}")
            all_exist = False
    else:
        print("警告: API密钥配置文件不存在")
        print("请复制 config/api_keys.yaml.example 为 config/api_keys.yaml")
        all_exist = False

    return all_exist

def check_data_files():
    """检查数据文件"""
    print("\n检查数据文件...")

    data_files = [
        'data/seed_concepts.txt'
    ]

    all_exist = True
    for data_file in data_files:
        if Path(data_file).exists():
            if data_file.endswith('.txt'):
                with open(data_file, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    print(f"通过: {data_file} ({len(lines)} 个概念)")
            else:
                print(f"通过: {data_file}")
        else:
            print(f"缺失: {data_file}")
            all_exist = False

    return all_exist

def main():
    """主检查函数"""
    print("=" * 60)
    print("MemCube 环境检查")
    print("=" * 60)

    checks = [
        ("Python版本", check_python_version),
        ("Python包", check_packages),
        ("Ollama服务", check_ollama),
        ("Embedding功能", test_embedding),
        ("配置文件", check_config),
        ("数据文件", check_data_files)
    ]

    results = []

    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"错误: {check_name}检查出错: {e}")
            results.append((check_name, False))

    # 总结
    print("\n" + "=" * 60)
    print("检查结果总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "通过" if result else "失败"
        print(f"{check_name}: {status}")

    print(f"\n总体结果: {passed}/{total} 项通过")

    if passed == total:
        print("环境配置完美！可以开始使用MemCube")
        print("\n下一步:")
        print("   python run.py start-db    # 数据库版快速启动")
        return True
    else:
        print("请解决上述问题后再运行系统")
        print("\n帮助:")
        print("   - 查看 OLLAMA_SETUP.md 了解Ollama设置")
        print("   - 查看 USAGE.md 了解详细使用方法")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)