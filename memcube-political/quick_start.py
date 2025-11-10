#!/usr/bin/env python3
"""
MemCube 政治理论概念图扩增系统 - 快速启动脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")

    required_packages = [
        'openai',
        'sentence-transformers',
        'numpy',
        'pandas',
        'pyyaml',
        'tqdm',
        'requests',
        'jsonlines',
        'python-dotenv',
        'loguru',
        'networkx'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")

    if missing_packages:
        print(f"\n📦 安装缺失的依赖包...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + missing_packages)
        print("✅ 依赖安装完成")
    else:
        print("✅ 所有依赖已满足")

def setup_api_config():
    """设置API配置"""
    api_config_file = Path("config/api_keys.yaml")

    if not api_config_file.exists():
        print("📝 创建API配置文件...")
        api_config_file.parent.mkdir(exist_ok=True)

        # 复制示例文件
        example_file = Path("config/api_keys.yaml.example")
        if example_file.exists():
            import shutil
            shutil.copy(example_file, api_config_file)
        else:
            # 创建基本配置文件
            api_config_content = """# API密钥配置文件
# 请填入你的真实API密钥

openai:
  api_key: "your-openai-api-key-here"
  organization: "your-organization-id-here"  # 可选
"""
            with open(api_config_file, 'w', encoding='utf-8') as f:
                f.write(api_config_content)

        print(f"📝 API配置文件已创建: {api_config_file}")
        print("⚠️  请编辑此文件，填入你的OpenAI API密钥")
        return False

    # 检查是否为示例配置
    with open(api_config_file, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'your-openai-api-key-here' in content:
            print("⚠️  请先配置API密钥！")
            print(f"编辑文件: {api_config_file}")
            return False

    print("✅ API配置已就绪")
    return True

def check_data_files():
    """检查数据文件"""
    data_files = [
        "data/seed_concepts.txt"
    ]

    print("📁 检查数据文件...")

    for file_path in data_files:
        path = Path(file_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                print(f"✅ {file_path} ({len(lines)} 个概念)")
        else:
            print(f"❌ {file_path} (文件不存在)")
            return False

    return True

def run_demo():
    """运行演示"""
    print("🚀 开始运行MemCube演示...")

    # 选择运行阶段
    print("\n选择运行阶段:")
    print("1. 完整流程 (推荐)")
    print("2. 仅概念图扩增")
    print("3. 仅QA生成")

    try:
        choice = input("\n请选择 (1-3): ").strip()

        if choice == "1":
            print("🔄 运行完整流程...")
            subprocess.run([
                sys.executable, "-m", "src.main",
                "--stage", "all"
            ])
        elif choice == "2":
            print("🔄 运行概念图扩增...")
            subprocess.run([
                sys.executable, "-m", "src.main",
                "--stage", "concept-expansion"
            ])
        elif choice == "3":
            print("🔄 运行QA生成...")
            subprocess.run([
                sys.executable, "-m", "src.main",
                "--stage", "qa-generation"
            ])
        else:
            print("❌ 无效选择")
            return False

    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        return False
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return False

    return True

def show_results():
    """显示结果"""
    print("\n📊 查看生成的结果:")

    results_dir = Path("results")
    if results_dir.exists():
        print(f"📁 结果目录: {results_dir.absolute()}")

        # 列出生成的文件
        for file_path in results_dir.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
                print(f"  📄 {file_path.relative_to(results_dir)} ({size_str})")
    else:
        print("❌ 结果目录不存在")

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 MemCube 政治理论概念图扩增系统")
    print("   基于OpenAI API的政治理论知识图谱构建工具")
    print("=" * 60)

    # 切换到项目目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # 检查步骤
    steps = [
        ("检查依赖", check_dependencies),
        ("设置API配置", setup_api_config),
        ("检查数据文件", check_data_files)
    ]

    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"❌ {step_name}失败，请解决问题后重试")
            return

    print("\n✅ 环境检查完成！")

    # 询问是否运行演示
    try:
        run_demo_choice = input("\n是否现在运行演示? (y/n): ").strip().lower()
        if run_demo_choice in ['y', 'yes', '是']:
            if run_demo():
                show_results()
        else:
            print("\n💡 使用说明:")
            print("   安装依赖: pip install -r requirements.txt")
            print("   配置API: 编辑 config/api_keys.yaml")
            print("   运行系统: python -m src.main --stage all")
            print("   查看帮助: python -m src.main --help")
    except KeyboardInterrupt:
        print("\n👋 再见！")

if __name__ == "__main__":
    main()