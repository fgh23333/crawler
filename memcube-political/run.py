#!/usr/bin/env python3
"""
MemCube 政治理论概念图扩增系统 - 简化启动脚本
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """设置运行环境"""
    # 确保可以导入src模块
    project_root = Path(__file__).parent
    src_path = project_root / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

def show_help():
    """显示帮助信息"""
    print("MemCube 政治理论概念图扩增系统")
    print("=" * 50)
    print()
    print("使用方法:")
    print("  python run.py check          - 检查环境")
    print("  python run.py test-api       - 测试API配置")
    print("  python run.py test-system    - 测试系统功能")
    print("  python run.py start-db       - 数据库版快速启动")
    print("  python run.py start          - 内存版快速启动")
    print("  python run.py all            - 运行完整流程")
    print()
    print("示例:")
    print("  python run.py check")
    print("  python run.py start-db")

def check_environment():
    """检查运行环境"""
    print("检查运行环境...")
    try:
        # 直接运行check_env脚本
        import subprocess
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "scripts" / "check_env.py")
        ], cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"环境检查失败: {e}")
        return False

def test_api():
    """测试API配置"""
    print("测试API配置...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "scripts" / "test_api_simple.py")
        ], cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"API测试失败: {e}")
        return False

def test_system():
    """测试系统功能"""
    print("测试系统功能...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "scripts" / "test_system.py")
        ], cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"系统测试失败: {e}")
        return False

def quick_start_db():
    """数据库版快速启动"""
    print("启动数据库版快速启动...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "scripts" / "quick_start_database.py")
        ], cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"数据库版启动失败: {e}")
        return False

def quick_start():
    """内存版快速启动"""
    print("启动内存版快速启动...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "scripts" / "quick_start.py")
        ], cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"内存版启动失败: {e}")
        return False

def run_all():
    """运行完整流程"""
    print("运行完整流程...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "src.main", "--stage", "all"
        ], cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"运行完整流程失败: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) < 2:
        show_help()
        return 0

    command = sys.argv[1]

    # 设置环境
    setup_environment()

    if command == "check":
        return 0 if check_environment() else 1
    elif command == "test-api":
        return 0 if test_api() else 1
    elif command == "test-system":
        return 0 if test_system() else 1
    elif command == "start-db":
        return 0 if quick_start_db() else 1
    elif command == "start":
        return 0 if quick_start() else 1
    elif command == "all":
        return 0 if run_all() else 1
    else:
        print(f"未知命令: {command}")
        show_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())