#!/usr/bin/env python3
"""
MemCube 政治理论概念图扩增系统 - 主启动脚本
统一的入口点，支持完整的系统工作流程
"""

import argparse
import sys
import os
from pathlib import Path

def setup_environment():
    """设置运行环境"""
    # 确保可以导入src模块
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    scripts_path = project_root / "scripts"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))

def run_system_stage(stage: str):
    """运行系统特定阶段"""
    try:
        import main as system_main
        sys.argv = ['main.py', '--stage', stage]
        system_main()
    except ImportError as e:
        print(f"导入系统主模块失败: {e}")
        return False
    except Exception as e:
        print(f"运行系统失败: {e}")
        return False
    return True

def run_quick_start():
    """运行快速启动"""
    try:
        import subprocess
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "scripts" / "quick_start.py")
        ], cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"运行快速启动失败: {e}")
        return False

def run_database_quick_start():
    """运行数据库版快速启动"""
    try:
        import subprocess
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "scripts" / "quick_start_database.py")
        ], cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"运行数据库版快速启动失败: {e}")
        return False

def check_environment():
    """检查环境"""
    try:
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
    try:
        import subprocess
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "scripts" / "test_system.py")
        ], cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f("系统测试失败: {e}"))
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MemCube 政治理论概念图扩增系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --stage all                    # 运行完整流程
  %(prog)s --stage concept-expansion     # 仅概念图扩增
  %(prog)s --stage qa-generation         # 仅QA生成
  %(prog)s --check-env                   # 检查环境
  %(prog)s --test-api                    # 测试API
  %(prog)s --test-system                 # 测试系统
  %(prog)s --quick-start                 # 快速启动（内存模式）
  %(prog)s --quick-start-db              # 快速启动（数据库模式）
        """
    )

    # 系统运行选项
    parser.add_argument(
        '--stage',
        choices=['all', 'concept-analysis', 'concept-expansion', 'qa-generation', 'evaluation'],
        help='运行指定阶段的系统功能'
    )

    # 工具选项
    parser.add_argument(
        '--check-env',
        action='store_true',
        help='检查运行环境'
    )

    parser.add_argument(
        '--test-api',
        action='store_true',
        help='测试API配置'
    )

    parser.add_argument(
        '--test-system',
        action='store_true',
        help='测试系统功能'
    )

    # 快速启动选项
    parser.add_argument(
        '--quick-start',
        action='store_true',
        help='快速启动（内存模式）'
    )

    parser.add_argument(
        '--quick-start-db',
        action='store_true',
        help='快速启动（数据库模式）'
    )

    # 配置选项
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='配置文件路径（默认: config/config.yaml）'
    )

    args = parser.parse_args()

    # 设置环境
    setup_environment()

    print("=" * 60)
    print("MemCube 政治理论概念图扩增系统")
    print("=" * 60)

    # 执行相应功能
    if args.check_env:
        print("🔍 检查运行环境...")
        success = check_environment()
        if success:
            print("✅ 环境检查通过")
        else:
            print("❌ 环境检查失败")
            return 1

    elif args.test_api:
        print("🔧 测试API配置...")
        success = test_api()
        if success:
            print("✅ API测试通过")
        else:
            print("❌ API测试失败")
            return 1

    elif args.test_system:
        print("🧪 测试系统功能...")
        success = test_system()
        if success:
            print("✅ 系统测试通过")
        else:
            print("❌ 系统测试失败")
            return 1

    elif args.quick_start:
        print("🚀 快速启动（内存模式）...")
        success = run_quick_start()
        if success:
            print("✅ 快速启动完成")
        else:
            print("❌ 快速启动失败")
            return 1

    elif args.quick_start_db:
        print("🚀 快速启动（数据库模式）...")
        success = run_database_quick_start()
        if success:
            print("✅ 快速启动完成")
        else:
            print("❌ 快速启动失败")
            return 1

    elif args.stage:
        print(f"🔄 运行系统阶段: {args.stage}")
        success = run_system_stage(args.stage)
        if success:
            print(f"✅ 阶段 {args.stage} 完成")
        else:
            print(f"❌ 阶段 {args.stage} 失败")
            return 1

    else:
        # 没有指定参数，显示帮助信息
        parser.print_help()
        print("\n💡 建议先运行环境检查:")
        print("   python main.py --check-env")
        print("\n🚀 然后运行快速启动:")
        print("   python main.py --quick-start-db")

    return 0

if __name__ == "__main__":
    sys.exit(main())