#!/usr/bin/env python3
"""
MemCube  - 
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """"""
    print("检查依赖...")

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
            print(f" {package}")
        except ImportError:
            missing_packages.append(package)
            print(f" {package}")

    if missing_packages:
        print(f"\n ...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + missing_packages)
        print(" ")
    else:
        print(" ")

def setup_api_config():
    """API"""
    api_config_file = Path("config/api_keys.yaml")

    if not api_config_file.exists():
        print(" API...")
        api_config_file.parent.mkdir(exist_ok=True)

        # 
        example_file = Path("config/api_keys.yaml.example")
        if example_file.exists():
            import shutil
            shutil.copy(example_file, api_config_file)
        else:
            # 
            api_config_content = """# API
# API

openai:
  api_key: "your-openai-api-key-here"
  organization: "your-organization-id-here"  # 
"""
            with open(api_config_file, 'w', encoding='utf-8') as f:
                f.write(api_config_content)

        print(f" API: {api_config_file}")
        print("  OpenAI API")
        return False

    # 
    with open(api_config_file, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'your-openai-api-key-here' in content:
            print("  API")
            print(f": {api_config_file}")
            return False

    print(" API")
    return True

def check_data_files():
    """"""
    data_files = [
        "data/seed_concepts.txt"
    ]

    print(" ...")

    for file_path in data_files:
        path = Path(file_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                print(f" {file_path} ({len(lines)} )")
        else:
            print(f" {file_path} ()")
            return False

    return True

def run_demo():
    """"""
    print(" MemCube...")

    # 
    print("\n:")
    print("1.  ()")
    print("2. ")
    print("3. QA")

    try:
        choice = input("\n (1-3): ").strip()

        if choice == "1":
            print(" ...")
            subprocess.run([
                sys.executable, "-m", "src.main",
                "--stage", "all"
            ])
        elif choice == "2":
            print(" ...")
            subprocess.run([
                sys.executable, "-m", "src.main",
                "--stage", "concept-expansion"
            ])
        elif choice == "3":
            print(" QA...")
            subprocess.run([
                sys.executable, "-m", "src.main",
                "--stage", "qa-generation"
            ])
        else:
            print(" ")
            return False

    except KeyboardInterrupt:
        print("\n  ")
        return False
    except Exception as e:
        print(f" : {e}")
        return False

    return True

def show_results():
    """"""
    print("\n :")

    results_dir = Path("results")
    if results_dir.exists():
        print(f" : {results_dir.absolute()}")

        # 
        for file_path in results_dir.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
                print(f"   {file_path.relative_to(results_dir)} ({size_str})")
    else:
        print(" ")

def main():
    """"""
    print("=" * 60)
    print("MemCube ")
    print("   OpenAI API")
    print("=" * 60)

    # 
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # 
    steps = [
        ("", check_dependencies),
        ("API", setup_api_config),
        ("", check_data_files)
    ]

    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f" {step_name}")
            return

    print("\n ")

    # 
    try:
        run_demo_choice = input("\n? (y/n): ").strip().lower()
        if run_demo_choice in ['y', 'yes', '']:
            if run_demo():
                show_results()
        else:
            print("\n :")
            print("   : pip install -r requirements.txt")
            print("   API:  config/api_keys.yaml")
            print("   : python -m src.main --stage all")
            print("   : python -m src.main --help")
    except KeyboardInterrupt:
        print("\n ")

if __name__ == "__main__":
    main()