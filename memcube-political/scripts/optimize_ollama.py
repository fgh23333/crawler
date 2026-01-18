#!/usr/bin/env python3
"""
优化Ollama配置，避免模型过载
"""

import subprocess
import sys
import time
import requests
import psutil
import os

def check_ollama_status():
    """检查Ollama状态"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama服务运行正常")
            return True
        else:
            print(f"✗ Ollama服务异常，状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到Ollama服务，请确保Ollama正在运行")
        return False
    except Exception as e:
        print(f"✗ 检查Ollama状态时出错: {e}")
        return False

def check_system_resources():
    """检查系统资源"""
    print("\n=== 系统资源检查 ===")

    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU使用率: {cpu_percent}%")

    # 内存使用率
    memory = psutil.virtual_memory()
    print(f"内存使用率: {memory.percent}%")
    print(f"可用内存: {memory.available / (1024**3):.1f} GB")

    # GPU内存（如果可用）
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"GPU使用率: {gpu.load*100:.1f}%")
            print(f"GPU内存使用率: {gpu.memoryUtil*100:.1f}%")
            print(f"可用GPU内存: {gpu.memoryFree:.1f} MB")
    except ImportError:
        print("GPU: 未安装GPUtil，无法检查GPU状态")
    except Exception as e:
        print(f"GPU: 检查失败: {e}")

def optimize_ollama_settings():
    """优化Ollama设置"""
    print("\n=== Ollama优化建议 ===")

    # 检查内存情况
    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024**3)

    print(f"当前可用内存: {available_memory_gb:.1f} GB")

    if available_memory_gb < 4:
        print("⚠️ 内存不足，建议:")
        print("  1. 关闭其他应用程序")
        print("  2. 使用更小的模型（如 nomic-embed-text）")
        print("  3. 减少批处理大小")
        print("  4. 考虑使用在线API服务")
    elif available_memory_gb < 8:
        print("⚠️ 内存有限，建议:")
        print("  1. 批处理大小设置为2-4")
        print("  2. 并发数设置为1")
        print("  3. 添加请求延迟")
    else:
        print("✓ 内存充足，可以使用默认设置")

    # CPU核心数建议
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)

    print(f"\nCPU核心数: {cpu_count}")
    print(f"当前CPU使用率: {cpu_percent}%")

    if cpu_count < 4:
        print("⚠️ CPU核心较少，建议:")
        print("  1. 并发数设置为1")
        print("  2. 增加处理间隔")
    elif cpu_percent > 80:
        print("⚠️ CPU使用率高，建议:")
        print("  1. 减少并发数")
        print("  2. 降低批处理大小")

def restart_ollama():
    """重启Ollama服务"""
    print("\n=== 重启Ollama服务 ===")

    try:
        # 尝试停止Ollama
        print("正在停止Ollama...")
        subprocess.run(["pkill", "-f", "ollama"], check=False)
        time.sleep(3)

        # 启动Ollama
        print("正在启动Ollama...")
        subprocess.Popen(["ollama", "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

        # 等待服务启动
        for i in range(10):
            time.sleep(2)
            if check_ollama_status():
                print("✓ Ollama重启成功")
                return True

        print("✗ Ollama启动超时")
        return False

    except Exception as e:
        print(f"✗ 重启Ollama失败: {e}")
        return False

def set_environment_variables():
    """设置环境变量优化"""
    print("\n=== 设置环境变量 ===")

    env_vars = {
        "OLLAMA_MAX_LOADED_MODELS": "1",  # 限制同时加载的模型数量
        "OLLAMA_NUM_PARALLEL": "1",       # 限制并行请求数
        "OLLAMA_MAX_QUEUE": "10",         # 限制队列大小
        "OLLAMA_REQUEST_TIMEOUT": "120",  # 请求超时时间
    }

    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置 {key}={value}")

def create_optimized_config():
    """创建优化的配置文件"""
    print("\n=== 生成优化配置 ===")

    config_content = """
# Ollama优化配置
# 保存为 config/ollama_optimized.yaml

api:
  model_expander: "gemini-2.5-flash"
  max_tokens: 2048
  max_retries: 5
  timeout: 180
  rate_limit_delay: 5.0
  enable_rate_limiting: true

concept_expansion:
  batch_size: 5
  max_workers: 1
  similarity_threshold: 0.85

embedding:
  batch_size: 2
  request_delay: 3.0

qa_generation:
  concepts_per_batch: 3
  max_workers: 1
  qa_pairs_per_concept: 1
"""

    config_path = "config/ollama_optimized.yaml"
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"✓ 优化配置已保存到: {config_path}")
        print("使用方法: python main.py --config config/ollama_optimized.yaml")
    except Exception as e:
        print(f"✗ 保存配置失败: {e}")

def main():
    """主函数"""
    print("=== Ollama过载优化工具 ===")

    # 检查系统资源
    check_system_resources()

    # 检查Ollama状态
    if not check_ollama_status():
        print("\n是否尝试重启Ollama服务? (y/n): ", end="")
        if input().lower() in ['y', 'yes']:
            restart_ollama()

    # 设置环境变量
    set_environment_variables()

    # 提供优化建议
    optimize_ollama_settings()

    # 创建优化配置
    create_optimized_config()

    print("\n=== 优化完成 ===")
    print("建议:")
    print("1. 降低批处理大小和并发数")
    print("2. 增加请求间隔时间")
    print("3. 监控系统资源使用情况")
    print("4. 考虑使用更轻量的模型")
    print("5. 如果持续过载，考虑使用在线API服务")

if __name__ == "__main__":
    main()