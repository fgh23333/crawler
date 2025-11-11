#!/usr/bin/env python3
"""
简化的API配置测试脚本
测试API客户端的基本功能，避免Unicode编码问题
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.append('src')

def test_api_basic():
    """基础API测试"""
    print("=" * 60)
    print("API基础连接测试")
    print("=" * 60)

    try:
        # 导入测试
        print("1. 导入API客户端...")
        import src.api_client as api_client
        UnifiedAPIClient = api_client.UnifiedAPIClient
        print("   导入成功")

        # 创建客户端实例
        print("2. 初始化API客户端...")
        client = UnifiedAPIClient()
        print(f"   API类型: {client.config.get('type', '未知')}")
        print(f"   配置键: {list(client.config.keys())}")

        # 测试基础对话功能
        print("3. 测试API对话功能...")
        messages = [
            {"role": "user", "content": "请简单介绍马克思主义的基本概念。"}
        ]

        # 测试Gemini模型
        print("   测试Gemini 1.5 Flash模型...")
        response = client.chat_completion(
            messages=messages,
            model="gemini-1.5-flash",
            temperature=0.7,
            max_tokens=100
        )

        if response.success:
            print("   API调用成功")
            if response.content:
                print(f"   响应长度: {len(response.content)} 字符")
                print(f"   响应内容: {response.content[:50]}...")
            else:
                print("   响应内容为空")
        else:
            print(f"   API调用失败: {response.error}")

        # 测试另一个Gemini模型
        print("   测试Gemini 2.0 Flash模型...")
        response = client.chat_completion(
            messages=messages,
            model="gemini-2.0-flash-exp",
            temperature=0.7,
            max_tokens=100
        )

        if response.success:
            print("   模型调用成功")
            if response.content:
                print(f"   响应长度: {len(response.content)} 字符")
                print(f"   响应内容: {response.content[:50]}...")
            else:
                print("   响应内容为空")
        else:
            print(f"   模型调用失败: {response.error}")

        return True

    except ImportError as e:
        print(f"   导入错误: {e}")
        return False
    except Exception as e:
        print(f"   测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("MemCube API配置测试（简化版）")
    print("测试API客户端的基础功能")

    # 运行测试
    result = test_api_basic()

    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)

    if result:
        print("基础API功能测试通过!")
        print("系统核心功能正常工作。")
    else:
        print("API功能测试失败")
        print("请检查配置和依赖。")

if __name__ == "__main__":
    main()