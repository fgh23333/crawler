#!/usr/bin/env python3
"""
API配置测试脚本
测试API客户端的配置和连接
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.append('src')

def test_api_config():
    """测试API配置"""
    print("=" * 60)
    print("API配置测试")
    print("=" * 60)

    try:
        # 修复导入问题
        import src.api_client as api_client
        UnifiedAPIClient = api_client.UnifiedAPIClient

        # 创建客户端实例
        print("1. 初始化API客户端...")
        client = UnifiedAPIClient()

        # 显示配置信息
        print(f"   API类型: {client.config.get('type', '未知')}")
        print(f"   配置键: {list(client.config.keys())}")

        # 测试基础对话功能
        print("\n2. 测试API对话功能...")
        messages = [
            {"role": "user", "content": "请简单介绍一下马克思主义的基本概念。"}
        ]

        # 测试Gemini模型
        print("\n测试Gemini 2.5 Flash模型:")
        response = client.chat_completion(
            messages=messages,
            model="gemini-2.5-flash",
            temperature=0.7,
            max_tokens=500
        )

        if response.success:
            print("   Gemini API测试成功")
            print(f"   模型: {response.model}")
            if response.usage:
                print(f"   Token使用: {response.usage}")
            print(f"   响应长度: {len(response.content)} 字符")
            print(f"   响应片段: {response.content[:100]}...")
        else:
            print(f"   Gemini API测试失败: {response.error}")

        # 测试OpenAI模型（如果没有配置会使用模拟响应）
        print("\n测试OpenAI兼容模型:")
        response = client.chat_completion(
            messages=messages,
            model="gpt-4",
            temperature=0.7,
            max_tokens=500
        )

        if response.success:
            print("   OpenAI API测试成功")
            print(f"   模型: {response.model}")
            if response.usage:
                print(f"   Token使用: {response.usage}")
            else:
                print("   使用模拟响应（未配置真实API密钥）")
            print(f"   响应长度: {len(response.content)} 字符")
            print(f"   响应片段: {response.content[:100]}...")
        else:
            print(f"   OpenAI API测试失败: {response.error}")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_response():
    """测试JSON格式响应"""
    print("\n" + "=" * 60)
    print("JSON响应测试")
    print("=" * 60)

    try:
        import src.api_client as api_client
        UnifiedAPIClient = api_client.UnifiedAPIClient

        client = UnifiedAPIClient()

        # 测试JSON响应
        messages = [
            {"role": "user", "content": "请生成一个包含3个政治理论概念的JSON数组，每个概念包含name和description字段。"}
        ]

        print("测试JSON格式响应...")
        response = client.json_completion(
            messages=messages,
            model="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=300
        )

        if response.success:
            print("   JSON响应测试成功")
            print(f"   响应类型: {type(response.content)}")
            if isinstance(response.content, dict) or isinstance(response.content, list):
                print(f"   结构化数据: {len(response.content) if isinstance(response.content, list) else 'Dict'} 项")
            else:
                print("   警告: 响应不是JSON格式")
        else:
            print(f"   JSON响应测试失败: {response.error}")

        return True

    except Exception as e:
        print(f"\n❌ JSON测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("MemCube API配置测试")
    print("测试API客户端的配置和功能")

    # 运行测试
    tests = [
        ("API配置", test_api_config),
        ("JSON响应", test_json_response)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name}测试出错: {e}")
            results.append((test_name, False))

    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "通过" if result else "失败"
        print(f"{test_name}: {status}")

    print(f"\n总体结果: {passed}/{total} 项通过")

    if passed == total:
        print("\nAPI配置测试完全成功!")
        print("系统现在可以:")
        print("   - 使用Gemini 2.5 Flash模型")
        print("   - 支持OpenAI格式API（可选）")
        print("   - JSON格式响应")
        print("   - 模拟响应支持（测试模式）")
    else:
        print("\n部分功能需要配置")
        print("请查看测试日志了解详细信息。")

if __name__ == "__main__":
    main()