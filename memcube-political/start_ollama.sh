#!/bin/bash

# Ollama + BGE-M3 启动脚本
# 适用于Linux/macOS

echo "🚀 启动Ollama服务..."

# 检查Ollama是否已安装
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama未安装，请先安装Ollama"
    echo "安装指南：https://github.com/ollama/ollama"
    exit 1
fi

# 检查BGE-M3模型是否已下载
if ! ollama list | grep -q "bge-m3"; then
    echo "📥 下载BGE-M3模型..."
    ollama pull bge-m3
    if [ $? -ne 0 ]; then
        echo "❌ 模型下载失败"
        exit 1
    fi
    echo "✅ BGE-M3模型下载完成"
else
    echo "✅ BGE-M3模型已存在"
fi

# 启动Ollama服务
echo "🔄 启动Ollama服务..."
ollama serve &
OLLAMA_PID=$!

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 5

# 测试连接
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama服务启动成功"
    echo "🌐 服务地址: http://localhost:11434"
    echo "📊 可用模型:"
    ollama list
    echo ""
    echo "💡 现在可以运行MemCube项目:"
    echo "   cd memcube-political"
    echo "   python quick_start.py"
    echo ""
    echo "停止服务: kill $OLLAMA_PID"
else
    echo "❌ Ollama服务启动失败"
    kill $OLLAMA_PID 2>/dev/null
    exit 1
fi