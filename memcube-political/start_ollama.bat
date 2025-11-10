@echo off
REM Ollama + BGE-M3 启动脚本
REM 适用于Windows

echo 🚀 启动Ollama服务...

REM 检查Ollama是否已安装
ollama --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama未安装，请先安装Ollama
    echo 下载地址: https://ollama.com/download
    pause
    exit /b 1
)

REM 检查BGE-M3模型是否已下载
ollama list | findstr "bge-m3" >nul
if errorlevel 1 (
    echo 📥 下载BGE-M3模型...
    ollama pull bge-m3
    if errorlevel 1 (
        echo ❌ 模型下载失败
        pause
        exit /b 1
    )
    echo ✅ BGE-M3模型下载完成
) else (
    echo ✅ BGE-M3模型已存在
)

REM 启动Ollama服务
echo 🔄 启动Ollama服务...
start "Ollama Service" ollama serve

REM 等待服务启动
echo ⏳ 等待服务启动...
timeout /t 10 /nobreak >nul

REM 测试连接
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama服务启动失败
    pause
    exit /b 1
)

echo ✅ Ollama服务启动成功
echo 🌐 服务地址: http://localhost:11434
echo 📊 可用模型:
ollama list
echo.
echo 💡 现在可以运行MemCube项目:
echo    cd memcube-political
echo    venv\Scripts\activate
echo    python quick_start.py
echo.
echo 按任意键继续...
pause >nul