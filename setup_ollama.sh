#!/bin/bash
# 设置 Ollama 模型

echo "🚀 检查并下载必需的 Ollama 模型..."
echo "="*50

# 检查 Ollama 是否安装
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama 未安装"
    echo "💡 请运行: brew install ollama"
    exit 1
fi

echo "✅ Ollama 已安装"

# 检查 Ollama 服务是否运行
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "⚠️  Ollama 服务未运行"
    echo "💡 请在新终端运行: ollama serve"
    echo ""
fi

# 下载 LLM 模型
echo "📥 检查 LLM 模型 (llama3.2:3b)..."
if ollama list | grep -q "llama3.2:3b"; then
    echo "✅ llama3.2:3b 已安装"
else
    echo "📥 下载 llama3.2:3b (这可能需要几分钟)..."
    ollama pull llama3.2:3b
fi

# 下载嵌入模型
echo ""
echo "📥 检查嵌入模型 (nomic-embed-text)..."
if ollama list | grep -q "nomic-embed-text"; then
    echo "✅ nomic-embed-text 已安装"
else
    echo "📥 下载 nomic-embed-text..."
    ollama pull nomic-embed-text
fi

echo ""
echo "🎉 所有模型已准备就绪！"
echo ""
echo "📋 已安装的模型:"
ollama list
