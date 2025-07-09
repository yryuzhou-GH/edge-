#!/bin/bash
# Edge Voice Assistant 简洁启动脚本

# 设置日志文件和基本参数
LOG_FILE="edge_assistant.log"
PYTHON_CMD="python3"
GUI_APP="edge_assistant_gui.py"
USE_PRESET=1

echo "========================================" > "$LOG_FILE"
echo "Edge Voice Assistant - 启动日志" >> "$LOG_FILE" 
echo "时间: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# 检查DeepSeek可执行文件
DEEPSEEK_EXECUTABLE="/home/elf/work/deepseek/rknn-llm-main/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/deploy/build/build_linux_aarch64_Release/llm_demo"
if [ -f "$DEEPSEEK_EXECUTABLE" ]; then
    echo "正在检查DeepSeek可执行文件..." | tee -a "$LOG_FILE"
    if [ ! -x "$DEEPSEEK_EXECUTABLE" ]; then
        echo "正在添加执行权限..." | tee -a "$LOG_FILE"
        chmod +x "$DEEPSEEK_EXECUTABLE" 2>> "$LOG_FILE"
    fi
else
    echo "警告: DeepSeek可执行文件不存在，将使用预设回复" | tee -a "$LOG_FILE"
fi

# 设置环境变量
export USE_PRESET_RESPONSES=$USE_PRESET
export PYTHONPATH=$(pwd):$PYTHONPATH

# 创建必要的目录
mkdir -p config agents tools knowledge

echo "启动Edge Voice Assistant..." | tee -a "$LOG_FILE"
$PYTHON_CMD $GUI_APP 2>&1 | tee -a "$LOG_FILE"

# 检查退出状态
if [ $? -ne 0 ]; then
    echo "应用退出状态: $?" | tee -a "$LOG_FILE"
fi 