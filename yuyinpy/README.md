# Edge Voice Assistant (Edge智音)

Edge Voice Assistant是一个为RK3588等嵌入式平台设计的本地语音助手系统，特别针对ELF 2开发板优化。该系统利用本地语音识别和DeepSeek LLM模型，即使在弱网或离线环境也能提供语音交互能力。

## 主要特性

- **本地语音识别**：使用优化的ASR模型实现离线语音识别
- **本地大语言模型**：集成DeepSeek-R1-Distill-Qwen-1.5B模型提供智能回复
- **离线运行**：完全本地化部署，不依赖云服务
- **简洁GUI界面**：提供简单直观的用户界面
- **支持中文界面**：界面支持中文显示（语音交互仅支持英文）

## 快速开始

1. 确认DeepSeek模型和可执行文件已安装在正确位置
2. 运行启动脚本:

```bash
./start.sh
```

**注意**：请使用英文进行语音交互，目前DeepSeek模型仅支持英文输入。

## 项目结构

```
edge_assistant/
├── edge_assistant.py        # 核心功能模块
├── edge_assistant_gui.py    # 图形界面
├── asr_io_test.py           # ASR和TTS功能
├── start.sh                 # 启动脚本
├── config/                  # 配置文件目录
├── agents/                  # 代理配置目录
├── tools/                   # 工具目录
└── knowledge/               # 知识库目录
```

## DeepSeek回复处理

由于DeepSeek模型输出格式特殊，系统采用以下策略处理回复：
- 保存所有原始输出到`deepseek_raw_output.txt`文件
- 简化处理逻辑，提取"robot:"后的内容
- 处理`<think>`标记，获取有效回复
- 使用预设回复机制（可通过环境变量`USE_PRESET_RESPONSES`启用或禁用）

## 重要提示

- 目前系统仅支持英文语音输入，中文输入会返回错误提示
- DeepSeek模型需要足够的系统资源，在资源受限设备上可能表现不佳
- 如遇问题，请查看`edge_assistant.log`日志文件获取详细信息

## 最近更新

### 性能优化 (2025-06-18)

1. **查询缓存机制**：添加了查询缓存系统，重复问题将直接从缓存获取回答
2. **模型预热**：系统启动时自动预热模型，减少首次查询的等待时间
3. **异步处理**：使用线程池实现异步查询处理，提高系统响应性
4. **优化进程管理**：改进了DeepSeek模型进程的启动和终止逻辑，更加稳定
5. **智能超时处理**：即使超时也不会强制中断模型，而是提供临时回复

### 响应优化 (2025-06-16)

1. **响应长度限制**：所有AI回复现在限制在100个字符以内，确保简洁明了的回答
2. **早期响应检测**：新增早期有效回复检测机制，当检测到完整句子时可提前结束等待
3. **超时处理优化**：调整了等待时间参数，加快响应速度
   - 完整回复等待时间：60秒 → 10秒
   - 思考过程等待时间：90秒 → 20秒
   - 无输出最长等待时间：120秒 → 30秒
4. **智能截断**：在句子自然结束处截断过长回复，保持语义完整性

以上优化显著提高了系统响应速度和稳定性，同时保证了回复质量 

# Edge智音 - RAG知识库增强

Edge智音现在支持基于检索增强生成（RAG）的知识库功能。这使得系统能够在回答用户问题时，从本地知识库中检索相关信息，大大提高了回答的准确性和专业性。

## 知识库功能特点

- **本地知识库**：所有知识存储在本地，无需联网，保护隐私
- **多智能体知识支持**：不同智能体可以关联不同的知识库，提供专业领域支持
- **文本分割与向量化**：自动将文档分割成适合检索的块，并进行向量化存储
- **高效检索**：支持基于相似度的快速检索
- **缓存机制**：向量存储支持缓存，提高频繁查询的速度

## 目录结构

```
knowledge/         # 知识库根目录
  ├── technology/  # 技术相关知识库
  │   └── programming_tips.txt
  ├── weather/     # 天气相关知识库
  │   └── weather_info.txt
  └── general/     # 通用知识库
      └── ...
```

## 知识库管理工具

项目提供了`kb_tools.py`工具脚本，可以方便地管理知识库：

```bash
# 列出所有知识库
python kb_tools.py list

# 创建新知识库
python kb_tools.py create <知识库名称>

# 将文件添加到知识库
python kb_tools.py add <知识库名称> <文件路径>

# 列出知识库中的文件
python kb_tools.py files <知识库名称>

# 测试知识库检索
python kb_tools.py test <知识库名称> <查询词>

# 添加问答对到知识库
python kb_tools.py qa <知识库名称> <问题> <回答>
```

## 使用方法

### 1. 创建和管理知识库

首先，创建所需的知识库：

```bash
python kb_tools.py create technology
python kb_tools.py create weather
```

然后，向知识库添加文件：

```bash
python kb_tools.py add technology /path/to/tech_docs.txt
python kb_tools.py add weather /path/to/weather_data.txt
```

### 2. 智能体配置

在智能体配置文件中，使用`knowledge_base`字段指定该智能体可以访问的知识库：

```json
{
    "name": "Tech Assistant",
    "prompt": "You are a Technology Assistant that specializes in answering questions about computers, programming, and technology.",
    "knowledge_base": ["technology", "general"],
    "tools": []
}
```

### 3. 运行系统

启动Edge智音系统，智能体会根据用户的查询自动从相应的知识库中检索信息。

## 支持的文件类型

目前支持以下文件类型：
- 纯文本文件 (.txt)
- JSON格式的问答对文件 (.json)

## 技术细节

知识库功能使用了以下技术：
- 文本分割：将长文档分割为较小的块
- TF-IDF向量化：使用简单而有效的词袋模型进行向量化
- FAISS索引（可选）：如果安装了FAISS库，将使用高效的向量索引
- 余弦相似度：用于计算查询与文档的相似性

## 依赖项

知识库功能依赖以下Python包：
- numpy：用于向量计算
- faiss-cpu（可选）：用于高效向量检索，如果不安装则回退到暴力计算

可以使用以下命令安装依赖：

```bash
pip install numpy
pip install faiss-cpu  # 可选，但推荐
``` 