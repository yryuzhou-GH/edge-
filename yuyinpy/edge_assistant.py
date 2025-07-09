import os
import sys
import json
import time
import threading
import subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
import requests
from pathlib import Path
import re

# 导入知识库工具
try:
    from knowledge_utils import KnowledgeBase, format_documents_for_prompt
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False
    print("Knowledge base utils are not available. RAG functionality will be disabled.")

# 检查是否启用预设回复模式
USE_PRESET_RESPONSES = os.environ.get('USE_PRESET_RESPONSES', '0') == '1'

# 导入ASR和TTS模块
from asr_io_test import record_audio, play_audio, run_asr, run_tts

# 全局配置
CONFIG_DIR = "./config"
AGENTS_DIR = "./agents"
KNOWLEDGE_DIR = "./knowledge"
TOOLS_DIR = "./tools"
OUTPUT_FILE = "test.wav"
TTS_AUDIO_OUTPUT = "tts_output.wav"
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # 录音时长（秒）
PLAYBACK_DURATION = 10  # 默认播放时长（秒），可通过配置文件覆盖

# 快速响应模式 - 预定义回答，不调用大模型
FAST_RESPONSE_MODE = False  # 设置为False禁用快速响应模式，启用大模型
PREDEFINED_RESPONSES = {
    "hello": "Hello! How can I help you today?",
    "hi": "Hi there! What can I do for you?",
    "how are you": "I'm functioning well, thank you for asking!",
    "what time": "I don't have the current time, but I'm here to assist you.",
    "your name": "I'm your Edge Voice Assistant.",
    "weather": "I don't have access to current weather data.",
    "thank": "You're welcome! Is there anything else you need?",
    "bye": "Goodbye! Have a great day!",
    "help": "I can answer simple questions. Just speak clearly in English.",
}
DEFAULT_RESPONSE = "I'm your Edge Voice Assistant. I'm listening."

# 添加查询缓存机制，避免重复处理相同问题
QUERY_CACHE = {}
MAX_CACHE_SIZE = 20  # 最多缓存20个问题的回答

# 智能体路由缓存，记录最近使用的智能体
CURRENT_AGENT = None
AGENT_CACHE = {}  # 缓存已加载的智能体

# 全局预热标志
MODEL_WARMED_UP = False

# DeepSeek模型配置
DEEPSEEK_MODEL_PATH = "/home/elf/work/deepseek/rknn-llm-main/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/export/DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm"
DEEPSEEK_EXECUTABLE = "/home/elf/work/deepseek/rknn-llm-main/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/deploy/build/build_linux_aarch64_Release/llm_demo"

# 模型参数优化 - 确保完整回复
DEEPSEEK_MAX_NEW_TOKENS = 10000  # 保持10000
DEEPSEEK_TOP_P = 10000

# 模型等待和输出参数
DEEPSEEK_WAIT_TIMEOUT = 120       # 总超时降低到120秒
DEEPSEEK_COMPLETION_WAIT = 5      # 降低到5秒
DEEPSEEK_THINKING_WAIT = 10       # 降低到10秒
DEEPSEEK_WITHOUT_OUTPUT_MAX = 15  # 降低到15秒

# 新增：最小有效回复长度阈值和早期检测设置
DEEPSEEK_MIN_VALID_RESPONSE = 10  # 最少10个字符就认为是有效回复
DEEPSEEK_EARLY_DETECTION = True   # 启用早期检测模式
DEEPSEEK_MAX_RESPONSE_LENGTH = 200  # 限制最大回复长度，加快处理速度

DEEPSEEK_CMD = f"cd {os.path.dirname(DEEPSEEK_EXECUTABLE)} && ./llm_demo {DEEPSEEK_MODEL_PATH} {DEEPSEEK_MAX_NEW_TOKENS} {DEEPSEEK_TOP_P}"

# Debug logging
DEBUG_MODE = True

# 后台线程池，用于并行处理
MODEL_THREAD_POOL = None

# 日志函数定义 - 移到前面，确保在使用前已定义
def log(message):
    """Log a message with timestamp"""
    if DEBUG_MODE:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{current_time}][CORE] {message}")

# 创建全局知识库管理器
KB_MANAGER = None
if KNOWLEDGE_BASE_AVAILABLE:
    try:
        KB_MANAGER = KnowledgeBase()
        log("知识库管理器初始化成功")
    except Exception as e:
        log(f"知识库管理器初始化失败: {e}")
        KNOWLEDGE_BASE_AVAILABLE = False

# 初始化线程池
def init_thread_pool():
    global MODEL_THREAD_POOL
    if MODEL_THREAD_POOL is None:
        log("初始化线程池...")
        try:
            from concurrent.futures import ThreadPoolExecutor
            MODEL_THREAD_POOL = ThreadPoolExecutor(max_workers=1)
            log("线程池初始化成功")
            return True
        except Exception as e:
            log(f"初始化线程池失败: {e}")
            return False
    return True

# 异步版本的query_deepseek函数，用于在后台线程中调用
def async_query_deepseek(prompt, query, agent_config=None):
    """
    在后台线程中调用query_deepseek函数
    返回Future对象，可以用来获取结果
    """
    global MODEL_THREAD_POOL
    
    if not init_thread_pool():
        log("线程池初始化失败，回退到同步调用")
        # 更新处理状态 - 线程池错误
        try:
            with open("processing_status.txt", "w") as f:
                f.write("THREAD_POOL_ERROR\n")
        except:
            pass
        return None
        
    try:
        # 更新处理状态 - 异步处理开始
        try:
            with open("processing_status.txt", "w") as f:
                f.write("ASYNC_PROCESSING\n")
        except:
            log("无法更新处理状态为异步处理")
            
        # 提交任务到线程池
        log(f"异步提交查询: '{query}'")
        future = MODEL_THREAD_POOL.submit(query_deepseek, prompt, query, DEEPSEEK_WAIT_TIMEOUT, agent_config)
        return future
    except Exception as e:
        log(f"异步查询提交失败: {e}")
        # 更新处理状态 - 异步处理错误
        try:
            with open("processing_status.txt", "w") as f:
                f.write(f"ASYNC_ERROR\nERROR: {str(e)}\n")
        except:
            pass
        return None

# 确保所需目录存在
def ensure_dirs():
    for dir_path in [CONFIG_DIR, AGENTS_DIR, KNOWLEDGE_DIR, TOOLS_DIR]:
        Path(dir_path).mkdir(exist_ok=True)
    log(f"Directories created: {CONFIG_DIR}, {AGENTS_DIR}, {KNOWLEDGE_DIR}, {TOOLS_DIR}")

# 检查并修复DeepSeek可执行文件权限
def check_deepseek_executable():
    if not os.path.exists(DEEPSEEK_EXECUTABLE):
        log(f"ERROR: DeepSeek executable not found at {DEEPSEEK_EXECUTABLE}")
        return False
        
    # 检查是否有执行权限
    if not os.access(DEEPSEEK_EXECUTABLE, os.X_OK):
        log(f"Adding execute permission to {DEEPSEEK_EXECUTABLE}")
        try:
            # 添加执行权限
            os.chmod(DEEPSEEK_EXECUTABLE, 0o755)  # rwxr-xr-x
            if not os.access(DEEPSEEK_EXECUTABLE, os.X_OK):
                log("Failed to add execute permission")
                return False
            log("Execute permission added successfully")
        except Exception as e:
            log(f"Error adding execute permission: {e}")
            return False
    else:
        log("DeepSeek executable already has execute permission")
        
    # 检查模型文件是否存在
    if not os.path.exists(DEEPSEEK_MODEL_PATH):
        log(f"ERROR: DeepSeek model not found at {DEEPSEEK_MODEL_PATH}")
        return False
        
    log("DeepSeek executable and model checked and ready")
    return True

# 加载配置文件
def load_config(config_file="config/system_config.json"):
    try:
        log(f"Loading config from {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            log(f"Config loaded: {config}")
            return config
    except FileNotFoundError:
        # 创建默认配置
        log(f"Config file not found, creating default")
        default_config = {
            "active_agent": "default",
            "manager_agent": "manager",
            "record_duration": 5,
            "language": "zh"
        }
        Path(config_file).parent.mkdir(exist_ok=True)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        return default_config

# 加载Agent配置
def load_agent(agent_name):
    """
    加载智能体配置，如果缓存中存在则从缓存加载
    """
    global AGENT_CACHE
    
    # 先检查缓存
    if agent_name in AGENT_CACHE:
        log(f"Agent '{agent_name}' loaded from cache")
        return AGENT_CACHE[agent_name]
    
    try:
        agent_file = f"{AGENTS_DIR}/{agent_name}.json"
        log(f"Loading agent from {agent_file}")
        with open(agent_file, "r", encoding="utf-8") as f:
            agent = json.load(f)
            log(f"Agent loaded: {agent['name']}")
            
            # 添加到缓存
            AGENT_CACHE[agent_name] = agent
            
            return agent
    except FileNotFoundError:
        # 创建默认Agent
        log(f"Agent file not found, creating default")
        default_agent = {
            "name": agent_name,
            "prompt": "You are Edge Voice Assistant. Please provide brief answers to user questions.",
            "knowledge_base": ["general"],
            "tools": ["weather"]
        }
        Path(f"{AGENTS_DIR}").mkdir(exist_ok=True)
        with open(f"{AGENTS_DIR}/{agent_name}.json", "w", encoding="utf-8") as f:
            json.dump(default_agent, f, ensure_ascii=False, indent=2)
        
        # 添加到缓存
        AGENT_CACHE[agent_name] = default_agent
        
        return default_agent

# 新增: 基于LLM的智能体选择器
def agent_selector(query, available_agents=None):
    """
    An intelligent agent selector that uses a dedicated lightweight model approach
    to select the most appropriate agent for a user query.
    
    Args:
        query: The user's query text
        available_agents: List of available agent names (if None, will be auto-loaded)
        
    Returns:
        Selected agent name
    """
    log("Starting agent selector for query")
    
    if not query or query.strip() == "":
        log("Empty query, defaulting to default agent")
        return "default"
        
    try:
        # Get available agents if not provided
        if not available_agents:
            available_agents = []
            # List all agent JSON files in the agents directory
            for file_name in os.listdir(AGENTS_DIR):
                if file_name.endswith('.json'):
                    agent_name = file_name[:-5]  # Remove .json extension
                    if agent_name != "manager":  # Skip manager agent
                        available_agents.append(agent_name)
                        
            log(f"Found available agents: {available_agents}")
            
        # If no agents available, use default
        if not available_agents:
            log("No available agents found, using default")
            return "default"
            
        # 首先尝试使用关键词匹配
        log("尝试使用关键词匹配选择智能体")
        try:
            manager_agent = load_agent("manager")
            if "routing_rules" in manager_agent:
                for rule in manager_agent["routing_rules"]:
                    agent_name = rule.get("agent")
                    if agent_name in available_agents:
                        keywords = rule.get("keywords", [])
                        for keyword in keywords:
                            if keyword.lower() in query.lower():
                                log(f"根据关键词'{keyword}'选择智能体: {agent_name}")
                                return agent_name
            log("关键词匹配未找到匹配的智能体")
        except Exception as e:
            log(f"关键词匹配时出错: {e}")
            
        # 如果关键词匹配失败，再使用DeepSeek模型
        log("使用DeepSeek模型选择智能体")
            
        # Create the selector prompt
        prompt = "You are an intelligent Agent Selector that analyzes user queries to route them to the most appropriate specialized agent. Your job is to understand the user's intent and select the best agent to handle their request. Consider the context of the conversation and be extremely concise in your selection.\n\n"
        prompt += "When analyzing queries:\n"
        prompt += "1. Determine the core intent and domain of the question\n"
        prompt += "2. Match it to the most relevant agent based on their expertise\n"
        prompt += "3. Only return the name of the selected agent with no additional text\n\n"
        prompt += "For example, if given 'What's the weather like?', just respond with 'weather'.\n\n"
        
        # Add available agents with descriptions
        prompt += "Available agents:\n"
        for agent_name in available_agents:
            try:
                agent_data = load_agent(agent_name)
                agent_display_name = agent_data.get("name", agent_name)
                prompt += f"- {agent_name}: {agent_display_name}\n"
            except:
                prompt += f"- {agent_name}: {agent_name.capitalize()} Assistant\n"
                
        # Add the current query
        prompt += f"\n\nCurrent query: {query}\n\n"
        prompt += "Selected agent:"
        
        # Save prompt to file for debugging
        with open("agent_selector_input.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
            
        log("Agent selector prompt saved to agent_selector_input.txt")
        
        # Call the model (using the same DeepSeek backend)
        log("Calling model for agent selection")
        response = query_deepseek("", prompt).strip()
        
        log(f"Agent selector raw response: '{response}'")
        
        # Extract just the agent name (first word/line)
        if response:
            # Clean up response - get first line and first word
            response_text = response.strip().lower()
            
            # 首先尝试直接匹配
            for agent in available_agents:
                if agent in response_text:
                    log(f"直接匹配到智能体: {agent}")
                    return agent
            
            # 尝试获取第一行第一个词作为智能体名称
            selected_agent = response_text.split('\n')[0].split()[0].lower() if response_text else ""
            
            # 验证选择的智能体是否存在于可用列表中
            if selected_agent in available_agents:
                log(f"Agent selector selected: {selected_agent}")
                return selected_agent
            else:
                # 如果模型选择失败，使用default
                log(f"无法匹配任何智能体，使用default")
                return "default"
        else:
            log("Empty response from agent selector, using default")
            return "default"
            
    except Exception as e:
        log(f"Error in agent selector: {e}")
        import traceback
        traceback.print_exc()
        return "default"

# 修改: 更新select_agent_for_query函数使用新的agent_selector
def select_agent_for_query(query, config):
    """
    根据用户的查询，选择最合适的智能体
    现在使用专门的智能体选择器而不是简单的关键词匹配
    """
    global CURRENT_AGENT
    
    # 获取默认智能体名称
    default_agent_name = config.get("active_agent", "default")
    
    # 使用智能体选择器选择最合适的智能体
    try:
        log("使用智能体选择器分析查询")
        
        # 获取所有可用的智能体
        available_agents = []
        for file_name in os.listdir(AGENTS_DIR):
            if file_name.endswith('.json'):
                agent_name = file_name[:-5]  # 移除.json扩展名
                if agent_name != "manager":  # 跳过manager智能体
                    available_agents.append(agent_name)
        
        # 使用智能体选择器
        selected_agent = agent_selector(query, available_agents)
        
        # 保存当前选择的智能体
        CURRENT_AGENT = selected_agent
        return selected_agent
    
    except Exception as e:
        # 出现错误时使用默认智能体
        log(f"智能体路由发生错误: {e}，使用默认智能体: {default_agent_name}")
        CURRENT_AGENT = default_agent_name
        return default_agent_name

# 添加知识库检索函数
def get_knowledge_from_kb(agent, query):
    """
    从智能体关联的知识库中检索相关知识
    
    Args:
        agent: 智能体配置
        query: 用户查询
        
    Returns:
        格式化的知识文本，如果没有相关知识则返回空字符串
    """
    if not KNOWLEDGE_BASE_AVAILABLE or not KB_MANAGER:
        log("知识库功能不可用")
        return ""
    
    # 获取智能体关联的知识库
    knowledge_bases = agent.get("knowledge_base", [])
    if not knowledge_bases:
        log("智能体未关联知识库")
        return ""
    
    try:
        log(f"为查询 '{query}' 从知识库 {knowledge_bases} 中检索信息")
        all_docs = []
        
        # 从每个关联的知识库中检索文档
        for kb_name in knowledge_bases:
            if kb_name == "general":
                # general是特殊的知识库名称，表示不需要专门的知识
                continue
                
            # 检索文档
            docs = KB_MANAGER.search_knowledge_base(kb_name, query, k=3)
            if docs:
                log(f"从知识库 '{kb_name}' 检索到 {len(docs)} 个相关文档")
                all_docs.extend(docs)
        
        # 格式化文档
        if all_docs:
            formatted_docs = format_documents_for_prompt(all_docs)
            log(f"生成的知识提示词长度: {len(formatted_docs)}")
            return formatted_docs
        else:
            log("未检索到相关知识")
            return ""
    except Exception as e:
        log(f"知识库检索出错: {e}")
        import traceback
        traceback.print_exc()
        return ""

# 修改DeepSeek调用函数，添加知识增强
def query_deepseek(prompt, query, timeout=DEEPSEEK_WAIT_TIMEOUT, agent_config=None):
    """
    使用交互式方式调用DeepSeek模型
    这种方法模拟终端交互，与直接在终端中使用相同
    """
    global QUERY_CACHE
    
    # 首先检查可执行文件权限
    if not check_deepseek_executable():
        log("DeepSeek executable check failed")
        # 更新处理状态 - 模型错误
        try:
            with open("processing_status.txt", "w") as f:
                f.write("LLM_CONFIG_ERROR\n")
        except:
            pass
        return "Sorry, there was a problem with the DeepSeek model configuration."
        
    # 检查空白输入
    if not query or query.strip() == "" or query.strip() == "[BLANK_AUDIO]":
        log(f"检测到空白输入: '{query}'，返回默认回复")
        # 更新处理状态 - 空白输入
        try:
            with open("processing_status.txt", "w") as f:
                f.write("EMPTY_INPUT\n")
        except:
            pass
        return "I didn't hear anything. Could you please speak again?"
    
    # 检查是否只包含少量字符或噪声字符
    if len(query.strip()) < 3 or query.strip().lower() in ['a', 'an', 'the', 'um', 'ah', 'eh', 'oh']:
        log(f"检测到可能的噪声输入: '{query}'，返回默认回复")
        # 更新处理状态 - 噪声输入
        try:
            with open("processing_status.txt", "w") as f:
                f.write("NOISE_INPUT\n")
        except:
            pass
        return "I didn't hear anything clearly. Could you please speak again?"
    
    # 处理和规范化查询
    normalized_query = query.strip().lower()
    
    # 检查缓存中是否有此查询的回复
    if normalized_query in QUERY_CACHE:
        cached_response = QUERY_CACHE[normalized_query]
        log(f"从缓存获取回复: '{normalized_query}' -> '{cached_response}'")
        # 更新处理状态 - 缓存命中
        try:
            with open("processing_status.txt", "w") as f:
                f.write("CACHE_HIT\n")
        except:
            pass
        return cached_response
        
    child = None
    try:
        # 准备查询 - 增强提示词，加入知识库检索结果
        user_query = query.strip()
        log(f"准备发送查询: '{user_query}'")
        
        # 构建完整提示词，包含智能体的基础提示词
        full_prompt = prompt
        
        # 如果有智能体配置，检索相关知识并添加到提示词中
        if agent_config and KNOWLEDGE_BASE_AVAILABLE:
            # 从知识库检索相关知识
            knowledge_text = get_knowledge_from_kb(agent_config, user_query)
            if knowledge_text:
                log("将知识库检索结果添加到提示词")
                full_prompt += knowledge_text
        
        # 添加用户查询
        if full_prompt:
            # 如果有提示词，使用完整格式
            final_prompt = f"{full_prompt}\n\nUser Query: {user_query}\nAnswer:"
        else:
            # 如果没有提示词，直接发送查询
            final_prompt = user_query
            
        log(f"最终提示词长度: {len(final_prompt)}")
        
        # 使用pexpect模拟交互式终端
        try:
            import pexpect
        except ImportError:
            log("需要安装pexpect库: pip3 install pexpect")
            os.system("pip3 install pexpect")  # 尝试自动安装
            import pexpect
        
        # 准备调用DeepSeek模型
        executable_dir = os.path.dirname(DEEPSEEK_EXECUTABLE)
        executable_name = os.path.basename(DEEPSEEK_EXECUTABLE)
        model_path = DEEPSEEK_MODEL_PATH
        
        # 构建命令 - 直接进入交互模式，使用优化后的参数
        cmd = f"cd {executable_dir} && ./{executable_name} {model_path} {DEEPSEEK_MAX_NEW_TOKENS} {DEEPSEEK_TOP_P}"
        log(f"Executing command: {cmd}")
        
        # 启动进程
        log("Starting DeepSeek process in interactive mode...")
        start_time = time.time()
        
        # 使用pexpect创建子进程
        child = pexpect.spawn('/bin/bash', ['-c', cmd], encoding='utf-8')
        
        # 设置日志记录（仅在高级调试模式时）
        if DEBUG_MODE:
            log("Recording full model output...")
            child.logfile_read = sys.stdout
        
        # 等待模型加载完成并准备好接收输入
        log("Waiting for model to load...")
        
        # 等待模型初始化完成，显示菜单后会出现"user:"提示符
        idx = child.expect(["user:", pexpect.TIMEOUT, pexpect.EOF], timeout=timeout)
        
        # 关闭详细日志避免干扰输出解析
        if DEBUG_MODE:
            child.logfile_read = None
            
        if idx == 0:
            log("Model loaded, prompt found: user:")
        else:  # 超时或EOF
            log(f"Failed to get prompt from model: {child.before}")
            try:
                child.close(force=True)
            except:
                log("Note: Error closing child process, but this doesn't affect the test result")
            return "Sorry, the DeepSeek model is taking too long to respond."
        
        log(f"Sending query: '{user_query}'")
        
        # 发送查询 - 发送完整提示词
        child.sendline(final_prompt)
        
        # 获取输出，等待模型生成完成
        all_output = ""
        log("Waiting for model response...")
        
        # 设置最大等待时间（秒）
        wait_interval = 0.5  # 每次等待0.5秒，更频繁检查
        max_total_wait = timeout  # 最多等待设定的timeout时间
        total_waited = 0
        last_output_time = time.time()
        
        # 储存是否看到了robot:标记
        seen_robot_marker = False
        seen_think_marker = False
        think_end_marker = False
        early_valid_response = False
        valid_response_content = ""
        
        # 连续收集输出，不使用任何匹配模式，只是简单读取所有可用内容
        while total_waited < max_total_wait:
            try:
                # 调用read_nonblocking读取任何可用输出
                try:
                    chunk = child.read_nonblocking(size=4096, timeout=wait_interval)
                    if chunk:
                        all_output += chunk
                        last_output_time = time.time()
                        
                        # 尝试检测有效回复的早期信号，例如完整句子的结束
                        # 只有在EARLY_DETECTION启用时才检查
                        if DEEPSEEK_EARLY_DETECTION and len(all_output) > DEEPSEEK_MIN_VALID_RESPONSE:
                            # 检查完整句子的结束标志
                            sentences_ended = re.findall(r'[\.\?\!]\s+[A-Z]|[\.\?\!]$', all_output)
                            if len(sentences_ended) > 0:
                                valid_response_content = all_output
                                early_valid_response = True
                                log("检测到早期完整回复，可以提前结束")
                        
                        # 更高级的响应解析逻辑
                        # 检查是否看到了robot:标记
                        if 'robot:' in all_output.lower() and not seen_robot_marker:
                            seen_robot_marker = True
                            log("检测到robot:标记")
                        
                        # 检查是否看到了think标记
                        if '<think>' in all_output.lower() and not seen_think_marker:
                            seen_think_marker = True
                            log("检测到<think>标记，模型正在思考")
                            
                        # 检测思考结束
                        if seen_think_marker and '</think>' in all_output.lower() and not think_end_marker:
                            think_end_marker = True
                            log("检测到</think>标记，模型完成思考")
                            
                        # 保存原始输出到文件，方便调试
                        try:
                            with open("deepseek_raw_output.txt", "w", encoding="utf-8") as f:
                                f.write(all_output)
                        except:
                            pass
                            
                        # 打印进度
                        if DEBUG_MODE:
                            output_preview = all_output[-100:] if len(all_output) > 100 else all_output
                            log(f"收到输出片段: '{output_preview.strip()}'")
                            
                        # 如果检测到了robot:标记或有效的早期回复，可以缩短等待时间
                        if seen_robot_marker or early_valid_response:
                            max_completion_wait = DEEPSEEK_COMPLETION_WAIT / 2  # 缩短完成等待时间
                        else:
                            max_completion_wait = DEEPSEEK_COMPLETION_WAIT
                            
                        # 提前检测可能的回复结束
                        if all_output and total_waited > 5:  # 至少等待5秒
                            if '?' in all_output[-2:] or '.' in all_output[-2:] or '!' in all_output[-2:]:
                                log("检测到可能的句子结束，但继续等待以确认...")
                                
                except pexpect.TIMEOUT:
                    # 超时但可能仍在生成，继续等待
                    if time.time() - last_output_time > DEEPSEEK_WITHOUT_OUTPUT_MAX:
                        log(f"超过{DEEPSEEK_WITHOUT_OUTPUT_MAX}秒无新输出，结束等待")
                        break
                    continue
                    
                # 等待短暂时间后继续循环
                time.sleep(wait_interval)
                total_waited += wait_interval
                
                # 检查是否需要提前结束等待
                if early_valid_response and time.time() - last_output_time > 2:
                    log("检测到有效回复，且2秒内没有新输出，提前结束等待")
                    break
                    
                # 检查是否需要进行思考超时检查
                if seen_think_marker and not think_end_marker and total_waited > DEEPSEEK_THINKING_WAIT:
                    log(f"思考阶段超过{DEEPSEEK_THINKING_WAIT}秒，结束等待")
                    break
                    
                # 检查是否已达到最大等待时间的90%
                if total_waited > max_total_wait * 0.9:
                    log(f"接近总超时限制({max_total_wait}秒的{int(total_waited)}秒)，准备结束等待")
                    
            except pexpect.EOF:
                # EOF表示进程已结束
                log("读取输出时收到EOF，进程可能已结束")
                break
            except Exception as e:
                # 其他错误
                log(f"读取输出时出错: {e}")
                break
                
        # 尝试关闭子进程
        try:
            log("正常关闭子进程")
            child.close(force=True)
        except:
            log("关闭子进程时出错，但不影响运行")
        
        # 记录总等待时间
        log(f"总等待时间: {total_waited:.1f}秒")
        
        # 更高级的输出处理，优化提取模型回复的逻辑
        try:
            # 先保存原始输出，便于调试
            with open("deepseek_raw_output.txt", "w", encoding="utf-8") as f:
                f.write(all_output)
            
            # 清理输出以提取实际回复
            # 策略1：提取robot:后的内容
            if 'robot:' in all_output.lower():
                response = all_output.lower().split('robot:')[1].strip()
                log("使用robot:标记提取回复")
            # 策略2：使用思考标签
            elif '<think>' in all_output.lower() and '</think>' in all_output.lower():
                # 尝试提取思考后的实际回复
                post_think_content = all_output.lower().split('</think>')[1].strip()
                if post_think_content:
                    response = post_think_content
                    log("使用思考标签后的内容提取回复")
                else:
                    # 如果思考后没有内容，使用全部输出
                    response = all_output.strip()
                    log("思考后无内容，使用全部输出")
            # 策略3：如果有早期检测到的有效回复
            elif early_valid_response and valid_response_content:
                response = valid_response_content.strip()
                log("使用早期检测到的有效回复")
            # 策略4：使用整个输出，但尝试去除用户输入
            else:
                # 尝试去除用户输入部分
                if f"user: {query}" in all_output.lower():
                    response = all_output.lower().split(f"user: {query}")[1].strip()
                    log("去除用户输入后提取回复")
                else:
                    # 最后的方案：使用全部输出
                    response = all_output.strip()
                    log("使用全部原始输出作为回复")
            
            # 额外的清理和格式化
            # 去除特殊标记和前缀
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL|re.IGNORECASE)
            response = re.sub(r'robot:', '', response, flags=re.IGNORECASE)
            response = re.sub(r'assistant:', '', response, flags=re.IGNORECASE)
            
            # 确保回复不太长，如果需要截断，在句子边界处截断
            if len(response) > DEEPSEEK_MAX_RESPONSE_LENGTH:  # 限制为最多200个字符
                log(f"回复过长({len(response)}字符)，将进行智能截断")
                # 找到最后一个句子边界
                sentence_boundaries = list(re.finditer(r'[\.\?\!]\s', response[:DEEPSEEK_MAX_RESPONSE_LENGTH]))
                if sentence_boundaries:
                    last_boundary = sentence_boundaries[-1].end()
                    response = response[:last_boundary].strip()
                    log(f"在句子边界处截断，保留{len(response)}字符")
                else:
                    # 如果没找到句子边界，直接截断
                    response = response[:DEEPSEEK_MAX_RESPONSE_LENGTH].strip()
                    log("未找到句子边界，直接截断至200字符")
            
            # 最后的清理
            response = response.strip()
            
        except Exception as e:
            log(f"处理输出时出错: {e}")
            response = all_output.strip() if all_output else "Sorry, I couldn't generate a proper response."
        
        # 如果不是来自缓存，且是个新查询，就添加到缓存
        if normalized_query not in QUERY_CACHE and len(response) > 10:
            # 防止缓存过大，如果超过最大大小，移除最旧的项
            if len(QUERY_CACHE) >= MAX_CACHE_SIZE:
                # 简单方案：完全清除缓存
                # QUERY_CACHE.clear()
                # 更好的方案：只移除一个最旧的项（需要改变数据结构）
                # 这里我们暂时随机移除一个
                if QUERY_CACHE:
                    for _ in range(max(1, len(QUERY_CACHE) - MAX_CACHE_SIZE + 1)):
                        QUERY_CACHE.pop(next(iter(QUERY_CACHE)))
            
            log(f"将查询添加到缓存: '{normalized_query}'")
            QUERY_CACHE[normalized_query] = response

        # 保存对话历史到conversation_context.json
        try:
            # 读取现有历史（如果有）
            conversation_history = []
            try:
                with open("conversation_context.json", "r") as f:
                    conversation_history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                conversation_history = []
            
            # 添加新的对话
            conversation_history.append({
                "query": query,
                "agent": CURRENT_AGENT,
                "response": response,
                "timestamp": time.time()
            })
            
            # 仅保留最近的5条记录
            if len(conversation_history) > 5:
                conversation_history = conversation_history[-5:]
            
            # 保存更新后的历史
            with open("conversation_context.json", "w") as f:
                json.dump(conversation_history, f, indent=2)
            
            log(f"更新了对话历史，当前有 {len(conversation_history)} 条记录")
        except Exception as e:
            log(f"保存对话历史时出错: {e}")
        
        # 在函数结尾处，返回响应前，添加状态更新
        try:
            with open("processing_status.txt", "w") as f:
                f.write("LLM_RESPONSE_COMPLETE\n")
        except:
            log("无法更新处理状态文件")
        
        return response
        
    except Exception as e:
        log(f"DeepSeek model call failed: {e}")
        import traceback
        log(traceback.format_exc())
        return "Sorry, I cannot answer that question right now."
    finally:
        # 确保子进程被终止
        try:
            if child and hasattr(child, 'isalive') and child.isalive():
                log("Final attempt to terminate child process")
                try:
                    child.terminate(force=True)
                except:
                    pass
                    
            if hasattr(child, 'pid'):
                try:
                    import signal
                    os.kill(child.pid, signal.SIGKILL)
                    log(f"发送SIGKILL到进程 {child.pid}")
                except Exception as e:
                    log(f"无法终止进程 {child.pid}：{e}")
        except:
            log("Final termination attempt failed, continuing")

# 检查wav文件是否存在和有效
def check_wav_file(file_path):
    """Check if WAV file exists and is valid"""
    if not os.path.exists(file_path):
        log(f"ERROR: WAV file does not exist: {file_path}")
        return False
    
    try:
        data, sr = sf.read(file_path, dtype='int16')
        log(f"WAV file validated: {file_path}, samples: {len(data)}, sr: {sr}")
        return True
    except Exception as e:
        log(f"ERROR: Invalid WAV file: {file_path}, error: {e}")
        return False

# 检查文本是否为英文
def is_english(text):
    """Check if text contains only English characters and common symbols"""
    import re
    # 允许英文字母、数字、常用标点和空格
    return bool(re.match(r'^[a-zA-Z0-9\s.,!?;:\'"-_+*&%$#@()\[\]{}]+$', text))

# 预热DeepSeek模型，减少第一次查询时的加载时间
def warmup_deepseek_model(timeout=30):
    """
    预热DeepSeek模型，以缩短首次查询的响应时间
    缩短预热超时时间到30秒
    """
    global MODEL_WARMED_UP
    
    if MODEL_WARMED_UP:
        log("模型已预热，跳过")
        return True
    
    log("开始预热DeepSeek模型...")
    
    # 检查模型和可执行文件
    if not check_deepseek_executable():
        log("DeepSeek可执行文件或模型检查失败，无法预热")
        return False
    
    try:
        # 更新处理状态 - 模型预热
        try:
            with open("processing_status.txt", "w") as f:
                f.write("MODEL_WARMING_UP\n")
        except:
            pass
            
        # 使用简单问题进行预热 - 使用更短的查询
        warmup_query = "Hello"  # 使用最简单的查询加速预热
        log(f"使用预热查询: '{warmup_query}'")
        
        # 设置预热超时保护
        start_time = time.time()
        
        # 创建预热线程
        warmup_thread = threading.Thread(
            target=lambda: query_deepseek("", warmup_query, timeout=timeout)
        )
        warmup_thread.daemon = True
        warmup_thread.start()
        
        # 等待预热完成或超时 - 使用更短的等待时间
        warmup_thread.join(timeout=min(20, timeout))  # 最多等待20秒
        
        # 检查是否超时
        if warmup_thread.is_alive():
            log(f"预热未完成但已达到足够程度，继续运行")
            # 注意：我们不中断线程，让它在后台继续执行
        
        # 设置预热标志
        MODEL_WARMED_UP = True
        
        # 更新处理状态 - 预热完成
        try:
            with open("processing_status.txt", "w") as f:
                f.write("MODEL_WARMED_UP\n")
        except:
            pass
            
        execution_time = time.time() - start_time
        log(f"模型预热完成，耗时: {execution_time:.2f}秒")
        return True
    except Exception as e:
        log(f"模型预热失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 更新处理状态 - 预热失败
        try:
            with open("processing_status.txt", "w") as f:
                f.write("MODEL_WARMUP_FAILED\n")
        except:
            pass
            
        # 即使预热失败，我们仍然继续运行
        # 但不设置预热标志，以便下次再次尝试
        return False

# 主流程函数
def process_voice_query():
    # 加载配置
    config = load_config()
    
    # 获取录音和播放时长
    duration = config.get("record_duration", DURATION)
    playback_duration = config.get("playback_duration", PLAYBACK_DURATION)
    
    # 更新处理状态 - 开始
    try:
        with open("processing_status.txt", "w") as f:
            f.write("RECORDING\n")
    except:
        pass
        
    try:
        # 录制音频
        log(f"Starting audio recording for {duration} seconds...")
        try:
            record_audio(duration, SAMPLE_RATE, CHANNELS, OUTPUT_FILE)
            log("Recording completed successfully")
            
            # 更新处理状态 - 录音完成
            with open("processing_status.txt", "w") as f:
                f.write("RECORDING_COMPLETE\n")
        except Exception as e:
            log(f"Error during recording: {e}")
            import traceback
            log(traceback.format_exc())
            return False
        
        # 验证录音文件
        if not check_wav_file(OUTPUT_FILE):
            log("Recording verification failed")
            return False
        
        # 更新处理状态 - 开始语音识别
        with open("processing_status.txt", "w") as f:
            f.write("ASR_PROCESSING\n")
            
        # 语音识别
        log("Starting speech recognition...")
        try:
            query = run_asr(OUTPUT_FILE)
            log(f"User query: {query}")
            
            # 更新处理状态 - 语音识别完成
            with open("processing_status.txt", "w") as f:
                f.write(f"ASR_COMPLETE\nQUERY: {query}\n")
                
            # 创建输入文件以供GUI读取
            with open("deepseek_input.txt", "w", encoding="utf-8") as f:
                f.write("User: " + query)
        except Exception as e:
            log(f"Error during ASR: {e}")
            import traceback
            log(traceback.format_exc())
            return False
        
        # 处理特殊标记或错误
        if not query or query.strip() == "" or query == "[BLANK_AUDIO]" or query == "[ERROR]":
            log("Empty or error ASR result, using default response")
            response = "I didn't hear anything. Could you please speak again?"
            
            # 更新处理状态 - 空输入响应
            with open("processing_status.txt", "w") as f:
                f.write("EMPTY_INPUT\n")
                
            # 写入调试输出文件以供GUI读取
            with open("debug_deepseek_response.txt", "w", encoding="utf-8") as f:
                f.write("使用智能体: default\nDeepSeek完整响应: " + response)
                
            # 直接执行TTS
            try:
                log(f"Sending to TTS: '{response}'")
                run_tts(response, playback_duration)
                log("TTS completed successfully")
                
                # 验证TTS输出文件并播放
                if check_wav_file(TTS_AUDIO_OUTPUT):
                    play_audio(TTS_AUDIO_OUTPUT)
                    log("Audio playback completed for empty input response")
                else:
                    log("TTS output verification failed for empty input")
            except Exception as e:
                log(f"Error handling empty input: {e}")
            return True
        
        # 检查是否为英文，如果不是，返回错误提示
        if not is_english(query):
            log("Non-English query detected, model only supports English")
            response = "Sorry, this assistant currently only supports English. Please ask your question in English."
            
            # 更新处理状态 - 语言错误
            with open("processing_status.txt", "w") as f:
                f.write("LANGUAGE_ERROR\n")
                
            # 写入调试输出文件以供GUI读取
            with open("debug_deepseek_response.txt", "w", encoding="utf-8") as f:
                f.write("使用智能体: default\nDeepSeek完整响应: " + response)
                
            # 语音合成
            log("Starting speech synthesis for language error message...")
            try:
                run_tts(response, playback_duration)
                log("TTS for language error completed")
            except Exception as e:
                log(f"Error during TTS for language error: {e}")
                import traceback
                log(traceback.format_exc())
                return False
            
            # 验证TTS输出文件
            if not check_wav_file(TTS_AUDIO_OUTPUT):
                log("TTS output verification failed")
                return False
                
            # 播放回答
            log("Playing audio response...")
            try:
                play_audio(TTS_AUDIO_OUTPUT)
                log("Audio playback completed")
            except Exception as e:
                log(f"Error during audio playback: {e}")
                import traceback
                log(traceback.format_exc())
            return True
        
        # 更新处理状态 - 智能体选择
        with open("processing_status.txt", "w") as f:
            f.write("AGENT_SELECTING\nSTAGE: ANALYZING\n")
            
        # 先使用manager智能体进行智能体选择
        manager_agent_name = config.get("manager_agent", "manager")
        log(f"使用管理智能体 '{manager_agent_name}' 进行智能体选择")
        
        try:
            # 更新处理状态 - 管理智能体开始分析
            with open("processing_status.txt", "w") as f:
                f.write("AGENT_SELECTING\nSTAGE: ANALYZING\n")
                
            # 加载manager智能体
            manager_agent = load_agent(manager_agent_name)
            if not manager_agent:
                log(f"无法加载管理智能体 {manager_agent_name}，使用默认选择方式")
                agent_name = select_agent_for_query(query, config)
            else:
                # 更新处理状态 - 管理智能体搜索可用智能体
                with open("processing_status.txt", "w") as f:
                    f.write("AGENT_SELECTING\nSTAGE: SEARCHING\n")
                    
                # 获取所有可用的智能体
                available_agents = []
                for file_name in os.listdir(AGENTS_DIR):
                    if file_name.endswith('.json'):
                        agent_name = file_name[:-5]  # 移除.json扩展名
                        if agent_name != manager_agent_name:  # 排除manager自身
                            available_agents.append(agent_name)
                
                # 更新处理状态 - 管理智能体决策中
                with open("processing_status.txt", "w") as f:
                    f.write(f"AGENT_SELECTING\nSTAGE: DECIDING\nAGENTS: {','.join(available_agents)}\n")
                    
                # 使用agent_selector函数，但明确指定这是manager在做选择
                log(f"管理智能体正在从 {available_agents} 中选择合适的智能体处理查询")
                agent_name = agent_selector(query, available_agents)
                log(f"管理智能体选择了: {agent_name}")
        except Exception as e:
            log(f"管理智能体选择过程出错: {e}，回退到默认选择方式")
            agent_name = select_agent_for_query(query, config)
        
        # 加载选定的智能体
        agent = load_agent(agent_name)
        log(f"选择智能体 '{agent_name}' 处理查询")
        
        # 更新处理状态 - 智能体已选择
        with open("processing_status.txt", "w") as f:
            f.write(f"AGENT_SELECTED\nAGENT: {agent_name}\n")
            
        # 获取智能体显示名称
        agent_display_name = agent.get("name", agent_name)
            
        # 写入调试输出文件以供GUI读取
        with open("debug_deepseek_response.txt", "w", encoding="utf-8") as f:
            f.write(f"使用智能体: {agent_name}\n智能体显示名称: {agent_display_name}\n")
        
        # 处理响应
        if FAST_RESPONSE_MODE:
            # 快速响应模式 - 使用预定义回答
            log("Fast response mode enabled, using predefined responses")
            
            # 更新处理状态 - 快速响应模式
            with open("processing_status.txt", "w") as f:
                f.write("FAST_RESPONSE_MODE\n")
                
            # 清理和准备查询文本
            clean_query = query.lower().strip()
            response = DEFAULT_RESPONSE
            
            # 尝试匹配预定义回答
            for key, resp in PREDEFINED_RESPONSES.items():
                if key in clean_query:
                    response = resp
                    log(f"Matched keyword '{key}', using response: {response}")
                    break
                    
            log(f"Assistant response (fast mode): {response}")
        else:
            # 更新处理状态 - 调用大模型
            with open("processing_status.txt", "w") as f:
                f.write("LLM_PROCESSING\n")
                
            # 调用大模型
            log("Calling DeepSeek model...")
            try:
                # 从历史记录获取上下文（如果有）
                try:
                    with open("conversation_context.json", "r") as f:
                        conversation_history = json.load(f)
                        # 构建上下文字符串
                        context = ""
                        if conversation_history:
                            # 最多使用最近的3条对话
                            recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
                            for item in recent_history:
                                context += f"User: {item['query']}\nAssistant ({item['agent']}): {item['response']}\n\n"
                except:
                    context = ""
                
                # 构建完整的提示词，包含智能体的提示词和上下文
                full_prompt = agent.get("prompt", "")
                if context:
                    full_prompt += "\n\n" + context
                
                # 创建输入文件
                with open("deepseek_input.txt", "w", encoding="utf-8") as f:
                    f.write(full_prompt)
                
                # 检查是否有缓存结果
                normalized_query = query.lower().strip()
                if normalized_query in QUERY_CACHE:
                    response = QUERY_CACHE[normalized_query]
                    log(f"从缓存获取回复: '{normalized_query}' -> '{response}'")
                    
                    # 更新处理状态 - 缓存命中
                    with open("processing_status.txt", "w") as f:
                        f.write("CACHE_HIT\n")
                else:
                    # 更新处理状态 - 模型思考中
                    with open("processing_status.txt", "w") as f:
                        f.write("LLM_THINKING\n")
                        
                    # 异步调用DeepSeek模型，传入智能体配置以支持知识库检索
                    future = async_query_deepseek(agent.get("prompt", ""), query, agent_config=agent)
                    if future:
                        # 使用异步调用，设置超时保护
                        try:
                            from concurrent.futures import TimeoutError, as_completed
                            log("等待异步查询结果...")
                            
                            # 此处实现一个轮询机制，每隔0.5秒检查一次结果（原来是1秒）
                            # 这样可以更快检测到完成的任务
                            max_wait = DEEPSEEK_WAIT_TIMEOUT
                            waited = 0
                            done = False
                            
                            # 使用更短的检查间隔，加快响应速度
                            check_interval = 0.5  # 每0.5秒检查一次（原来是1秒）
                            
                            while waited < max_wait and not done:
                                try:
                                    # 检查是否已完成
                                    if future.done():
                                        response = future.result()
                                        log(f"异步查询完成: {response[:50]}...")
                                        done = True
                                        break
                                        
                                    # 每0.5秒钟检查一次
                                    time.sleep(check_interval)
                                    waited += check_interval
                                    
                                    # 每3秒显示等待消息和更新状态（原来是5秒）
                                    if waited % 3 < check_interval:
                                        log(f"已等待 {waited} 秒...")
                                        with open("processing_status.txt", "w") as f:
                                            f.write(f"LLM_THINKING\nWAIT_TIME: {waited}\n")
                                            
                                    # 如果等待超过30秒，尝试提前返回部分结果
                                    if waited > 30 and not done:
                                        try:
                                            # 尝试读取deepseek_raw_output.txt看是否有部分输出
                                            partial_response = ""
                                            try:
                                                with open("deepseek_raw_output.txt", "r", encoding="utf-8") as f:
                                                    partial_content = f.read()
                                                    if len(partial_content) > 20:  # 有足够内容
                                                        log("检测到部分输出，提前返回部分结果")
                                                        partial_response = partial_content[:200] + "..." if len(partial_content) > 200 else partial_content
                                                        done = True
                                                        response = partial_response
                                                        break
                                            except:
                                                pass
                                        except:
                                            pass
                                        
                                except Exception as e:
                                    log(f"检查异步结果时出错: {e}")
                                    break
                                    
                            # 如果超时仍未完成，给出合理错误提示
                            if not done:
                                log("异步查询超时，但允许继续等待")
                                
                                # 更新处理状态 - 模型响应超时
                                with open("processing_status.txt", "w") as f:
                                    f.write("LLM_TIMEOUT\n")
                                    
                                try:
                                    # 继续等待，但给出合理回复
                                    response = "I'm thinking about your question. Please wait a moment or ask me again."
                                finally:
                                    # 不取消任务，让它在后台继续运行，结果会被缓存
                                    pass
                                    
                        except Exception as e:
                            log(f"等待异步结果时发生异常: {e}")
                            response = "Sorry, I encountered a problem while processing your request."
                    else:
                        log("异步调用失败，尝试同步调用")
                        
                        # 更新处理状态 - 同步调用
                        with open("processing_status.txt", "w") as f:
                            f.write("SYNC_PROCESSING\n")
                            
                        # 同步调用
                        response = query_deepseek(agent.get("prompt", ""), query)
            except Exception as e:
                log(f"调用DeepSeek模型时出错: {e}")
                import traceback
                log(traceback.format_exc())
                
                # 更新处理状态 - 模型错误
                with open("processing_status.txt", "w") as f:
                    f.write(f"LLM_ERROR\nERROR: {str(e)}\n")
                    
                response = "I'm sorry, I encountered an error while processing your request."
                
            log(f"Assistant response: {response}")
            
            # 如果不是来自缓存，且是个新查询，就添加到缓存
            if normalized_query not in QUERY_CACHE and len(response) > 10:
                # 防止缓存过大，如果超过最大大小，移除最旧的项
                if len(QUERY_CACHE) >= MAX_CACHE_SIZE:
                    # 简单方案：完全清除缓存
                    # QUERY_CACHE.clear()
                    # 更好的方案：只移除一个最旧的项（需要改变数据结构）
                    # 这里我们暂时随机移除一个
                    if QUERY_CACHE:
                        for _ in range(max(1, len(QUERY_CACHE) - MAX_CACHE_SIZE + 1)):
                            QUERY_CACHE.pop(next(iter(QUERY_CACHE)))
            
                log(f"将查询添加到缓存: '{normalized_query}'")
                QUERY_CACHE[normalized_query] = response

            # 保存对话历史到conversation_context.json
            try:
                # 读取现有历史（如果有）
                conversation_history = []
                try:
                    with open("conversation_context.json", "r") as f:
                        conversation_history = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    conversation_history = []
                
                # 添加新的对话
                conversation_history.append({
                    "query": query,
                    "agent": CURRENT_AGENT,
                    "response": response,
                    "timestamp": time.time()
                })
                
                # 仅保留最近的5条记录
                if len(conversation_history) > 5:
                    conversation_history = conversation_history[-5:]
                
                # 保存更新后的历史
                with open("conversation_context.json", "w") as f:
                    json.dump(conversation_history, f, indent=2)
                
                log(f"更新了对话历史，当前有 {len(conversation_history)} 条记录")
            except Exception as e:
                log(f"保存对话历史时出错: {e}")
        
        # 完整回复保存到调试文件
        with open("debug_deepseek_response.txt", "a", encoding="utf-8") as f:
            f.write(f"DeepSeek完整响应: {response}\n")
        
        # 更新处理状态 - 开始TTS
        with open("processing_status.txt", "w") as f:
            f.write("TTS_PROCESSING\n")
            
        # 语音合成
        log("Starting speech synthesis...")
        try:
            run_tts(response, playback_duration)
            log("TTS completed successfully")
            
            # 更新处理状态 - TTS完成
            with open("processing_status.txt", "w") as f:
                f.write("TTS_COMPLETE\n")
        except Exception as e:
            log(f"Error during TTS: {e}")
            import traceback
            log(traceback.format_exc())
            return False
        
        # 验证TTS输出文件
        if not check_wav_file(TTS_AUDIO_OUTPUT):
            log("TTS output verification failed")
            return False
            
        # 更新处理状态 - 开始播放
        with open("processing_status.txt", "w") as f:
            f.write("AUDIO_PLAYING\n")
            
        # 播放回答
        log("Playing audio response...")
        try:
            play_audio(TTS_AUDIO_OUTPUT)
            log("Audio playback completed")
            
            # 更新处理状态 - 播放完成
            with open("processing_status.txt", "w") as f:
                f.write("PROCESS_COMPLETE\n")
        except Exception as e:
            log(f"Error during audio playback: {e}")
            import traceback
            log(traceback.format_exc())
            
            # 更新处理状态 - 播放错误
            with open("processing_status.txt", "w") as f:
                f.write("PLAYBACK_ERROR\n")
            
            return False
        
        return True
    except Exception as e:
        log(f"处理语音查询时发生严重错误: {e}")
        import traceback
        log(traceback.format_exc())
        
        # 更新处理状态 - 处理失败
        try:
            with open("processing_status.txt", "w") as f:
                f.write(f"CRITICAL_ERROR\nERROR: {str(e)}\n")
        except:
            pass
            
        return False

# 作为独立程序运行时的入口
if __name__ == "__main__":
    log("Edge Voice Assistant starting")
    ensure_dirs()
    
    # 预热模型，加快首次响应速度
    warmup_deepseek_model()
    
    result = process_voice_query()
    log(f"Process completed with result: {result}")
    if not result:
        log("There was an error processing the voice query") 