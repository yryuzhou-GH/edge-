import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
import shutil
import paramiko
import threading

# 配置
CONFIG_DIR = "./config"
AGENTS_DIR = "./agents"
KNOWLEDGE_DIR = "./knowledge"
TOOLS_DIR = "./tools"

# 设备连接配置
DEFAULT_HOST = "192.168.1.100"
DEFAULT_PORT = 22
DEFAULT_USERNAME = "elf"
DEFAULT_PASSWORD = "123456"
REMOTE_PATH = "/home/elf/edge_assistant"

class PCConfigTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Edge智音 - PC配置工具")
        self.root.geometry("800x600")
        
        # 确保本地目录存在
        self.ensure_dirs()
        
        # 创建主框架
        self.create_ui()
        
        # 加载配置和Agent
        self.load_configs()
    
    def ensure_dirs(self):
        """确保所需目录存在"""
        for dir_path in [CONFIG_DIR, AGENTS_DIR, KNOWLEDGE_DIR, TOOLS_DIR]:
            Path(dir_path).mkdir(exist_ok=True)
    
    def create_ui(self):
        """创建用户界面"""
        # 创建notebook（选项卡）
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建Agent配置选项卡
        self.agent_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.agent_frame, text="Agent配置")
        
        # 创建连接选项卡
        self.connection_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.connection_frame, text="设备连接")

        # 创建路由配置选项卡
        self.routing_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.routing_frame, text="智能体路由配置")
        
        # 配置Agent选项卡
        self.setup_agent_tab()
        
        # 配置连接选项卡
        self.setup_connection_tab()

        # 配置路由选项卡
        self.setup_routing_tab()
    
    def setup_agent_tab(self):
        """设置Agent配置选项卡"""
        # 左侧Agent列表
        left_frame = ttk.Frame(self.agent_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        ttk.Label(left_frame, text="Agent列表").pack(anchor=tk.W)
        
        # Agent列表
        self.agent_listbox = tk.Listbox(left_frame, width=25, height=20)
        self.agent_listbox.pack(fill=tk.Y, expand=True)
        self.agent_listbox.bind('<<ListboxSelect>>', self.on_agent_select)
        
        # 添加/删除按钮
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="添加", command=self.add_agent).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="删除", command=self.delete_agent).pack(side=tk.LEFT, padx=5)
        
        # 右侧编辑区域
        right_frame = ttk.Frame(self.agent_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Agent名称
        name_frame = ttk.Frame(right_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="Agent名称:").pack(side=tk.LEFT)
        self.agent_name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=self.agent_name_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 提示词
        ttk.Label(right_frame, text="提示词:").pack(anchor=tk.W, pady=(10, 5))
        self.prompt_text = tk.Text(right_frame, height=10)
        self.prompt_text.pack(fill=tk.X)
        
        # 知识库
        ttk.Label(right_frame, text="知识库:").pack(anchor=tk.W, pady=(10, 5))
        
        kb_frame = ttk.Frame(right_frame)
        kb_frame.pack(fill=tk.X, pady=5)
        
        self.kb_listbox = tk.Listbox(kb_frame, height=5)
        self.kb_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        kb_btn_frame = ttk.Frame(kb_frame)
        kb_btn_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(kb_btn_frame, text="添加", command=self.add_knowledge).pack(pady=2)
        ttk.Button(kb_btn_frame, text="删除", command=self.delete_knowledge).pack(pady=2)
        
        # 工具集
        ttk.Label(right_frame, text="工具集:").pack(anchor=tk.W, pady=(10, 5))
        
        tools_frame = ttk.Frame(right_frame)
        tools_frame.pack(fill=tk.X, pady=5)
        
        self.tools_listbox = tk.Listbox(tools_frame, height=5)
        self.tools_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tools_btn_frame = ttk.Frame(tools_frame)
        tools_btn_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(tools_btn_frame, text="添加", command=self.add_tool).pack(pady=2)
        ttk.Button(tools_btn_frame, text="删除", command=self.delete_tool).pack(pady=2)
        
        # 保存按钮
        ttk.Button(right_frame, text="保存", command=self.save_agent).pack(anchor=tk.E, pady=10)
    
    def setup_connection_tab(self):
        """设置设备连接选项卡"""
        # 连接配置
        config_frame = ttk.LabelFrame(self.connection_frame, text="连接配置")
        config_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 主机地址
        host_frame = ttk.Frame(config_frame)
        host_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(host_frame, text="主机地址:").pack(side=tk.LEFT)
        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        ttk.Entry(host_frame, textvariable=self.host_var).pack(side=tk.LEFT, padx=5)
        
        # 端口
        port_frame = ttk.Frame(config_frame)
        port_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(port_frame, text="端口:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value=str(DEFAULT_PORT))
        ttk.Entry(port_frame, textvariable=self.port_var).pack(side=tk.LEFT, padx=5)
        
        # 用户名
        username_frame = ttk.Frame(config_frame)
        username_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(username_frame, text="用户名:").pack(side=tk.LEFT)
        self.username_var = tk.StringVar(value=DEFAULT_USERNAME)
        ttk.Entry(username_frame, textvariable=self.username_var).pack(side=tk.LEFT, padx=5)
        
        # 密码
        password_frame = ttk.Frame(config_frame)
        password_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(password_frame, text="密码:").pack(side=tk.LEFT)
        self.password_var = tk.StringVar(value=DEFAULT_PASSWORD)
        ttk.Entry(password_frame, textvariable=self.password_var, show="*").pack(side=tk.LEFT, padx=5)
        
        # 测试连接按钮
        ttk.Button(config_frame, text="测试连接", command=self.test_connection).pack(anchor=tk.W, pady=10)
        
        # 同步操作
        sync_frame = ttk.LabelFrame(self.connection_frame, text="同步操作")
        sync_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 选择管理智能体
        active_frame = ttk.Frame(sync_frame)
        active_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(active_frame, text="管理智能体:").pack(side=tk.LEFT)
        
        self.manager_agent_var = tk.StringVar()
        self.manager_agent_combo = ttk.Combobox(active_frame, textvariable=self.manager_agent_var)
        self.manager_agent_combo.pack(side=tk.LEFT, padx=5)
        
        # 活跃Agent
        active_agent_frame = ttk.Frame(sync_frame)
        active_agent_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(active_agent_frame, text="默认活跃Agent:").pack(side=tk.LEFT)
        
        self.active_agent_var = tk.StringVar()
        self.active_agent_combo = ttk.Combobox(active_agent_frame, textvariable=self.active_agent_var)
        self.active_agent_combo.pack(side=tk.LEFT, padx=5)
        
        # 同步按钮
        ttk.Button(sync_frame, text="同步配置到设备", command=self.sync_to_device).pack(anchor=tk.W, pady=5)
        
        # 状态和日志
        ttk.Label(sync_frame, text="同步日志:").pack(anchor=tk.W, pady=5)
        
        self.log_text = tk.Text(sync_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)

    def setup_routing_tab(self):
        """设置智能体路由配置选项卡"""
        # 左侧路由规则列表
        left_frame = ttk.Frame(self.routing_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        ttk.Label(left_frame, text="选择管理智能体:").pack(anchor=tk.W)
        
        self.routing_manager_var = tk.StringVar()
        self.routing_manager_combo = ttk.Combobox(left_frame, textvariable=self.routing_manager_var, width=23)
        self.routing_manager_combo.pack(anchor=tk.W, pady=5)
        self.routing_manager_combo.bind("<<ComboboxSelected>>", self.on_routing_manager_select)
        
        ttk.Label(left_frame, text="路由规则列表:").pack(anchor=tk.W, pady=(10, 5))
        
        # 路由规则列表
        self.routing_listbox = tk.Listbox(left_frame, width=25, height=15)
        self.routing_listbox.pack(fill=tk.Y, expand=True)
        self.routing_listbox.bind('<<ListboxSelect>>', self.on_routing_rule_select)
        
        # 添加/删除按钮
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="添加规则", command=self.add_routing_rule).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="删除规则", command=self.delete_routing_rule).pack(side=tk.LEFT, padx=5)
        
        # 右侧编辑区域
        right_frame = ttk.Frame(self.routing_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 目标Agent
        agent_frame = ttk.Frame(right_frame)
        agent_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(agent_frame, text="目标智能体:").pack(side=tk.LEFT)
        self.target_agent_var = tk.StringVar()
        self.target_agent_combo = ttk.Combobox(agent_frame, textvariable=self.target_agent_var)
        self.target_agent_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 关键词
        ttk.Label(right_frame, text="关键词列表(逗号分隔):").pack(anchor=tk.W, pady=(10, 5))
        self.keywords_text = tk.Text(right_frame, height=5)
        self.keywords_text.pack(fill=tk.X)
        
        # 描述
        ttk.Label(right_frame, text="规则描述:").pack(anchor=tk.W, pady=(10, 5))
        self.rule_description_text = tk.Text(right_frame, height=5)
        self.rule_description_text.pack(fill=tk.X)
        
        # 保存按钮
        ttk.Button(right_frame, text="保存规则", command=self.save_routing_rule).pack(anchor=tk.E, pady=10)
    
    def load_configs(self):
        """加载配置和Agent列表"""
        # 加载Agent列表
        self.update_agent_list()
        
        # 更新活跃Agent和管理智能体下拉框
        self.update_active_agent_combo()
        self.update_manager_agent_combo()
        self.update_routing_manager_combo()
    
    def update_agent_list(self):
        """更新Agent列表"""
        self.agent_listbox.delete(0, tk.END)
        
        # 获取所有Agent文件
        agent_files = list(Path(AGENTS_DIR).glob("*.json"))
        
        for agent_file in agent_files:
            self.agent_listbox.insert(tk.END, agent_file.stem)

    def update_manager_agent_combo(self):
        """更新管理智能体下拉框"""
        # 获取所有Agent文件
        agent_files = list(Path(AGENTS_DIR).glob("*.json"))
        agent_names = [agent_file.stem for agent_file in agent_files]
        
        self.manager_agent_combo['values'] = agent_names
        
        # 加载系统配置
        try:
            with open(f"{CONFIG_DIR}/system_config.json", "r", encoding="utf-8") as f:
                system_config = json.load(f)
                if "manager_agent" in system_config:
                    self.manager_agent_var.set(system_config["manager_agent"])
                else:
                    # 默认使用manager作为管理智能体，如果存在
                    if "manager" in agent_names:
                        self.manager_agent_var.set("manager")
                    elif agent_names:
                        self.manager_agent_var.set(agent_names[0])
        except FileNotFoundError:
            if "manager" in agent_names:
                self.manager_agent_var.set("manager")
            elif agent_names:
                self.manager_agent_var.set(agent_names[0])

    def update_routing_manager_combo(self):
        """更新路由配置选项卡的管理智能体下拉框"""
        # 获取所有Agent文件
        agent_files = list(Path(AGENTS_DIR).glob("*.json"))
        agent_names = [agent_file.stem for agent_file in agent_files]
        
        self.routing_manager_combo['values'] = agent_names
        
        # 加载系统配置
        try:
            with open(f"{CONFIG_DIR}/system_config.json", "r", encoding="utf-8") as f:
                system_config = json.load(f)
                if "manager_agent" in system_config:
                    self.routing_manager_var.set(system_config["manager_agent"])
                else:
                    # 默认使用manager作为管理智能体，如果存在
                    if "manager" in agent_names:
                        self.routing_manager_var.set("manager")
                    elif agent_names:
                        self.routing_manager_var.set(agent_names[0])
        except FileNotFoundError:
            if "manager" in agent_names:
                self.routing_manager_var.set("manager")
            elif agent_names:
                self.routing_manager_var.set(agent_names[0])
    
    def update_active_agent_combo(self):
        """更新活跃Agent下拉框"""
        # 获取所有Agent文件
        agent_files = list(Path(AGENTS_DIR).glob("*.json"))
        agent_names = [agent_file.stem for agent_file in agent_files]
        
        self.active_agent_combo['values'] = agent_names
        self.target_agent_combo['values'] = agent_names
        
        # 加载系统配置
        try:
            with open(f"{CONFIG_DIR}/system_config.json", "r", encoding="utf-8") as f:
                system_config = json.load(f)
                if "active_agent" in system_config:
                    self.active_agent_var.set(system_config["active_agent"])
        except FileNotFoundError:
            if agent_names:
                self.active_agent_var.set(agent_names[0])

    def on_routing_manager_select(self, event):
        """处理路由管理智能体选择事件"""
        manager_name = self.routing_manager_var.get()
        if not manager_name:
            return
            
        # 更新路由规则列表
        self.update_routing_rules_list(manager_name)
    
    def update_routing_rules_list(self, manager_name):
        """更新路由规则列表"""
        self.routing_listbox.delete(0, tk.END)
        
        try:
            # 加载管理智能体配置
            with open(f"{AGENTS_DIR}/{manager_name}.json", "r", encoding="utf-8") as f:
                agent_config = json.load(f)
                
                # 检查是否有路由规则
                if "routing_rules" in agent_config:
                    for i, rule in enumerate(agent_config["routing_rules"]):
                        target_agent = rule.get("agent", "未知")
                        description = rule.get("description", "")
                        self.routing_listbox.insert(tk.END, f"{target_agent} - {description[:20]}")
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载管理智能体配置: {str(e)}")
    
    def on_routing_rule_select(self, event):
        """处理路由规则选择事件"""
        selection = self.routing_listbox.curselection()
        if not selection:
            return
            
        index = selection[0]
        manager_name = self.routing_manager_var.get()
        
        try:
            # 加载管理智能体配置
            with open(f"{AGENTS_DIR}/{manager_name}.json", "r", encoding="utf-8") as f:
                agent_config = json.load(f)
                
                # 检查是否有路由规则
                if "routing_rules" in agent_config and index < len(agent_config["routing_rules"]):
                    rule = agent_config["routing_rules"][index]
                    
                    # 更新UI
                    self.target_agent_var.set(rule.get("agent", ""))
                    
                    # 更新关键词文本框
                    self.keywords_text.delete("1.0", tk.END)
                    if "keywords" in rule:
                        self.keywords_text.insert("1.0", ", ".join(rule["keywords"]))
                    
                    # 更新描述文本框
                    self.rule_description_text.delete("1.0", tk.END)
                    if "description" in rule:
                        self.rule_description_text.insert("1.0", rule["description"])
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载路由规则: {str(e)}")
    
    def add_routing_rule(self):
        """添加新路由规则"""
        manager_name = self.routing_manager_var.get()
        if not manager_name:
            messagebox.showwarning("添加失败", "请先选择一个管理智能体")
            return
            
        try:
            # 加载管理智能体配置
            with open(f"{AGENTS_DIR}/{manager_name}.json", "r", encoding="utf-8") as f:
                agent_config = json.load(f)
                
                # 确保有路由规则键
                if "routing_rules" not in agent_config:
                    agent_config["routing_rules"] = []
                
                # 添加新规则
                new_rule = {
                    "agent": "",
                    "keywords": [],
                    "description": "新规则"
                }
                agent_config["routing_rules"].append(new_rule)
                
                # 保存配置
                with open(f"{AGENTS_DIR}/{manager_name}.json", "w", encoding="utf-8") as f:
                    json.dump(agent_config, f, ensure_ascii=False, indent=2)
                
                # 更新列表
                self.update_routing_rules_list(manager_name)
                
                # 选择新规则
                index = len(agent_config["routing_rules"]) - 1
                self.routing_listbox.selection_set(index)
                self.on_routing_rule_select(None)
        except Exception as e:
            messagebox.showerror("添加失败", f"无法添加路由规则: {str(e)}")
    
    def delete_routing_rule(self):
        """删除选中的路由规则"""
        selection = self.routing_listbox.curselection()
        if not selection:
            return
            
        index = selection[0]
        manager_name = self.routing_manager_var.get()
        
        # 确认删除
        if not messagebox.askyesno("确认删除", "确定要删除这条路由规则吗?"):
            return
        
        try:
            # 加载管理智能体配置
            with open(f"{AGENTS_DIR}/{manager_name}.json", "r", encoding="utf-8") as f:
                agent_config = json.load(f)
                
                # 检查是否有路由规则
                if "routing_rules" in agent_config and index < len(agent_config["routing_rules"]):
                    # 删除规则
                    agent_config["routing_rules"].pop(index)
                    
                    # 保存配置
                    with open(f"{AGENTS_DIR}/{manager_name}.json", "w", encoding="utf-8") as f:
                        json.dump(agent_config, f, ensure_ascii=False, indent=2)
                    
                    # 更新列表
                    self.update_routing_rules_list(manager_name)
        except Exception as e:
            messagebox.showerror("删除失败", f"无法删除路由规则: {str(e)}")
    
    def save_routing_rule(self):
        """保存当前路由规则"""
        selection = self.routing_listbox.curselection()
        if not selection:
            messagebox.showwarning("保存失败", "请先选择一个路由规则")
            return
            
        index = selection[0]
        manager_name = self.routing_manager_var.get()
        
        # 获取UI数据
        target_agent = self.target_agent_var.get()
        keywords_text = self.keywords_text.get("1.0", tk.END).strip()
        description = self.rule_description_text.get("1.0", tk.END).strip()
        
        # 解析关键词列表
        keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
        
        try:
            # 加载管理智能体配置
            with open(f"{AGENTS_DIR}/{manager_name}.json", "r", encoding="utf-8") as f:
                agent_config = json.load(f)
                
                # 检查是否有路由规则
                if "routing_rules" in agent_config and index < len(agent_config["routing_rules"]):
                    # 更新规则
                    agent_config["routing_rules"][index] = {
                        "agent": target_agent,
                        "keywords": keywords,
                        "description": description
                    }
                    
                    # 保存配置
                    with open(f"{AGENTS_DIR}/{manager_name}.json", "w", encoding="utf-8") as f:
                        json.dump(agent_config, f, ensure_ascii=False, indent=2)
                    
                    # 更新列表
                    self.update_routing_rules_list(manager_name)
                    
                    messagebox.showinfo("保存成功", "路由规则已保存")
        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存路由规则: {str(e)}")
    
    def on_agent_select(self, event):
        """处理Agent选择事件"""
        # 获取选中的Agent
        selection = self.agent_listbox.curselection()
        if not selection:
            return
        
        agent_name = self.agent_listbox.get(selection[0])
        
        # 加载Agent配置
        try:
            with open(f"{AGENTS_DIR}/{agent_name}.json", "r", encoding="utf-8") as f:
                agent_config = json.load(f)
                
                # 更新UI
                self.agent_name_var.set(agent_config.get("name", agent_name))
                self.prompt_text.delete("1.0", tk.END)
                self.prompt_text.insert("1.0", agent_config.get("prompt", ""))
                
                # 更新知识库列表
                self.kb_listbox.delete(0, tk.END)
                for kb in agent_config.get("knowledge_base", []):
                    self.kb_listbox.insert(tk.END, kb)
                
                # 更新工具集列表
                self.tools_listbox.delete(0, tk.END)
                for tool in agent_config.get("tools", []):
                    self.tools_listbox.insert(tk.END, tool)
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载Agent配置: {str(e)}")
    
    def add_agent(self):
        """添加新Agent"""
        new_name = simpledialog.askstring("新Agent", "请输入新Agent的名称:")
        if not new_name:
            return
        
        # 创建默认配置
        agent_config = {
            "name": new_name,
            "prompt": f"你是{new_name}助手，你能简短回答用户的问题。",
            "knowledge_base": [],
            "tools": []
        }
        
        # 保存配置
        try:
            with open(f"{AGENTS_DIR}/{new_name}.json", "w", encoding="utf-8") as f:
                json.dump(agent_config, f, ensure_ascii=False, indent=2)
            
            # 更新列表
            self.update_agent_list()
            self.update_active_agent_combo()
            
            # 选中新Agent
            idx = self.agent_listbox.get(0, tk.END).index(new_name)
            self.agent_listbox.selection_set(idx)
            self.on_agent_select(None)
        except Exception as e:
            messagebox.showerror("创建失败", f"无法创建Agent: {str(e)}")
    
    def delete_agent(self):
        """删除选中的Agent"""
        selection = self.agent_listbox.curselection()
        if not selection:
            return
        
        agent_name = self.agent_listbox.get(selection[0])
        
        # 确认删除
        if not messagebox.askyesno("确认删除", f"确定要删除Agent '{agent_name}'吗?"):
            return
        
        # 删除文件
        try:
            os.remove(f"{AGENTS_DIR}/{agent_name}.json")
            
            # 更新列表
            self.update_agent_list()
            self.update_active_agent_combo()
        except Exception as e:
            messagebox.showerror("删除失败", f"无法删除Agent: {str(e)}")
    
    def save_agent(self):
        """保存当前Agent配置"""
        selection = self.agent_listbox.curselection()
        if not selection:
            messagebox.showwarning("保存失败", "请先选择一个Agent")
            return
        
        agent_name = self.agent_listbox.get(selection[0])
        
        # 获取UI数据
        name = self.agent_name_var.get()
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        
        knowledge_base = list(self.kb_listbox.get(0, tk.END))
        tools = list(self.tools_listbox.get(0, tk.END))
        
        # 更新配置
        agent_config = {
            "name": name,
            "prompt": prompt,
            "knowledge_base": knowledge_base,
            "tools": tools
        }
        
        # 保存配置
        try:
            with open(f"{AGENTS_DIR}/{agent_name}.json", "w", encoding="utf-8") as f:
                json.dump(agent_config, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("保存成功", f"Agent '{agent_name}' 已保存")
        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存Agent配置: {str(e)}")
    
    def add_knowledge(self):
        """添加知识库"""
        # 创建对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("添加知识库")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 知识库名称
        name_frame = ttk.Frame(dialog)
        name_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(name_frame, text="知识库名称:").pack(side=tk.LEFT)
        name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=name_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 文件选择
        file_frame = ttk.Frame(dialog)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        file_var = tk.StringVar()
        ttk.Label(file_frame, text="PDF文件:").pack(side=tk.LEFT)
        ttk.Entry(file_frame, textvariable=file_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_pdf(file_var)).pack(side=tk.LEFT)
        
        # 按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="确定", command=lambda: self.add_kb_confirm(name_var.get(), file_var.get(), dialog)).pack(side=tk.RIGHT, padx=5)
        
        # 居中显示
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        dialog.wait_window()
    
    def browse_pdf(self, file_var):
        """浏览选择PDF文件"""
        file_path = filedialog.askopenfilename(
            title="选择PDF文件",
            filetypes=[("PDF文件", "*.pdf"), ("所有文件", "*.*")]
        )
        if file_path:
            file_var.set(file_path)
    
    def add_kb_confirm(self, name, file_path, dialog):
        """确认添加知识库"""
        if not name:
            messagebox.showwarning("警告", "请输入知识库名称", parent=dialog)
            return
        
        # 创建知识库记录
        kb_info = {"name": name}
        
        # 如果选择了文件，复制到知识库目录
        if file_path:
            # 确保知识库目录存在
            kb_dir = Path(f"{KNOWLEDGE_DIR}/{name}")
            kb_dir.mkdir(exist_ok=True)
            
            # 复制文件
            dest_file = kb_dir / Path(file_path).name
            try:
                shutil.copy2(file_path, dest_file)
                kb_info["files"] = [Path(file_path).name]
            except Exception as e:
                messagebox.showerror("错误", f"复制文件失败: {str(e)}", parent=dialog)
                return
        
        # 保存知识库信息
        try:
            with open(f"{KNOWLEDGE_DIR}/{name}.json", "w", encoding="utf-8") as f:
                json.dump(kb_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("错误", f"保存知识库信息失败: {str(e)}", parent=dialog)
            return
        
        # 添加到列表
        self.kb_listbox.insert(tk.END, name)
        
        # 关闭对话框
        dialog.destroy()
    
    def delete_knowledge(self):
        """删除选中的知识库"""
        selection = self.kb_listbox.curselection()
        if not selection:
            return
        
        kb_name = self.kb_listbox.get(selection[0])
        
        # 确认删除
        if not messagebox.askyesno("确认删除", f"确定要删除知识库 '{kb_name}'吗? 这将删除所有相关文件。"):
            return
        
        # 删除知识库目录和文件
        try:
            kb_dir = Path(f"{KNOWLEDGE_DIR}/{kb_name}")
            if kb_dir.exists():
                shutil.rmtree(kb_dir)
            
            # 删除知识库信息文件
            kb_info_file = Path(f"{KNOWLEDGE_DIR}/{kb_name}.json")
            if kb_info_file.exists():
                kb_info_file.unlink()
            
            # 从列表中删除
            self.kb_listbox.delete(selection[0])
        except Exception as e:
            messagebox.showerror("删除失败", f"无法删除知识库: {str(e)}")
    
    def add_tool(self):
        """添加工具"""
        tool_name = simpledialog.askstring("添加工具", "请输入工具名称:")
        if not tool_name:
            return
        
        self.tools_listbox.insert(tk.END, tool_name)
    
    def delete_tool(self):
        """删除选中的工具"""
        selection = self.tools_listbox.curselection()
        if not selection:
            return
        
        self.tools_listbox.delete(selection[0])
    
    def test_connection(self):
        """测试与设备的连接"""
        # 获取连接参数
        host = self.host_var.get()
        port = int(self.port_var.get())
        username = self.username_var.get()
        password = self.password_var.get()
        
        # 启动测试线程
        threading.Thread(target=self._test_connection_thread, 
                         args=(host, port, username, password),
                         daemon=True).start()
    
    def _test_connection_thread(self, host, port, username, password):
        """连接测试线程"""
        self.log("正在测试连接...")
        
        try:
            # 创建SSH客户端
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 连接
            client.connect(host, port, username, password, timeout=5)
            
            # 执行测试命令
            stdin, stdout, stderr = client.exec_command("echo 连接测试成功")
            result = stdout.read().decode('utf-8').strip()
            
            # 关闭连接
            client.close()
            
            self.log(f"测试结果: {result}")
        except Exception as e:
            self.log(f"连接失败: {str(e)}")
    
    def sync_to_device(self):
        """同步配置到设备"""
        # 获取管理智能体和活跃Agent
        manager_agent = self.manager_agent_var.get()
        active_agent = self.active_agent_var.get()
        
        if not manager_agent:
            messagebox.showwarning("同步失败", "请先选择管理智能体")
            return
            
        if not active_agent:
            messagebox.showwarning("同步失败", "请先选择默认活跃Agent")
            return
        
        # 更新系统配置
        system_config = {
            "active_agent": active_agent,
            "manager_agent": manager_agent,
            "record_duration": 5,
            "language": "zh"
        }
        
        with open(f"{CONFIG_DIR}/system_config.json", "w", encoding="utf-8") as f:
            json.dump(system_config, f, ensure_ascii=False, indent=2)
        
        # 获取连接参数
        host = self.host_var.get()
        port = int(self.port_var.get())
        username = self.username_var.get()
        password = self.password_var.get()
        
        # 启动同步线程
        threading.Thread(target=self._sync_thread, 
                         args=(host, port, username, password),
                         daemon=True).start()
    
    def _sync_thread(self, host, port, username, password):
        """同步线程"""
        self.log("开始同步...")
        
        try:
            # 创建SSH客户端
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 连接
            client.connect(host, port, username, password)
            
            # 创建SFTP客户端
            sftp = client.open_sftp()
            
            # 确保远程目录存在
            self.log(f"创建远程目录: {REMOTE_PATH}")
            for path in [REMOTE_PATH, 
                         f"{REMOTE_PATH}/config", 
                         f"{REMOTE_PATH}/agents",
                         f"{REMOTE_PATH}/knowledge",
                         f"{REMOTE_PATH}/tools"]:
                try:
                    sftp.stat(path)
                except FileNotFoundError:
                    stdin, stdout, stderr = client.exec_command(f"mkdir -p {path}")
                    stdout.read()
            
            # 同步系统配置
            self.log("同步系统配置...")
            sftp.put(f"{CONFIG_DIR}/system_config.json", f"{REMOTE_PATH}/config/system_config.json")
            
            # 同步管理智能体配置
            self.log("同步管理智能体配置...")
            manager_agent = self.manager_agent_var.get()
            sftp.put(f"{AGENTS_DIR}/{manager_agent}.json", f"{REMOTE_PATH}/agents/{manager_agent}.json")
            
            # 同步默认活跃Agent配置
            self.log("同步默认活跃Agent配置...")
            active_agent = self.active_agent_var.get()
            sftp.put(f"{AGENTS_DIR}/{active_agent}.json", f"{REMOTE_PATH}/agents/{active_agent}.json")
            
            # 同步所有路由智能体配置
            self.log("同步所有路由智能体配置...")
            manager_path = Path(f"{AGENTS_DIR}/{manager_agent}.json")
            if manager_path.exists():
                with open(manager_path, "r", encoding="utf-8") as f:
                    manager_config = json.load(f)
                    if "routing_rules" in manager_config:
                        for rule in manager_config["routing_rules"]:
                            target_agent = rule.get("agent")
                            if target_agent and target_agent != manager_agent and target_agent != active_agent:
                                target_path = Path(f"{AGENTS_DIR}/{target_agent}.json")
                                if target_path.exists():
                                    self.log(f"同步智能体: {target_agent}")
                                    sftp.put(str(target_path), f"{REMOTE_PATH}/agents/{target_agent}.json")
            
            # 同步知识库文件
            self.log("同步知识库文件...")
            agent_files = list(Path(AGENTS_DIR).glob("*.json"))
            for agent_file in agent_files:
                with open(agent_file, "r", encoding="utf-8") as f:
                    agent_config = json.load(f)
                    
                    # 同步Agent使用的知识库
                    for kb_name in agent_config.get("knowledge_base", []):
                        kb_info_path = Path(f"{KNOWLEDGE_DIR}/{kb_name}.json")
                        if kb_info_path.exists():
                            # 确保远程知识库目录存在
                            remote_kb_dir = f"{REMOTE_PATH}/knowledge/{kb_name}"
                            try:
                                sftp.stat(remote_kb_dir)
                            except FileNotFoundError:
                                stdin, stdout, stderr = client.exec_command(f"mkdir -p {remote_kb_dir}")
                                stdout.read()
                            
                            # 同步知识库信息文件
                            sftp.put(str(kb_info_path), f"{REMOTE_PATH}/knowledge/{kb_name}.json")
                            
                            # 同步知识库文件
                            with open(kb_info_path, "r", encoding="utf-8") as kf:
                                kb_info = json.load(kf)
                                for file_name in kb_info.get("files", []):
                                    local_file = Path(f"{KNOWLEDGE_DIR}/{kb_name}/{file_name}")
                                    if local_file.exists():
                                        self.log(f"同步知识库文件: {file_name}")
                                        sftp.put(str(local_file), f"{remote_kb_dir}/{file_name}")
            
            # 同步Python文件
            self.log("同步Python文件...")
            for py_file in ["edge_assistant.py", "edge_assistant_gui.py"]:
                if os.path.exists(py_file):
                    sftp.put(py_file, f"{REMOTE_PATH}/{py_file}")
            
            # 同步成功
            self.log("同步完成!")
            
            # 关闭连接
            sftp.close()
            client.close()
        except Exception as e:
            self.log(f"同步失败: {str(e)}")
    
    def log(self, message):
        """向日志窗口添加消息"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update()


# 导入必要的对话框
from tkinter import simpledialog

# 程序入口
if __name__ == "__main__":
    root = tk.Tk()
    app = PCConfigTool(root)
    root.mainloop()