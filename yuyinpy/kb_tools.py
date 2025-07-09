#!/usr/bin/env python3
"""
知识库管理工具

这个脚本提供了一些简单的命令行功能来管理Edge智音的知识库。
可以用于添加、删除和查看知识库内容。
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path

# 导入知识库工具
try:
    from knowledge_utils import KnowledgeBase, DocumentLoader
    KNOWLEDGE_UTILS_AVAILABLE = True
except ImportError:
    KNOWLEDGE_UTILS_AVAILABLE = False
    print("警告: knowledge_utils 模块不可用，某些功能可能无法使用")

# 知识库根目录
KNOWLEDGE_DIR = "./knowledge"

def ensure_kb_dir():
    """确保知识库目录存在"""
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

def list_knowledge_bases():
    """列出所有可用的知识库"""
    ensure_kb_dir()
    
    kb_dirs = [d for d in os.listdir(KNOWLEDGE_DIR) 
              if os.path.isdir(os.path.join(KNOWLEDGE_DIR, d))]
    
    if not kb_dirs:
        print("没有找到任何知识库。可以使用 'create' 命令创建一个新的知识库。")
        return
    
    print("可用的知识库:")
    for kb in kb_dirs:
        kb_path = os.path.join(KNOWLEDGE_DIR, kb)
        file_count = len([f for f in os.listdir(kb_path) if os.path.isfile(os.path.join(kb_path, f))])
        print(f"- {kb} ({file_count} 个文件)")

def create_knowledge_base(name):
    """创建一个新的知识库"""
    ensure_kb_dir()
    
    kb_path = os.path.join(KNOWLEDGE_DIR, name)
    if os.path.exists(kb_path):
        print(f"知识库 '{name}' 已存在")
        return
    
    try:
        os.makedirs(kb_path)
        print(f"已创建知识库 '{name}'")
    except Exception as e:
        print(f"创建知识库时出错: {e}")

def delete_knowledge_base(name):
    """删除一个知识库"""
    ensure_kb_dir()
    
    kb_path = os.path.join(KNOWLEDGE_DIR, name)
    if not os.path.exists(kb_path):
        print(f"知识库 '{name}' 不存在")
        return
    
    try:
        # 删除前确认
        print(f"确定要删除知识库 '{name}' 及其中所有内容吗? [y/N] ", end='')
        confirm = input().lower()
        if confirm != 'y' and confirm != 'yes':
            print("操作已取消")
            return
        
        # 执行删除
        shutil.rmtree(kb_path)
        print(f"已删除知识库 '{name}'")
    except Exception as e:
        print(f"删除知识库时出错: {e}")

def add_file_to_kb(kb_name, file_path):
    """将文件添加到知识库"""
    ensure_kb_dir()
    
    kb_path = os.path.join(KNOWLEDGE_DIR, kb_name)
    if not os.path.exists(kb_path):
        print(f"知识库 '{kb_name}' 不存在")
        return
    
    if not os.path.exists(file_path):
        print(f"文件 '{file_path}' 不存在")
        return
    
    try:
        # 获取文件名
        filename = os.path.basename(file_path)
        dest_path = os.path.join(kb_path, filename)
        
        # 如果是目录，使用shutil.copytree
        if os.path.isdir(file_path):
            if os.path.exists(dest_path):
                print(f"目标目录 '{dest_path}' 已存在，请先删除")
                return
            shutil.copytree(file_path, dest_path)
            print(f"已将目录 '{file_path}' 复制到知识库 '{kb_name}'")
        # 如果是文件，使用shutil.copy2
        else:
            shutil.copy2(file_path, dest_path)
            print(f"已将文件 '{filename}' 添加到知识库 '{kb_name}'")
    except Exception as e:
        print(f"添加文件时出错: {e}")

def list_kb_files(kb_name):
    """列出知识库中的文件"""
    ensure_kb_dir()
    
    kb_path = os.path.join(KNOWLEDGE_DIR, kb_name)
    if not os.path.exists(kb_path):
        print(f"知识库 '{kb_name}' 不存在")
        return
    
    files = [f for f in os.listdir(kb_path) if os.path.isfile(os.path.join(kb_path, f))]
    
    if not files:
        print(f"知识库 '{kb_name}' 中没有文件")
        return
    
    print(f"知识库 '{kb_name}' 中的文件:")
    for f in files:
        file_path = os.path.join(kb_path, f)
        file_size = os.path.getsize(file_path)
        print(f"- {f} ({format_size(file_size)})")

def format_size(size_bytes):
    """将字节数格式化为人类可读的形式"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def test_knowledge_base(kb_name, query):
    """测试在知识库中检索内容"""
    if not KNOWLEDGE_UTILS_AVAILABLE:
        print("无法测试知识库，knowledge_utils 模块不可用")
        return
    
    try:
        kb = KnowledgeBase()
        print(f"正在从知识库 '{kb_name}' 中搜索与查询相关的内容: '{query}'")
        docs = kb.search_knowledge_base(kb_name, query)
        
        if not docs:
            print("未找到相关内容")
            return
        
        print(f"找到 {len(docs)} 条相关内容:")
        for i, doc in enumerate(docs):
            print(f"\n--- 结果 {i+1} ---")
            source = doc.metadata.get('source', '未知来源')
            print(f"来源: {source}")
            content_preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            print(f"内容: {content_preview}")
    except Exception as e:
        print(f"测试知识库时出错: {e}")
        import traceback
        traceback.print_exc()

def create_qa_pair(kb_name, question, answer):
    """创建问答对并添加到知识库"""
    ensure_kb_dir()
    
    kb_path = os.path.join(KNOWLEDGE_DIR, kb_name)
    if not os.path.exists(kb_path):
        print(f"知识库 '{kb_name}' 不存在")
        return
    
    # 创建或打开JSON文件
    qa_file = os.path.join(kb_path, "qa_pairs.json")
    
    try:
        # 读取现有内容（如果有）
        qa_pairs = []
        if os.path.exists(qa_file):
            with open(qa_file, 'r', encoding='utf-8') as f:
                try:
                    qa_pairs = json.load(f)
                except json.JSONDecodeError:
                    print("JSON文件格式不正确，将创建新文件")
                    qa_pairs = []
        
        # 添加新问答对
        qa_pairs.append({
            "question": question,
            "answer": answer
        })
        
        # 保存到文件
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        print(f"已将问答对添加到知识库 '{kb_name}'")
    except Exception as e:
        print(f"添加问答对时出错: {e}")

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="Edge智音知识库管理工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # list命令
    list_parser = subparsers.add_parser("list", help="列出所有知识库")
    
    # create命令
    create_parser = subparsers.add_parser("create", help="创建一个新的知识库")
    create_parser.add_argument("name", help="知识库名称")
    
    # delete命令
    delete_parser = subparsers.add_parser("delete", help="删除一个知识库")
    delete_parser.add_argument("name", help="知识库名称")
    
    # add命令
    add_parser = subparsers.add_parser("add", help="将文件添加到知识库")
    add_parser.add_argument("kb_name", help="知识库名称")
    add_parser.add_argument("file_path", help="文件路径")
    
    # files命令
    files_parser = subparsers.add_parser("files", help="列出知识库中的文件")
    files_parser.add_argument("kb_name", help="知识库名称")
    
    # test命令
    test_parser = subparsers.add_parser("test", help="测试在知识库中检索内容")
    test_parser.add_argument("kb_name", help="知识库名称")
    test_parser.add_argument("query", help="搜索查询")
    
    # qa命令
    qa_parser = subparsers.add_parser("qa", help="创建问答对并添加到知识库")
    qa_parser.add_argument("kb_name", help="知识库名称")
    qa_parser.add_argument("question", help="问题")
    qa_parser.add_argument("answer", help="回答")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_knowledge_bases()
    elif args.command == "create":
        create_knowledge_base(args.name)
    elif args.command == "delete":
        delete_knowledge_base(args.name)
    elif args.command == "add":
        add_file_to_kb(args.kb_name, args.file_path)
    elif args.command == "files":
        list_kb_files(args.kb_name)
    elif args.command == "test":
        test_knowledge_base(args.kb_name, args.query)
    elif args.command == "qa":
        create_qa_pair(args.kb_name, args.question, args.answer)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 