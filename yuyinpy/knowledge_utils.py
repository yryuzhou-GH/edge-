import os
import re
import json
import numpy as np
import logging
from pathlib import Path
import hashlib
from typing import List, Dict, Any, Optional, Tuple

# 检查是否有faiss，如果没有则提供安装提示
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS is not available. To enable vector search, install faiss-cpu: pip install faiss-cpu")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KnowledgeUtils")

# 知识库根目录
KNOWLEDGE_DIR = "./knowledge"

# 向量缓存目录
VECTOR_CACHE_DIR = "./vector_cache"

class Document:
    """表示知识库中的一个文档或文档片段"""
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}
    
    def __str__(self):
        return f"Document(content={self.content[:50]}..., metadata={self.metadata})"

class TextSplitter:
    """文本分割器，将长文本分割成较小的块"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """将文本分割成块"""
        if not text:
            return []
        
        # 简单的按句子分割
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size or not current_chunk:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for the space
            else:
                # 完成当前块
                chunks.append(" ".join(current_chunk))
                
                # 开始新块，保留重叠部分
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                
                # 添加当前句子
                current_chunk.append(sentence)
                current_length += sentence_length + 1
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def create_documents(self, texts: List[str], metadata: Dict[str, Any] = None) -> List[Document]:
        """从文本列表创建文档对象"""
        documents = []
        for i, text in enumerate(texts):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata["chunk_id"] = i
            documents.append(Document(text, doc_metadata))
        return documents

class DocumentLoader:
    """文档加载器，支持加载不同格式的文档"""
    
    def __init__(self):
        self.text_splitter = TextSplitter()
    
    def load_text(self, file_path: str) -> List[Document]:
        """加载文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 分割文本
            chunks = self.text_splitter.split_text(text)
            
            # 创建元数据
            metadata = {
                "source": file_path,
                "file_type": "text",
                "file_name": os.path.basename(file_path)
            }
            
            # 创建文档对象
            return self.text_splitter.create_documents(chunks, metadata)
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return []
    
    def load_json(self, file_path: str) -> List[Document]:
        """加载JSON文件，特别是问答对格式"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # 处理不同格式的JSON
            if isinstance(data, list):
                for i, item in enumerate(data):
                    # 假设JSON中包含问答对
                    if isinstance(item, dict) and ('question' in item or 'query' in item) and ('answer' in item or 'response' in item):
                        question = item.get('question', item.get('query', ''))
                        answer = item.get('answer', item.get('response', ''))
                        
                        content = f"Question: {question}\nAnswer: {answer}"
                        metadata = {
                            "source": file_path,
                            "file_type": "json",
                            "item_id": i,
                            "question": question
                        }
                        documents.append(Document(content, metadata))
            
            return documents
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return []
    
    def load_file(self, file_path: str) -> List[Document]:
        """根据文件扩展名加载不同类型的文件"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            return self.load_text(file_path)
        elif ext == '.json':
            return self.load_json(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return []
    
    def load_directory(self, directory: str, recursive: bool = True) -> List[Document]:
        """加载目录中的所有支持的文档"""
        documents = []
        
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.txt', '.json')):
                    file_path = os.path.join(root, file)
                    documents.extend(self.load_file(file_path))
            
            if not recursive:
                break
        
        return documents

class SimpleVectorizer:
    """简单的词袋模型向量化器，不依赖额外库"""
    
    def __init__(self):
        # 简单的词汇表和IDF值
        self.vocab = {}  # 词到索引的映射
        self.idf = {}    # 词到IDF值的映射
        self.doc_count = 0
    
    def fit(self, documents: List[Document]):
        """根据文档构建词汇表和IDF值"""
        # 构建词汇表
        word_doc_counts = {}  # 词在多少文档中出现
        
        for doc in documents:
            self.doc_count += 1
            words = set(re.findall(r'\b\w+\b', doc.content.lower()))
            
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                
                word_doc_counts[word] = word_doc_counts.get(word, 0) + 1
        
        # 计算IDF值
        for word, count in word_doc_counts.items():
            self.idf[word] = np.log(self.doc_count / (1 + count))
    
    def transform(self, text: str) -> np.ndarray:
        """将文本转换为向量"""
        # 创建一个零向量
        vector = np.zeros(len(self.vocab))
        
        # 统计词频
        word_counts = {}
        for word in re.findall(r'\b\w+\b', text.lower()):
            if word in self.vocab:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # 填充向量
        for word, count in word_counts.items():
            if word in self.vocab:
                idx = self.vocab[word]
                tf = count / max(1, len(word_counts))
                vector[idx] = tf * self.idf.get(word, 1.0)
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector

class VectorStore:
    """向量存储，用于文档检索"""
    
    def __init__(self, knowledge_base_name: str):
        self.knowledge_base_name = knowledge_base_name
        self.documents = []
        self.vectors = None
        self.index = None
        self.vectorizer = SimpleVectorizer()
        
        # 创建缓存目录
        os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)
    
    def get_cache_path(self) -> str:
        """获取向量缓存文件路径"""
        # 使用知识库名称的哈希作为缓存文件名
        hash_name = hashlib.md5(self.knowledge_base_name.encode()).hexdigest()
        return os.path.join(VECTOR_CACHE_DIR, f"{hash_name}.npz")
    
    def add_documents(self, documents: List[Document]) -> bool:
        """添加文档到向量存储"""
        if not documents:
            return False
        
        self.documents.extend(documents)
        
        # 训练向量化器
        self.vectorizer.fit(self.documents)
        
        # 向量化所有文档
        vectors = []
        for doc in self.documents:
            vector = self.vectorizer.transform(doc.content)
            vectors.append(vector)
        
        self.vectors = np.array(vectors)
        
        # 创建FAISS索引
        if FAISS_AVAILABLE and self.vectors.shape[0] > 0:
            dim = self.vectors.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.vectors.astype('float32'))
        
        return True
    
    def save_cache(self) -> bool:
        """保存向量缓存"""
        try:
            cache_path = self.get_cache_path()
            
            # 保存文档内容和向量
            data_to_save = {
                "document_contents": [doc.content for doc in self.documents],
                "document_metadata": [doc.metadata for doc in self.documents],
                "vectors": self.vectors
            }
            
            np.savez_compressed(cache_path, **data_to_save)
            return True
        except Exception as e:
            logger.error(f"Error saving vector cache: {e}")
            return False
    
    def load_cache(self) -> bool:
        """加载向量缓存"""
        try:
            cache_path = self.get_cache_path()
            
            if os.path.exists(cache_path):
                data = np.load(cache_path, allow_pickle=True)
                
                # 恢复文档
                document_contents = data["document_contents"]
                document_metadata = data["document_metadata"]
                self.documents = [
                    Document(content, metadata)
                    for content, metadata in zip(document_contents, document_metadata)
                ]
                
                # 恢复向量
                self.vectors = data["vectors"]
                
                # 重建FAISS索引
                if FAISS_AVAILABLE and self.vectors.shape[0] > 0:
                    dim = self.vectors.shape[1]
                    self.index = faiss.IndexFlatL2(dim)
                    self.index.add(self.vectors.astype('float32'))
                
                return True
        except Exception as e:
            logger.error(f"Error loading vector cache: {e}")
        
        return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """基于相似度搜索文档"""
        if not self.documents or self.vectors is None:
            return []
        
        # 向量化查询
        query_vector = self.vectorizer.transform(query)
        query_vector = query_vector.reshape(1, -1)
        
        results = []
        
        if FAISS_AVAILABLE and self.index is not None:
            # 使用FAISS进行高效检索
            distances, indices = self.index.search(query_vector.astype('float32'), min(k, len(self.documents)))
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents) and idx >= 0:
                    results.append((self.documents[idx], float(distances[0][i])))
        else:
            # 回退到暴力计算
            similarities = []
            query_norm = np.linalg.norm(query_vector)
            
            for i, doc_vector in enumerate(self.vectors):
                doc_norm = np.linalg.norm(doc_vector)
                
                if query_norm > 0 and doc_norm > 0:
                    # 使用余弦相似度
                    similarity = np.dot(query_vector, doc_vector) / (query_norm * doc_norm)
                else:
                    similarity = 0
                
                similarities.append((i, similarity))
            
            # 排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 取前k个
            for i, sim in similarities[:k]:
                results.append((self.documents[i], float(sim)))
        
        return results

class KnowledgeBase:
    """知识库管理类，处理知识库的加载、缓存和检索"""
    
    def __init__(self, base_dir: str = KNOWLEDGE_DIR):
        self.base_dir = base_dir
        self.vector_stores = {}
        
        # 确保目录存在
        os.makedirs(base_dir, exist_ok=True)
    
    def get_knowledge_base_path(self, knowledge_base_name: str) -> str:
        """获取知识库目录路径"""
        return os.path.join(self.base_dir, knowledge_base_name)
    
    def list_knowledge_bases(self) -> List[str]:
        """列出所有可用的知识库"""
        try:
            return [d for d in os.listdir(self.base_dir) 
                   if os.path.isdir(os.path.join(self.base_dir, d))]
        except Exception as e:
            logger.error(f"Error listing knowledge bases: {e}")
            return []
    
    def load_knowledge_base(self, knowledge_base_name: str) -> Optional[VectorStore]:
        """加载知识库并创建向量存储"""
        # 检查是否已加载
        if knowledge_base_name in self.vector_stores:
            return self.vector_stores[knowledge_base_name]
        
        # 创建向量存储
        vector_store = VectorStore(knowledge_base_name)
        
        # 尝试从缓存加载
        if vector_store.load_cache():
            self.vector_stores[knowledge_base_name] = vector_store
            logger.info(f"Loaded knowledge base '{knowledge_base_name}' from cache")
            return vector_store
        
        # 无缓存时从文件加载
        kb_path = self.get_knowledge_base_path(knowledge_base_name)
        if not os.path.exists(kb_path):
            logger.warning(f"Knowledge base directory not found: {kb_path}")
            return None
        
        # 加载文档
        loader = DocumentLoader()
        documents = loader.load_directory(kb_path)
        
        if not documents:
            logger.warning(f"No documents found in knowledge base: {knowledge_base_name}")
            return None
        
        # 添加文档到向量存储
        vector_store.add_documents(documents)
        
        # 保存缓存
        vector_store.save_cache()
        
        # 存储向量库
        self.vector_stores[knowledge_base_name] = vector_store
        
        logger.info(f"Loaded knowledge base '{knowledge_base_name}' with {len(documents)} documents")
        return vector_store
    
    def search_knowledge_base(self, knowledge_base_name: str, query: str, k: int = 3) -> List[Document]:
        """在知识库中搜索相关文档"""
        # 加载知识库
        vector_store = self.load_knowledge_base(knowledge_base_name)
        if not vector_store:
            return []
        
        # 执行搜索
        results = vector_store.similarity_search(query, k)
        
        # 返回文档
        return [doc for doc, _ in results]

def format_documents_for_prompt(documents: List[Document]) -> str:
    """将检索到的文档格式化为提示词"""
    if not documents:
        return ""
    
    formatted = "\n\n### Relevant Information:\n"
    
    for i, doc in enumerate(documents):
        source = doc.metadata.get('source', 'Unknown')
        formatted += f"\n[Document {i+1}] {os.path.basename(source)}\n"
        formatted += doc.content
        formatted += "\n---\n"
    
    return formatted

def main():
    """测试函数"""
    # 创建测试知识库目录
    test_kb_dir = os.path.join(KNOWLEDGE_DIR, "test_kb")
    os.makedirs(test_kb_dir, exist_ok=True)
    
    # 创建测试文档
    test_file = os.path.join(test_kb_dir, "sample.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("This is a test document. It contains information about knowledge bases. "
                "Knowledge bases are used to store and retrieve information. "
                "Retrieval-augmented generation (RAG) combines retrieval with generation.")
    
    # 创建知识库
    kb = KnowledgeBase()
    
    # 列出知识库
    print("Available knowledge bases:", kb.list_knowledge_bases())
    
    # 加载测试知识库
    print("Loading test knowledge base...")
    kb.load_knowledge_base("test_kb")
    
    # 搜索测试
    query = "How are knowledge bases used?"
    print(f"\nSearching for: '{query}'")
    docs = kb.search_knowledge_base("test_kb", query)
    
    for doc in docs:
        print("\nFound document:")
        print(f"Content: {doc.content}")
        print(f"Metadata: {doc.metadata}")
    
    # 格式化为提示词
    prompt = format_documents_for_prompt(docs)
    print("\nFormatted prompt:")
    print(prompt)

if __name__ == "__main__":
    main() 