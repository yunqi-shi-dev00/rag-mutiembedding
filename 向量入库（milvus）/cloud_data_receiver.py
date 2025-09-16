# coding=utf-8
"""
云服务器数据接收器和Milvus写入器
接收本地发送的数据，通过编码服务生成嵌入向量，写入Milvus
"""

import argparse
import json
import logging
import time
import requests
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
from flask import Flask, request, jsonify
import pymilvus
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class EmbeddingClient:
    """嵌入向量生成客户端"""
    
    def __init__(self, api_url: str = "http://localhost:8010/encode", 
                 max_workers: int = 10, 
                 timeout: int = 60):
        self.api_url = api_url
        self.max_workers = max_workers
        self.timeout = timeout
        self.session = requests.Session()
        
        # 配置连接池
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers,
            pool_maxsize=max_workers * 2,
            max_retries=requests.adapters.Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504]
            )
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # 设置请求头
        self.session.headers.update({
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        })
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取嵌入向量"""
        if not texts:
            return []
            
        payload = {"queries": texts}
        
        try:
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if 'error' in result:
                raise ValueError(f"API返回错误: {result['error']}")
                
            if 'embeddings' not in result:
                raise ValueError("API响应中缺少embeddings字段")
                
            return result['embeddings']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"请求嵌入向量失败: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            raise

class MilvusManager:
    """Milvus数据库管理器"""
    
    def __init__(self, host: str = "localhost", port: str = "19530", 
                 collection_name: str = "embed_db", dim: int = 4096):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        self.connected = False
    
    def connect(self) -> bool:
        """连接Milvus"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            self.connected = True
            logger.info(f"成功连接Milvus: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            return False
    
    def create_collection(self, version: str = "2023071310") -> bool:
        """创建或获取集合"""
        try:
            full_collection_name = f"{self.collection_name}_v{version}"
            
            if utility.has_collection(full_collection_name):
                logger.info(f"集合 {full_collection_name} 已存在，直接使用")
                self.collection = Collection(full_collection_name)
            else:
                # 创建新集合
                fields = [
                    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
                    FieldSchema(name="doc_id", dtype=DataType.INT64, auto_id=False),
                    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
                ]
                
                schema = CollectionSchema(fields, f"Collection for {full_collection_name}")
                self.collection = Collection(full_collection_name, schema)
                logger.info(f"成功创建集合: {full_collection_name}")
            
            return True
        except Exception as e:
            logger.error(f"创建/获取集合失败: {e}")
            return False
    
    def insert_data(self, indices: List[int], doc_ids: List[int], embeddings: List[List[float]]) -> bool:

        """插入数据到Milvus"""
        if not self.collection:
            logger.error("集合未初始化")
            return False
        
        try:
            # 重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.collection.insert([indices, doc_ids, embeddings])
                    logger.info(f"成功插入 {len(indices)} 条数据到Milvus")
                    return True
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"插入失败，第 {attempt + 1} 次重试: {e}")
                        time.sleep(2)
                    else:
                        raise e
        except Exception as e:
            logger.error(f"插入数据失败: {e}")
            return False
    
    def create_index(self, index_type: str = "HNSW", metric_type: str = "IP"):
        """创建索引"""
        if not self.collection:
            logger.error("集合未初始化")
            return False
        
        try:
            index_params = {
                "index_type": index_type,
                "metric_type": metric_type,
                "params": {"M": 48, "efConstruction": 128},
            }
            self.collection.create_index("embeddings", index_params)
            logger.info("成功创建索引")
            return True
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self.collection:
            return {}
        
        try:
            return {
                "num_entities": self.collection.num_entities,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, embedding_client: EmbeddingClient, milvus_manager: MilvusManager):
        self.embedding_client = embedding_client
        self.milvus_manager = milvus_manager
        self.processed_count = 0
        self.failed_count = 0
        self.lock = threading.Lock()
    
    def process_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理一批数据"""
        try:
            # 提取数据
            texts = [item['chunk_text'] for item in batch_data]
            indices = [item['index'] for item in batch_data]
            doc_ids = [item['doc_id'] for item in batch_data]
            
            # 生成嵌入向量
            embeddings = self.embedding_client.get_embeddings_batch(texts)
            
            if len(embeddings) != len(indices):
                raise ValueError(f"嵌入向量数量不匹配: {len(embeddings)} vs {len(indices)}")

            # 插入Milvus
            success = self.milvus_manager.insert_data(indices, doc_ids, embeddings)
            
            with self.lock:
                if success:
                    self.processed_count += len(batch_data)
                else:
                    self.failed_count += len(batch_data)
            
            return {
                "status": "success" if success else "error",
                "processed": len(batch_data),
                "message": "处理成功" if success else "插入Milvus失败"
            }
            
        except Exception as e:
            with self.lock:
                self.failed_count += len(batch_data)
            logger.error(f"处理批次数据失败: {e}")
            return {
                "status": "error",
                "processed": 0,
                "message": str(e)
            }
    
    def get_stats(self) -> Dict[str, int]:
        """获取处理统计"""
        with self.lock:
            return {
                "processed_count": self.processed_count,
                "failed_count": self.failed_count,
                "total_count": self.processed_count + self.failed_count
            }

# 全局变量
data_processor = None
start_time = time.time()

@app.route('/receive_data', methods=['POST'])
def receive_data():
    """接收数据的API端点"""
    global data_processor
    
    if not data_processor:
        return jsonify({
            "status": "error",
            "message": "数据处理器未初始化"
        }), 500
    
    try:
        data = request.get_json()
        if not data or 'batch_data' not in data:
            return jsonify({
                "status": "error",
                "message": "无效的请求数据"
            }), 400
        
        batch_data = data['batch_data']
        logger.info(f"接收到 {len(batch_data)} 条数据")
        
        # 处理数据
        result = data_processor.process_batch(batch_data)
        
        return jsonify({
            "status": result["status"],
            "message": result["message"],
            "processed": result["processed"],
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"处理请求失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """获取服务状态"""
    global data_processor, start_time
    
    if not data_processor:
        return jsonify({
            "status": "error",
            "message": "数据处理器未初始化"
        })
    
    stats = data_processor.get_stats()
    milvus_stats = data_processor.milvus_manager.get_stats()
    
    elapsed_time = time.time() - start_time
    rate = stats["processed_count"] / elapsed_time if elapsed_time > 0 else 0
    
    return jsonify({
        "status": "running",
        "uptime": elapsed_time,
        "processing_rate": f"{rate:.2f} records/sec",
        "stats": stats,
        "milvus_stats": milvus_stats,
        "timestamp": time.time()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time()
    })

def main():
    global data_processor, start_time
    
    parser = argparse.ArgumentParser(description='云服务器数据接收器')
    
    # 服务器参数
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器绑定地址')
    parser.add_argument('--port', type=int, default=8006, help='服务器端口')
    
    # 编码服务参数
    parser.add_argument('--encode_url', type=str, default='http://localhost:8005/encode',
                       help='编码服务URL')
    
    # Milvus参数
    parser.add_argument('--milvus_host', type=str, default='localhost', help='Milvus主机')
    parser.add_argument('--milvus_port', type=str, default='19530', help='Milvus端口')
    parser.add_argument('--collection_name', type=str, default='embed_db', help='集合名称')
    parser.add_argument('--dim', type=int, required=True, help='向量维度')
    parser.add_argument('--version', type=str, default='2023071310', help='版本号')
    
    # 性能参数
    parser.add_argument('--max_workers', type=int, default=40, help='最大工作线程数')
    
    args = parser.parse_args()
    
    # 初始化组件
    logger.info("初始化服务组件...")
    
    # 初始化嵌入向量客户端
    embedding_client = EmbeddingClient(args.encode_url, args.max_workers)
    
    # 初始化Milvus管理器
    milvus_manager = MilvusManager(
        args.milvus_host, 
        args.milvus_port, 
        args.collection_name, 
        args.dim
    )
    
    # 连接Milvus
    if not milvus_manager.connect():
        logger.error("无法连接Milvus，程序退出")
        return
    
    # 创建集合
    if not milvus_manager.create_collection(args.version):
        logger.error("无法创建/获取Milvus集合，程序退出")
        return
    
    # 初始化数据处理器
    data_processor = DataProcessor(embedding_client, milvus_manager)
    start_time = time.time()
    
    logger.info(f"服务初始化完成，准备在 {args.host}:{args.port} 启动服务器")
    logger.info(f"编码服务地址: {args.encode_url}")
    logger.info(f"Milvus地址: {args.milvus_host}:{args.milvus_port}")
    logger.info(f"集合名称: {args.collection_name}_v{args.version}")
    
    # 启动Flask服务器
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("服务器被用户中断")
    except Exception as e:
        logger.error(f"服务器错误: {e}")

if __name__ == '__main__':
    main()
