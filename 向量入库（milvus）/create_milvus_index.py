import logging
from pymilvus import Collection, connections, utility
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusIndexManager:
    """Milvus索引管理类"""
    
    def __init__(self, host: str = "localhost", port: str = "19530", collection_name: str = None):
        """
        初始化Milvus连接和集合
        
        Args:
            host: Milvus服务器地址
            port: Milvus服务器端口
            collection_name: 集合名称
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.connect()
        
    def connect(self):
        """连接到Milvus服务器"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"成功连接到Milvus服务器 {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def load_collection(self, collection_name: str = None):
        """加载集合"""
        if collection_name:
            self.collection_name = collection_name
            
        if not self.collection_name:
            logger.error("集合名称未指定")
            return False
            
        try:
            self.collection = Collection(self.collection_name)
            logger.info(f"成功加载集合: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False
    
    def create_index(self, field_name: str = "embeddings", index_type: str = "HNSW", 
                    metric_type: str = "IP", index_params: Optional[Dict[str, Any]] = None):
        """
        创建索引
        
        Args:
            field_name: 要创建索引的字段名
            index_type: 索引类型 (HNSW, IVF_FLAT, IVF_SQ8, IVF_PQ, ANNOY等)
            metric_type: 距离度量类型 (IP, L2, COSINE, HAMMING, JACCARD等)
            index_params: 自定义索引参数
        """
        if not self.collection:
            logger.error("集合未初始化，请先加载集合")
            return False
        
        try:
            # 检查字段是否存在
            if not self._check_field_exists(field_name):
                logger.error(f"字段 '{field_name}' 不存在")
                return False
            
            # 设置默认索引参数
            if index_params is None:
                index_params = self._get_default_index_params(index_type)
            
            index_config = {
                "index_type": index_type,
                "metric_type": metric_type,
                "params": index_params
            }
            
            # 创建索引
            self.collection.create_index(field_name, index_config)
            logger.info(f"成功为字段 '{field_name}' 创建 {index_type} 索引")
            return True
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False
    
    def _check_field_exists(self, field_name: str) -> bool:
        """检查字段是否存在"""
        try:
            schema = self.collection.schema
            for field in schema.fields:
                if field.name == field_name:
                    return True
            return False
        except Exception as e:
            logger.error(f"检查字段失败: {e}")
            return False
    
    def _get_default_index_params(self, index_type: str) -> Dict[str, Any]:
        """获取默认索引参数"""
        default_params = {
            "HNSW": {"M": 48, "efConstruction": 128},
            "IVF_FLAT": {"nlist": 1024},
            "IVF_SQ8": {"nlist": 1024},
            "IVF_PQ": {"nlist": 1024, "m": 16, "nbits": 8},
            "ANNOY": {"n_trees": 8},
            "RHNSW_FLAT": {"M": 16, "efConstruction": 200},
            "RHNSW_SQ": {"M": 16, "efConstruction": 200},
            "RHNSW_PQ": {"M": 16, "efConstruction": 200, "PQM": 64},
            "DISKANN": {"search_list_size": 100},
        }
        return default_params.get(index_type, {})
    
    def drop_index(self, field_name: str = "embeddings"):
        """删除索引"""
        if not self.collection:
            logger.error("集合未初始化")
            return False
        
        try:
            self.collection.drop_index(field_name)
            logger.info(f"成功删除字段 '{field_name}' 的索引")
            return True
        except Exception as e:
            logger.error(f"删除索引失败: {e}")
            return False
    
    def list_indexes(self):
        """列出所有索引"""
        if not self.collection:
            logger.error("集合未初始化")
            return []
        
        try:
            indexes = self.collection.indexes
            logger.info(f"集合 '{self.collection_name}' 的索引列表:")
            for idx in indexes:
                logger.info(f"  - 字段: {idx.field_name}, 类型: {idx.params.get('index_type', 'unknown')}")
            return indexes
        except Exception as e:
            logger.error(f"获取索引列表失败: {e}")
            return []
    
    def get_index_info(self, field_name: str = "embeddings"):
        """获取指定字段的索引信息"""
        if not self.collection:
            logger.error("集合未初始化")
            return None
        
        try:
            index = self.collection.index(field_name)
            if index:
                logger.info(f"字段 '{field_name}' 的索引信息:")
                logger.info(f"  - 索引类型: {index.params.get('index_type', 'unknown')}")
                logger.info(f"  - 距离度量: {index.params.get('metric_type', 'unknown')}")
                logger.info(f"  - 参数: {index.params.get('params', {})}")
                return index
            else:
                logger.info(f"字段 '{field_name}' 没有索引")
                return None
        except Exception as e:
            logger.error(f"获取索引信息失败: {e}")
            return None
    
    def load_collection_to_memory(self):
        """将集合加载到内存中"""
        if not self.collection:
            logger.error("集合未初始化")
            return False
        
        try:
            self.collection.load()
            logger.info(f"成功将集合 '{self.collection_name}' 加载到内存")
            return True
        except Exception as e:
            logger.error(f"加载集合到内存失败: {e}")
            return False
    
    def release_collection(self):
        """释放集合内存"""
        if not self.collection:
            logger.error("集合未初始化")
            return False
        
        try:
            self.collection.release()
            logger.info(f"成功释放集合 '{self.collection_name}' 的内存")
            return True
        except Exception as e:
            logger.error(f"释放集合内存失败: {e}")
            return False

# 使用示例
def main():
    """使用示例"""
    # 初始化索引管理器
    index_manager = MilvusIndexManager(
        host="localhost",
        port="19530",
        collection_name="qwen3_06b_chunk_change_1_v20250723"
    )
    
    # 加载集合
    if index_manager.load_collection():
        # 创建HNSW索引
        # index_manager.create_index(
        #     field_name="embeddings",
        #     index_type="HNSW",
        #     metric_type="IP"
        # )
        
        # 创建自定义参数的索引
        custom_params = {"nlist": 1024}
        index_manager.create_index(
            field_name="embeddings",
            index_type="IVF_FLAT",
            metric_type="IP",
            index_params=custom_params
        )
        
        # 查看索引信息
        index_manager.get_index_info("embeddings")
        
        # 列出所有索引
        index_manager.list_indexes()
        
        # 加载集合到内存
        index_manager.load_collection_to_memory()

if __name__ == "__main__":
    main()