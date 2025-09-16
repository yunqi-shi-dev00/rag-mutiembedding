# coding=utf-8
"""
本地MongoDB数据读取和发送器
从本地MongoDB读取数据，批量发送到云服务器
"""

import argparse
import json
import logging
import time
import requests
import pymongo
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSender:
    """数据发送器类"""
    
    def __init__(self, server_url: str, max_workers: int = 5, batch_size: int = 100, timeout: int = 60):
        self.server_url = server_url
        self.max_workers = max_workers
        self.batch_size = batch_size
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
        
        # 统计信息
        self.total_sent_count = 0
        self.total_failed_count = 0
        self.lock = threading.Lock()
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
    
    def send_batch(self, batch_data: List[Dict[str, Any]]) -> bool:
            """发送一批数据到云服务器"""
            try:
                payload = {
                    "batch_data": batch_data,
                    "batch_size": len(batch_data),
                    "timestamp": time.time()
                }
                
                response = self.session.post(
                    f"{self.server_url}/receive_data",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                if result.get("status") == "success":
                    with self.lock:
                        self.total_sent_count += len(batch_data)
                    logger.info(f"成功发送批次数据，大小: {len(batch_data)}")
                    return True
                else:
                    logger.error(f"服务器返回错误: {result.get('message', 'Unknown error')}")
                    with self.lock:
                        self.total_failed_count += len(batch_data)
                    return False
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"发送数据失败: {e}")
                with self.lock:
                    self.total_failed_count += len(batch_data)
                return False
            except Exception as e:
                logger.error(f"处理批次数据时发生错误: {e}")
                with self.lock:
                    self.total_failed_count += len(batch_data)
                return False

    def send_data_concurrent(self, data_batches: List[List[Dict[str, Any]]]) -> Dict[str, int]:
        """并发发送多批数据"""
        batch_sent = 0
        batch_failed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(self.send_batch, batch): batch 
                for batch in data_batches
            }
            
            # 等待完成
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    success = future.result()
                    if success:
                        batch_sent += len(batch)
                    else:
                        batch_failed += len(batch)
                except Exception as e:
                    logger.error(f"批次处理异常: {e}")
                    batch_failed += len(batch)
        
        return {
            "batch_sent": batch_sent,
            "batch_failed": batch_failed,
            "total_sent": self.total_sent_count,
            "total_failed": self.total_failed_count
        }
    
class MongoDataReader:
    """MongoDB数据读取器"""
    
    def __init__(self, url: str, db_name: str, table_name: str):
        self.url = url
        self.db_name = db_name
        self.table_name = table_name
        self.client = None
        self.collection = None
    
    def connect(self):
        """连接MongoDB"""
        try:
            self.client = pymongo.MongoClient(self.url)
            db = self.client[self.db_name]
            self.collection = db[self.table_name]
            logger.info(f'连接MongoDB成功，数据库: {self.db_name}, 集合: {self.table_name}')
            logger.info(f'可用集合: {db.list_collection_names()}')
            return True
        except Exception as e:
            logger.error(f"连接MongoDB失败: {e}")
            return False
    
    def close(self):
        """关闭MongoDB连接"""
        if self.client:
            self.client.close()
    
    def get_data_range(self, start_index: int, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """获取指定范围的数据"""
        try:
            find_data = self.collection.find(
                {'index': {'$gte': start_index, '$lt': start_index + batch_size}},
                {'index': 1, 'doc_id': 1, 'chunk_text': 1, '_id': 0}
            ).limit(batch_size)
            
            return list(find_data)
        except Exception as e:
            logger.error(f"读取数据失败: {e}")
            return []
    
    def get_total_count(self) -> int:
        """获取总数据量"""
        try:
            return self.collection.count_documents({})
        except Exception as e:
            logger.error(f"获取数据总量失败: {e}")
            return 0
    
    def get_max_index(self) -> int:
        """获取最大索引值"""
        try:
            result = self.collection.find().sort("index", -1).limit(1)
            max_doc = list(result)
            if max_doc:
                return max_doc[0].get('index', 0)
            return 0
        except Exception as e:
            logger.error(f"获取最大索引失败: {e}")
            return 0

def main():
    parser = argparse.ArgumentParser(description='本地MongoDB数据发送器')
    
    # MongoDB参数
    parser.add_argument('--mongo_url', type=str, 
                       default="mongodb://root:example@10.70.223.31:27017",
                       help='MongoDB连接URL')
    parser.add_argument('--db_name', type=str, default="rqa", help='数据库名称')
    parser.add_argument('--table_name', type=str, default="allchunk_newocr_20250723_wzh", help='表名称')
    
    # 云服务器参数
    parser.add_argument('--server_url', type=str, required=True,
                       help='云服务器URL，例如: http://your-server:8006')
    
    # 处理参数
    parser.add_argument('--start_index', type=int, default=0, help='开始索引')
    parser.add_argument('--end_index', type=int, default=None, help='结束索引')
    parser.add_argument('--batch_size', type=int, default=8000, help='每批处理数量')
    parser.add_argument('--send_batch_size', type=int, default=40, help='每次发送的批次数')
    parser.add_argument('--max_workers', type=int, default=20, help='并发工作线程数')
    parser.add_argument('--test', action='store_true', help='测试模式，只处理少量数据')
    
    args = parser.parse_args()
    
    # 初始化MongoDB读取器
    mongo_reader = MongoDataReader(args.mongo_url, args.db_name, args.table_name)
    if not mongo_reader.connect():
        logger.error("无法连接MongoDB，程序退出")
        return
    
    try:
        # 获取数据范围
        if args.end_index is None:
            max_index = mongo_reader.get_max_index()
            end_index = max_index + 1
            logger.info(f"自动检测到最大索引: {max_index}")
        else:
            end_index = args.end_index
        
        if args.test:
            end_index = min(args.start_index + 1000, end_index)
            logger.info("测试模式：只处理1000条数据")
        
        logger.info(f"处理范围: {args.start_index} - {end_index}")
        total_records = end_index - args.start_index
        logger.info(f"预计处理记录数: {total_records}")
        
        # 初始化数据发送器
        with DataSender(args.server_url, args.max_workers, args.batch_size) as sender:
            current_index = args.start_index
            total_processed_records = 0  # 重命名，明确含义
            start_time = time.time()
            
            while current_index < end_index:
                # 读取数据
                data_batch = mongo_reader.get_data_range(current_index, args.batch_size)
                
                if not data_batch:
                    logger.info(f"没有找到索引 {current_index} 的数据，跳过")
                    current_index += args.batch_size
                    continue
                
                # 记录本批次实际读取的数据量
                actual_batch_size = len(data_batch)
                total_processed_records += actual_batch_size
                
                # 将数据分组准备发送
                data_groups = []
                for i in range(0, len(data_batch), args.send_batch_size):
                    group = data_batch[i:i + args.send_batch_size]
                    data_groups.append(group)
                
                # 发送数据
                if data_groups:
                    result = sender.send_data_concurrent(data_groups)
                    
                    # 输出进度 - 使用实际处理的记录数计算
                    elapsed_time = time.time() - start_time
                    progress = (current_index - args.start_index) / total_records * 100
                    rate = total_processed_records / elapsed_time if elapsed_time > 0 else 0
                    
                    logger.info(f"进度: {progress:.1f}% | "
                              f"本批读取: {actual_batch_size} | "
                              f"累计读取: {total_processed_records} | "
                              f"累计发送成功: {result['total_sent']} | "
                              f"累计发送失败: {result['total_failed']} | "
                              f"处理速率: {rate:.1f} 记录/秒")
                
                current_index += args.batch_size
                time.sleep(0.1)
        
        # 最终统计
        total_time = time.time() - start_time
        logger.info(f"数据发送完成！")
        logger.info(f"总耗时: {total_time:.2f}秒")
        logger.info(f"总读取记录: {total_processed_records} 条")
        logger.info(f"发送成功: {sender.total_sent_count} 条")
        logger.info(f"发送失败: {sender.total_failed_count} 条")
        logger.info(f"平均处理速率: {total_processed_records/total_time:.1f} 记录/秒")
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序执行错误: {e}")
    finally:
        mongo_reader.close()

if __name__ == '__main__':
    main()
