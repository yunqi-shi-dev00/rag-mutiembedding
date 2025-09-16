# coding=utf-8

import argparse
import datetime
import os
import pymilvus
import pymongo
import random
import sys 
import time
import traceback
import json

from flask import Flask, request, Response, jsonify
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

"""
启动向量搜flask服务
"""
app = Flask(import_name=__name__)
app.config['JSON_AS_ASCII'] = False

# 加载milvus数据库的参数
host = '10.120.105.235'
port = '19530'
index_type = 'HNSW'
metric_type = 'IP'
nprobe = 128
embed_db_name = 'dim512_v2023061914'
embed_db: pymilvus.Collection = None
search_params = {
    "metric_type": metric_type,
    "params": {"ef": 128},
}

def load_milvus(embed_db_name):
    start_time = time.time()

    global embed_db
    connections.connect("default", host=host, port=port)

    embed_db = Collection(embed_db_name)
    embed_db.load()
    print(f'load from cache spent time = {time.time() - start_time}')    
    print(f'milvus num of embed vec = {embed_db.num_entities}') 
    return

# 0表示正常 -1表示接口出错 可以使用其他code适用于不同场景
def json_result(code: int, msg: str, data):
    return jsonify({'code': code, 'msg': msg, 'data': data})

@app.route('/api-vec-search/test', methods=['GET'])
def hello_world():
    return json_result(0, '', 'Service available')

@app.route('/api-vec-search/search', methods=['POST'])
def get_data():
    form = request.form
    query_vec_str = form.get('query_vec')
    query_vec=json.loads(query_vec_str)

    start_time = time.time()
    code = 0
    msg = ''
    data = {}
    topk = 0
    vec_num = 0
    # try:
    if 1:
        topk = form.get('topk', 3000, int)
        search_params['params']['ef'] = topk
        # print(111111)
        result = embed_db.search(data=query_vec, 
                              anns_field="embeddings", 
                              param=search_params, 
                              output_fields=['doc_id'],
                              limit=topk)
        # print(222222)
        indexes = result[0].ids
        distances = result[0].distances
        docids = [t.entity.get('doc_id') for t in result[0]]
        vec_num = len(indexes)

        res_arr = [f'{indexes[i]}:{distances[i]}:{docids[i]}' for i in range(vec_num)]
        data['arr'] = res_arr
        data['vec_num'] = vec_num
    # except Exception as e:
    else:
        code = -1
        msg = traceback.format_exc()
        data['msg'] = msg
        data['vec_num'] = 0
        data['arr'] = []

    now = datetime.datetime.now()
    data['ts'] = int(datetime.datetime.timestamp(now) * 1000)
    print(f'spent time = {time.time() - start_time}, request for {topk}, return {vec_num} index')
    return json_result(code, msg, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', '--test', type=int, dest='test')

    # milvus参数
    parser.add_argument('-host', '--host', type=str, dest='host')
    parser.add_argument('-port', '--port', type=str, dest='port')
    parser.add_argument('-index_type', '--index_type', type=str, dest='index_type')   
    parser.add_argument('-metric_type', '--metric_type', type=str, dest='metric_type')
    parser.add_argument('-nprobe', '--nprobe', type=int, dest='nprobe')
    parser.add_argument('-embed_db_name', '--embed_db_name', type=str, dest='embed_db_name')
    parser.add_argument('-flask_host', '--flask_host', type=str, dest='flask_host')
    parser.add_argument('-flask_port', '--flask_port', type=str, dest='flask_port')

    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.host != None: host = args.host
    if args.port != None: port = args.port
    if args.index_type != None: index_type = args.index_type
    if args.metric_type != None: metric_type = args.metric_type  
    if args.nprobe != None: nprobe = args.nprobe  
    if args.embed_db_name != None: embed_db_name = args.embed_db_name  
    flask_host = args.flask_host 
    flask_port = args.flask_port 

    search_params = {
        "metric_type": metric_type,
        "params": {"ef": nprobe}
    }

    load_milvus(embed_db_name)
    app.run(flask_host, port=flask_port)
