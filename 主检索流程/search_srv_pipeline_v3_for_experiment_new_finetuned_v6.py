# coding=utf-8

import chunk
import configparser
import datetime
import faulthandler
import jieba
import jieba.analyse
import json
import warnings
import os
import pymongo
import random
import re
import requests
import sys
import time
import tiktoken
import traceback
import torch
import warnings
import numpy as np
import pandas as pd

from apscheduler.schedulers.background import BackgroundScheduler
from elasticsearch import Elasticsearch
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer, util, models
from zhkeybert import KeyBERT, extract_kws_zh
from retriever.retriever_memory_keywords import read_keywords_with_score_from_memory
from keybert import KeyBERT as KBERT
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from FlagEmbedding import FlagLLMModel

def zhipu_translate(id: int, text: str, from_lang: str, to_lang: str):
    """
    调用质谱api中译英
    """
    # 质谱翻译API HTTP地址
    query_target = ''
    url = 'https://rqa-test.t-knows.com/api-rqa-web/v1/tcl_translate'
    headers = {
        'Content-Length': '<calculated when request is sent>',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'id': id,
        'query': text,
        'src_lang': from_lang,
        'dst_lang': to_lang
    }
    try:
        res_json = requests.post(url, verify=False, headers=headers, data=data,
                                 timeout=(30, 30)).content.decode('utf-8')
        json_data = json.loads(res_json)
        query_target = json_data['target']
    except Exception as e:
        print(f'get translate query failed!, return={e}')
    return query_target

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        # self.fc = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, 1024),
        #             nn.Linear(1024, 768),
        #             nn.Linear(768, num_classes)]
        #             )
        self.active = nn.LeakyReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        # for layer in self.fc:
        #     x = layer(x)
        #     x = self.active(x)
        # return x
        return logits

class mongodb:
    def __init__(self, url, db_name, table_name):
        self.url = url 
        self.db_name = db_name 
        self.table_name = table_name
        self.client = None
        self.db = None
        self.collection = None

    def connect(self):
        self.client = pymongo.MongoClient(self.url)
        self.db = self.client[self.db_name]
        print(f'db {self.db} collections={self.db.list_collection_names()}')
        self.collection = self.db[self.table_name]

    def insert_data(self, data):
        ##增加多条
        # stu2={'id':'002','name':'lisi','age':15}
        # stu3={'id':'003','name':'wangwu','age':20}
        # result = collection.insert_many([stu2,stu3])
        result = self.collection.insert_many(data)
        print(result)
        return
    
    def insert_one(self, data):
        result = self.collection.insert_one(data)
        # print(result)
        return

    def find_data(self, conditions):
        return self.collection.find(conditions)

sys.path.append("..")

from flask import Flask, request, Response, jsonify

# embedding 模型
# MODEL_NAME = '/mnt/hdd1/haoyangliu/em_model/embedding_res'
MODEL_NAME = '/mnt/hdd1/haoyangliu/em_model/bge-multilingual-gemma2'
# query class 模型
QUERY_CLASS_MODEL = BERTClassifier('/home/tcl/rqa_dir/query_class_model/bert-base-chinese', 2).to('cpu')
QUERY_CLASS_TOKENIZER = BertTokenizer.from_pretrained('/home/tcl/rqa_dir/query_class_model/bert-base-chinese')
SOFTMAX = nn.Softmax(dim=0)

PROFESSIONAL_DICT = set()

# 连接mongo数据库相关的变量
MONGO_URL = 'mongodb://root:example@10.70.223.31:27017'
MONGO_DB = 'rqa'
#VERSION = '20230908'
VERSION = '2023091110'


# TABLE_NAME = "test_table_lhy"

# 加载MongoDB的相关组件
MONGO_PIPELINE = None
MONGO_PIPELINE_RANK = None
DOWNLOAD_PIPELINE = None

# 加载向量库支持向量检索相关的变量
EMBED_MODEL = None  # SentenceTransformer('distiluse-base-multilingual-cased-v1')
ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")

# 加载ES相关组件
ES = Elasticsearch('http://10.70.222.234:9200')


#核心组件
MILVUS_BIND ='http://8.130.169.133:8010/api-vec-search/search'
MONGODB_C_NAME = "allchunk_newocr_20250723_wzh"
ENCODER_URL="http://8.130.169.133:8012/encode"
RERANKER_URL='http://8.130.169.133:8009/query_bge_reranker/'



TABLE_NAME = MONGODB_C_NAME

# recall
RETRIEVE_CHUNK_NUM = 4000
ES_RETRIEVE_CHUNK_NUM = 400

# rank
RANK_EMBED_DB = MONGODB_C_NAME
RANK_MODEL = None #SentenceTransformer('distiluse-base-multilingual-cased-v1')
DOC_SCORE_CHUNK_NUM = 8
SCORE_THREHOLD = 40  # score是检索embed相似度阈值，低于阈值不展示
TOKEN_LIMIT = 4000
TOKEN_ENCODE_MODEL = 'gpt-3.5-turbo'
CONCAT_CHUNK_NUM = 4

# memory keyword match
MEMORY_KEYWORD_MATCH = {}
MEMORY_QUERY_MATCH = {}

# keyword extract model
KW_ZH_MODEL = KeyBERT(model='/mnt/hdd1/haoyangliu/em_model/kw/paraphrase-multilingual-MiniLM-L12-v2')
KW_MODEL = KBERT(model='/mnt/hdd1/haoyangliu/em_model/kw/paraphrase-multilingual-MiniLM-L12-v2')

# nlp相关组件
PORTER_STEMMER = PorterStemmer() # 词干提取
REGEX_PATTERN = '|'.join(map(re.escape, [',', '\n', ';', '!', '?', '.', ' ', '~']))

app = Flask(import_name=__name__)
app.config['JSON_AS_ASCII'] = False

def has_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')  # 匹配中文字符的正则表达式
    return bool(re.search(pattern, text))*-1

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def mmrStep(lambda_param: float, selectedDocs: list, sorteddDocs: list, simMatrix: dict):
    """
    lambda_param: to balance diversity && relevance
    """
    mmr = -9999.999
    res_index = -1
    if not selectedDocs:
        selectedDocs.append(sorteddDocs[0]['index'])
        return

    for infoDict in sorteddDocs:
        index = infoDict['index']
        if index in selectedDocs:
            continue
        # relevance score
        mmr_score = lambda_param * infoDict.get('final_score', 0.0)
        diversity_score = max(
            [simMatrix.get((min(index, selected_index), max(index, selected_index)), 0.0) for selected_index in
             selectedDocs])
        mmr_score -= (1 - lambda_param) * diversity_score*30
        if mmr_score > mmr:
            mmr = mmr_score
            res_index = index

    selectedDocs.append(res_index)
    return

def load_data(table: str, model_name: str):
    # 加载需要在serve中持久化的item
    print('loading global data')
    start_time = time.time()

    # global CHROMA
    global ENCODING
    global EMBED_MODEL
    global MONGO_PIPELINE
    global MONGO_PIPELINE_RANK
    global RANK_MODEL
    global QUERY_CLASS_MODEL
    global QUERY_CLASS_TOKENIZER
    global PROFESSIONAL_DICT

    # 加载MongoDB连接相关组件
    MONGO_PIPELINE = mongodb(MONGO_URL, MONGO_DB, table)
    MONGO_PIPELINE.connect()
    client = pymongo.MongoClient("mongodb://root:example@10.70.223.112:27017")
    db = client['scrapy']
    MONGO_PIPELINE_RANK = db[MONGODB_C_NAME]
    # client = pymongo.MongoClient("mongodb://rqa:intermilano@10.70.223.31:27017")
    # db = client['rqa']
    # MONGO_PIPELINE_RANK = db['m3e_finetune_embed_2023091111']
    # print(f'[load_data] rank mongo list = {db.list_collection_names()}')
    # print(f'[load_data] table columns=[{MONGO_PIPELINE_RANK.find_one()}]')

    # 加载emebdding模型
    # EMBED_MODEL = FlagLLMModel(MODEL_NAME,  
    #                    use_fp16=True)
    # # word_embedding_model = models.Transformer('/home/tcl/.cache/torch/sentence_transformers/m3e_0906/model')
    # # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
    # #                                pooling_mode="mean"
    # #                                )
    # # RANK_MODEL = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # RANK_MODEL = FlagLLMModel(MODEL_NAME,  
    #                    use_fp16=True)

    # 加载token计算模块
    jieba.load_userdict("config/ext_dict2.dct")
    ENCODING = tiktoken.encoding_for_model(TOKEN_ENCODE_MODEL)

    # 加载query分类模型
    # QUERY_CLASS_MODEL = BERTClassifier('/home/tcl/rqa_dir/query_class_model/bert-base-chinese', 2).to('cpu')
    # QUERY_CLASS_TOKENIZER = BertTokenizer.from_pretrained('/home/tcl/rqa_dir/query_class_model/bert-base-chinese')
    ckpt = torch.load('/home/tcl/rqa_dir/query_class_model/query_classifier1.pth',map_location=torch.device('cpu'))
    QUERY_CLASS_MODEL.load_state_dict(ckpt,strict=False)

    fkw = open('/home/tcl/rqa_dir/ext_dict2.dct')
    for line in fkw:
        PROFESSIONAL_DICT.add(line.strip('\n'))
    fkw.close()

    # 加载MEMORY
    global MEMORY_KEYWORD_MATCH
    global MEMORY_QUERY_MATCH

    MEMORY_QUERY_MATCH, MEMORY_KEYWORD_MATCH = read_keywords_with_score_from_memory(
        'config/memory_keywords_recall_v20230916.txt')
    # print(f"MEMORY_QUERY_MATCH ={MEMORY_QUERY_MATCH}")

    # keyword提取
    global KW_ZH_MODEL
    global KW_MODEL
    global PORTER_STEMMER
    KW_ZH_MODEL = KeyBERT(model='/mnt/hdd1/haoyangliu/em_model/kw/paraphrase-multilingual-MiniLM-L12-v2')
    KW_MODEL = KBERT(model='/mnt/hdd1/haoyangliu/em_model/kw/paraphrase-multilingual-MiniLM-L12-v2')
    PORTER_STEMMER = PorterStemmer()

    print(f'load global data from mongo=[{table}] spent_time={time.time() - start_time}')
    return

def crontab_update_config():
    """
    实现动态加载配置
    """
    config = configparser.ConfigParser()
    config.read('./config/search_srv_pipeline.ini', encoding='UTF-8')
    # config.read( os.path.join(os.path.dirname( __file__ ),'config','search_srv_pipeline_multi.ini'), encoding='UTF-8')

    # 声明全局变量

    # embed模型相关变量
    # global MODEL_NAME
    # global TOKEN_ENCODE_MODEL

    # mongo查询相关变量
    # global TABLE_NAME
    # global VERSION
    # MONGO_URL = config['mongo']['mongo_url']
    # MONGO_DB = config['mongo_db']['mongo_db']

    # 召回相关变量
    # global MILVUS_BIND
    global RETRIEVE_CHUNK_NUM
    global SCORE_THREHOLD

    # 重排相关变量
    global DOC_SCORE_CHUNK_NUM
    global TOKEN_LIMIT
    global CONCAT_CHUNK_NUM

    # recall
    # RETRIEVE_CHUNK_NUM = int(config['recall']['recall_doc_num'])
    # MILVUS_BIND = config['recall']['milvus_bind']

    # rank

    # rerank

    # resort
    # SCORE_THREHOLD = float(config['resort']['score_threhold'])  # score是检索embed相似度阈值，低于阈值不展示
    DOC_SCORE_CHUNK_NUM = int(config['resort']['doc_score_chunk_num'])  # score是检索embed相似度阈值，低于阈值不展示

    # concat
    # TOKEN_LIMIT = int(config['concat']['token_limit'])
    # CONCAT_CHUNK_NUM = int(config['concat']['concat_chunk_num2'])
    TOKEN_ENCODE_MODEL = config['concat']['token_encode_model']

    print(
        f'[update config] 执行定时任务(minute=*/3)@{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}：{__name__}; Reading TOKEN_LIMIT={TOKEN_LIMIT}, SCORE_THREHOLD={SCORE_THREHOLD}, RETRIEVE_CHUNK_NUM={RETRIEVE_CHUNK_NUM}')
    return

def check_query_relevance(query: str) -> bool:
    """
    check query is relevant to semiconductor field:
    (1) Bert Classification Model
    (2) Max recall score
    (3) Keyword count
    """
    QUERY_CLASS_MODEL.eval()
    encoding = QUERY_CLASS_TOKENIZER(query, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to('cpu')
    attention_mask = encoding['attention_mask'].to('cpu')

    with torch.no_grad():
        outputs = QUERY_CLASS_MODEL(input_ids=input_ids, attention_mask=attention_mask)
        # _, preds = torch.max(outputs, dim=1)

    probs = SOFTMAX(torch.squeeze(outputs, dim=0))
    # print(f'outputs={outputs}, probs: {probs}')
    model_score = probs.detach().cpu().numpy()[1]
    flag_query_rel = model_score >= 0.4
    print(f'[check_query_relevance] Model score: {model_score}, text=[{query}]')
    return flag_query_rel

# def recall_pipeline(query, query_embed, result_dict: dict):
def recall_pipeline(**kwargs):
    """
    召回query相关的文本切片（chunk）
    """
    # print('################## process recall pipeline ##################')
    id: int = kwargs['id']
    query: str = kwargs['query']
    query_expand = kwargs['query_expand']
    query_embed_expand = kwargs['query_embed_expand']
    query_en: str = kwargs['query_en']
    query_zh: str = kwargs['query_zh']
    query_embed: np.ndarray = kwargs['query_embed']
    result_dict: dict = kwargs['result_dict']
    query_embed_expand = [query_embed]
    ############# kw extraction
    keywords, keywords_en, flag_zh = [], [], True
    if has_chinese(query):
        keywords = extract_kws_zh(query, KW_ZH_MODEL, ngram_range=(1, 1))
    else:
        keywords = KW_MODEL.extract_keywords(query)
        flag_zh = False
    kw0 = [k for (k, _) in keywords]


    # 加入jieba切分的token
    jieba_token_num = sum(1 for _ in jieba.cut(query))
    kw_jieba = jieba.analyse.extract_tags(query, allowPOS=['nz', 'nr', 'vd', 'n', 'vn', 'x', 'eng', 'v'],
                                          topK=jieba_token_num // 2)
    # kw_jieba_en = jieba.analyse.extract_tags(query_en, allowPOS=['nz', 'nr', 'vd', 'n', 'vn', 'x', 'eng', 'v'],
    #                                       topK=jieba_token_num // 2)
    kw_zh, kw_en = [], []
    # if flag_zh:
    kw_zh += kw0
    for kw in kw_jieba:
        kw_zh.append(kw)
    kwargs['kw_zh'] = set(kw_zh)
    # Chn. to Eng.
    # kw_en = zhipu_translate(id,','.join(kw_zh), 'zh', 'en').split(',')
    kw_en = KW_MODEL.extract_keywords(query_en)
    keywords_en_set = set([PORTER_STEMMER.stem(kw[0]) for kw in kw_en])
    kwargs['kw_en'] = keywords_en_set
    # else:
    #     kw_en += [PORTER_STEMMER.stem(k) for k in kw0]
    #     for kw in kw_jieba:
    #         kwst = PORTER_STEMMER.stem(kw)
    #         if kwst not in kw_en:
    #             kw_en.append(kwst)
    #     kwargs['kw_en'] = set(kw_en)
    #     # Eng. to Chn.
    #     # kw_zh = zhipu_translate(id, ','.join(kw_en), 'en', 'zh').split(',')
    #     kw_zh = extract_kws_zh(query_zh, KW_ZH_MODEL, ngram_range=(1, 1))
    #     kwargs['kw_zh'] = set(kw_zh)

    print(f'keyBert kw=[{kw0}],keywords=[{kw_zh}] get keywords in Eng=[{kw_en}]')

    ############# recall pipeline 1: sentence embedding -> chunk embedding
    
    url = MILVUS_BIND
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Content-Length': '<calculated when request is sent>',
        'Accept-Encoding': 'gzip, deflate, br'
    }
    res_vec_num = 0
    embed_recall_res = {}
    recall_max_score = 0.0  # max recall score
    for query_embed1 in query_embed_expand:
        
        sent_data = {'topk': RETRIEVE_CHUNK_NUM,
                     'query_vec': json.dumps(query_embed1) }
        res_json = requests.post(url, verify=False, headers=headers, data=sent_data, timeout=(5, 5)).content.decode(
            'utf-8')
        res_data = json.loads(res_json).get('data')
        res_arr = [t.split(':') for t in res_data.get('arr')]
        for [id_s, score_s, doc_s] in res_arr:
            # res_key = (int(id_s), int(doc_s))
            res_key = (int(id_s))
            if res_key in embed_recall_res:
                embed_recall_res[res_key] = max(float(score_s), embed_recall_res[res_key])
            else:
                embed_recall_res[res_key] = float(score_s)
                res_vec_num += 1
    print(f'[Recall Pipeline] after embed query chunks num = {res_vec_num}')
    if res_vec_num < RETRIEVE_CHUNK_NUM:
        print("Not enough retrieve chunk number")

    for ((id_s, score_s)) in embed_recall_res.items():
        dct = {}
        # 注意向量检索查询ids时转为了str类型
        dct['index'] = id_s
        dct['recall_score'] = score_s
        dct['final_score'] = -1.0
        dct['doc_name'] = 'null'
        dct['recall_channels'] = set(['sentence_embed'])
        result_dict[int(id_s)] = dct
        if recall_max_score < score_s: recall_max_score =score_s
    print(f'[Recall Pipeline] finally mivlus search recall num = {len(result_dict)}')

    ############# check query relevance
    key_match_cnt = len(PROFESSIONAL_DICT.intersection(set(kw_zh)))
    print(f'[Recall Pipeline] find kw_set={kw_zh},key_match_cnt={key_match_cnt},recall_max_score={recall_max_score}')
    if key_match_cnt < 2 and recall_max_score <= 0.5:
        # st1 = time.time()
        flag_query_rel = True
        nq = len(query_expand)
        for iq in range(nq):
            flag_query_rel = check_query_relevance(query_expand[iq])
            if flag_query_rel:
                break
        # print(f'[Recall Pipeline] query classification use time = {time.time()-st1}')
        if not flag_query_rel:
            kwargs['flag_query_rel'] = False
            return kwargs

    ############# recall pipeline 2: keyword embedding -> chunk match
    keywords, keywords_en, flag_zh = [], [], True
    if has_chinese(query):
        keywords = extract_kws_zh(query, KW_ZH_MODEL, ngram_range=(1, 1))
    else:
        keywords = KW_MODEL.extract_keywords(query)
        flag_zh = False

    # kw0 = [k for (k, _) in keywords]
    kw0 = None
    if kw0:
        # 请求向量检索的flask服务，召回近似向量
        url_kw = f'http://10.70.222.234:5200/api-kw-search/search'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Content-Length': '<calculated when request is sent>',
            'Accept-Encoding': 'gzip, deflate, br'
        }
        sent_data_kw = {'topk': 128,
                        'kw': ','.join(kw0)}
        res_json_kw = requests.post(url_kw, verify=False, headers=headers, data=sent_data_kw,
                                    timeout=(5, 5)).content.decode('utf-8')
        res_data_kw = json.loads(res_json_kw).get('data')
        res_vec_num_kw = res_data_kw.get('vec_num', 0)
        res_arr_kw = [t.split(':') for t in res_data_kw.get('arr')]
        kw_embed_cnt = 0
        for i in range(res_vec_num_kw):
            dct = {}
            [id_s, score_s] = res_arr_kw[i]
            # 注意向量检索查询ids时转为了str类型
            dct['index'] = int(id_s)
            dct['recall_score'] = 0.0
            dct['final_score'] = -1.0
            dct['doc_name'] = 'null'
            if int(id_s) in result_dict:
                result_dict[int(id_s)].get('recall_channels', set(['sentence_embed'])).add('kw_embed')
            else:
                dct['recall_channels'] = set(['kw_embed'])
                kw_embed_cnt += 1
            result_dict[int(id_s)] = dct
        print(f'[Recall Pipeline] keyword embed recall num ={kw_embed_cnt}')


    ############# recall pipeline 3: BM25 through Elasticsearch
    es_cnt = 0
    for query1 in kwargs['query_expand']:
        query_contains = {
            'match': {
                'chunk_text': query1
            }
        }
        es_res = ES.search(index="all_shards_finetuned_20241022_lhy", query=query_contains, size=ES_RETRIEVE_CHUNK_NUM)["hits"][
            "hits"]

        for r in es_res:
            dct = {}
            score = r["_score"] * 1.0
            id = r["_source"]["id"]
            dct['index'] = id
            dct['final_score'] = -1.0
            dct['doc_name'] = 'null'
            dct['recall_score'] = 0.0
            if id in result_dict:
                dct['bm25_score'] = max(score, result_dict[id].get('bm25_score', 0.0))
                dct['recall_score'] = result_dict[id].get('recall_score', 0.0)
                result_dict[id].get('recall_channels', set(['sentence_embed'])).add('bm25')
            else:
                dct['bm25_score'] = score
                dct['recall_channels'] = set(['bm25'])
                es_cnt += 1
            result_dict[id] = dct
            # if es_cnt <= 5:
            #     print(f'ES BM25 rank=[{es_rank}],index=[{id}],score=[{score}],shard=[{r["_source"]["shard"]}]')
    print(f'[Recall Pipeline] es BM25 recall num = [{es_cnt}]')

    # ES bm25 search in Eng.
    try:
        # if not query_en:
        #     query_en = zhipu_translate(id=id, text=query, from_lang='zh', to_lang='en')
        print(f'[Recall Pipeline] query in eng = [{query_en}]')
        query_en_contains = {
            'match': {
                'chunk_text': query_en,
            }
        }
        es_res_en = \
        ES.search(index="all_shards_finetuned_20241022_lhy", query=query_en_contains, size=ES_RETRIEVE_CHUNK_NUM)["hits"]["hits"]
        for r in es_res_en:
            dct = {}
            id = r["_source"]["id"]
            score = r["_score"]
            dct['index'] = id
            dct['final_score'] = -1.0
            dct['doc_name'] = 'null'
            dct['recall_score'] = 0.0
            if id in result_dict:
                dct['bm25_score'] += result_dict[id].get('bm25_score', 0.0)
                dct['recall_channels'] = result_dict[id]['recall_channels'] + ['bm25']
            else:
                dct['recall_channels'] = ['bm25']
            result_dict[id] = dct

    except Exception as e:
        print(f'baidu translate not available, use Chinese search only')
    # kwargs['result_dict'] = result_dict
    print(f"final recall chunk num = [{len(result_dict)}], es_cnt={es_cnt}")
    return kwargs

def rank_pipeline(**kwargs):
    """
    对文本切片做精排打分
    """
    # print('################## process rank pipeline ##################')
    # start_time = time.time()
    query: str = kwargs['query']
    query_embed: np.ndarray = kwargs['query_embed']
    query_rank_embed: np.ndarray = kwargs['query_rank_embed']
    result_dict: dict = kwargs['result_dict']
    # print(f'start rank_pipeline, query embed shape = {query_rank_embed.shape}')
    # find rank embed in MongoDB based on search index
    search_index = [t['index'] for t in result_dict.values()]
    print(f'search_index len={len(search_index)} example={search_index[:100]}, search mongo version = {VERSION}')
    find_condition = {'index': {'$in': search_index}}
    search_start_time = time.time()

    # search 请求用于m3e精排的数据：m3e embed
    embed_info = {}
    score_iter = MONGO_PIPELINE_RANK.find(find_condition,
                                                 {'_id':0, 'index':1, 'embedding':1})
    # for r in score_iter:
    #     embed_info[int(r['index'])] = r['embedding']

    print(f'[rank_pipeline] embed_info cnt={len(embed_info)}, mongo search time = {time.time()-search_start_time}')

    # search 请求切片、doc_name数据
    data_iter = MONGO_PIPELINE.find_data(find_condition)
    cnt = 0
    for dct in data_iter:
        cnt += 1
        index = dct['index']
        shard = dct['chunk_text']
        if len(shard) < 50:
            result_dict.pop(index)
            continue
        shard_id = dct['total_chunks_in_doc']
        doc_name = dct['doc_name']
        title = dct['title']

        shard = re.sub(r"\[[0-9].{0,5}\]", "", shard)

        row = result_dict[index]
        rank_score = 0
        if index in embed_info:
            rank_embed = np.array(embed_info[index])
            # rank_embed = RANK_MODEL.encode(shard, normalize_embeddings=True).astype(np.float64)
            row['rank_embed'] = rank_embed
            query_rank_embed = torch.tensor(query_rank_embed, dtype=torch.float32)
            rank_embed = torch.tensor(rank_embed, dtype=torch.float32)
            rank_score = util.cos_sim(query_rank_embed.to(torch.float32), rank_embed.to(torch.float32)).item()

        row['rank_score'] = rank_score
        row['final_score'] = row['recall_score']
        row['doc_name'] = doc_name
        row['title']=title
        row['chunk_text'] = shard
        row['total_chunks_in_doc'] = shard_id

    if cnt < RETRIEVE_CHUNK_NUM:
        print(f'mongodb err detect mongo num = {cnt} expect: {RETRIEVE_CHUNK_NUM}')
        # return

    ############# keyword -> doc_id match :
    # print(f'recall by keywords: query={query}')
    keyword_docid = {}
    # query match
    keys = MEMORY_QUERY_MATCH.keys()
    query1 = query.replace('?', '').replace('？', '').replace("\\n", '').strip(' ')
    if query1 in keys:
        keyword_docid = MEMORY_QUERY_MATCH[query1]

    # keyword match
    for (keys, doc_wieghts) in MEMORY_KEYWORD_MATCH.items():
        ks = keys.split(',')
        match_cnt = 0
        for k in ks:
            if k in query:
                match_cnt += 1
            else:
                break
        if match_cnt == len(ks):
            for (doc, w) in doc_wieghts.items():
                if doc not in keyword_docid or keyword_docid[doc] < w:
                    keyword_docid[doc] = w

    if keyword_docid:
        # print(f'By keyword find docId = {keyword_docid}')
        find_condition2 = {'doc_id': {'$in': list(keyword_docid.keys())}}
        data_iter2 = MONGO_PIPELINE.find_data(find_condition2)
        cnt2, cnt_final = 0, 0

        # 同时补充score的m3e embed
        score_iter2 = MONGO_PIPELINE_RANK.find(find_condition2,
                                                 {'_id':0, 'index':1, 'embedding':1})
        for r in score_iter2:
            embed_info[int(r['index'])] = r['embedding']
        print(f'after boosting by kw embed_info cnt={len(embed_info)}')

        # 每个doc_id取top-5
        docid_index_dict = {}
        weighted_doc_info = {}
        for dct in data_iter2:
            doc_id = int(dct['doc_id'])
            index = dct['index']
            if index in result_dict:
                result_dict[index]['final_score'] = result_dict[index]['final_score'] * keyword_docid[doc_id]
            shard = dct['chunk_text']
            if len(shard) < 50:
                continue
            shard_id = dct['total_chunks_in_doc']
            doc_name = dct['doc_name']
            if doc_id not in weighted_doc_info:
                weighted_doc_info[doc_id] = (doc_name, keyword_docid[doc_id])
            shard = re.sub(r"\[[0-9].{0,5}\]", "", shard)

            row = {}
            rank_score = 0
            if index in embed_info:
                # Ensure query_rank_embed is a float32 tensor
                query_rank_embed = query_rank_embed.to(torch.float32)
                
                # Convert rank_embed to a PyTorch tensor and ensure it's float32
                rank_embed = np.array([float(x) for x in embed_info[index].split(',')])
                rank_embed = torch.tensor(rank_embed).to(torch.float32)
                
                # Calculate cosine similarity
                rank_score = util.cos_sim(query_rank_embed, rank_embed).item()
                
                # Store rank_embed back in the row dictionary if needed
                row['rank_embed'] = rank_embed
            row['index'] = index
            row['recall_score'] = 0.0
            row['rank_score'] = rank_score
            row['final_score'] = rank_score * keyword_docid[doc_id]
            row['doc_name'] = doc_name
            row['title']=title
            row['chunk_text'] = shard
            row['total_chunks_in_doc'] = shard_id
            # result_dict[index] = row

            if doc_id not in docid_index_dict:
                docid_index_dict[doc_id] = [row]
            else:
                docid_index_dict[doc_id].append(row)
            cnt2 += 1

        print(f'weighted_doc_info = {weighted_doc_info}')
        # 每个doc_id取top-5
        for (doc_id, rows) in docid_index_dict.items():
            doc_top_rows = sorted(rows, key=lambda d: -d['final_score'])[:CONCAT_CHUNK_NUM * 2]
            for row in doc_top_rows:
                result_dict[row['index']] = row
                cnt_final += 1
                print(f"add doc id=[{doc_id}],name=[{row['doc_name']}],weight=[{keyword_docid.get(doc_id, 1.0)}],rank_score=[{row['rank_score']}], final_score=[{row['final_score']}]")

        print(f'Adding result: new cnt = {cnt2}, final adding cnt = {cnt_final}, adding files = {keyword_docid}')

    # for (k,v) in result_dict.items():
    #     if 'rank_score' not in v: print(f'缺少rank_score = {v}')
    print(f'search on mongo spent time = {time.time() - search_start_time}')
    return kwargs

def encode_from_net(querys):
    url=ENCODER_URL
    if isinstance(querys,list):
        payload = {
        "queries": querys
        }
    else:
        payload = {
        "queries": [querys]
        }

    # Set headers
    headers = {
        "Content-Type": "application/json"
    }

    # Send POST request
    response = requests.post(url, json=payload, headers=headers)

    # Print the response
    # if response.status_code == 200:
    #     print("Embeddings:", response.json())
    return response.json()['embeddings']

def rerank_pipeline(** kwargs):
    print('################## process rerank pipeline ##################')
    bge_server_url = RERANKER_URL
    # bge_server_url = 'http://10.120.105.223:35229/query_bge_reranker/'
    bge_score_weight = 3.0

    query: str = kwargs['query']
    # query_embed: np.ndarray = kwargs['query_embed']
    result_dict: dict = kwargs['result_dict']
    
    sorted_top_chunk = sorted(result_dict.items(), key=lambda d: -d[1]['final_score'])
    result_dict_sorted = {}
    bge_score_buff_dict = [[], []]
    add_en_num = 0
    sid_en_flag = False
    
    """ 
    for line in open('/home/tcl/rqa_dir/model/sid_en_target_doc.txt'):
        line = line.strip().split('\t')
        line_query_dis = levenshteinDistance(query, line[0])
        if line_query_dis / max(len(line[0]), len(query)) < 0.2:
            query_en_targe_year, query_en_targe_year_name = line[1], line[2]
            sid_en_flag = True
            break
    """
        
    for index in range(len(sorted_top_chunk)):
        if not ('chunk_text' not in sorted_top_chunk[index][1]):
            result_dict_sorted[sorted_top_chunk[index][0]] = sorted_top_chunk[index][1]
            if index < 500 :
                bge_score_buff_dict[0].append(sorted_top_chunk[index][0])
                bge_score_buff_dict[1].append([query, sorted_top_chunk[index][1]['chunk_text']])
                
                if len(bge_score_buff_dict[0]) == 15:
                    bge_multi_data = {'type': 'multi', 'multi_data': bge_score_buff_dict[1]}
                    bge_rerank_score_data_list = requests.post(bge_server_url, data=json.dumps(bge_multi_data))
                    # print(bge_rerank_score_data_list.json())
                    bge_rerank_score_list = bge_rerank_score_data_list.json()['score']
                    # print(len(bge_rerank_score_list))
                    # print(bge_score_buff_dict[0])
                    assert len(bge_rerank_score_list) == len(bge_score_buff_dict[0]) 
                    for score_index in range(len(bge_rerank_score_list)):
                        rank_score = result_dict_sorted[bge_score_buff_dict[0][score_index]]['final_score'] + \
                                bge_score_weight * bge_rerank_score_list[score_index]
                        result_dict_sorted[bge_score_buff_dict[0][score_index]]['rerank_score'] = rank_score
                        result_dict_sorted[bge_score_buff_dict[0][score_index]]['final_score'] = rank_score
                    bge_score_buff_dict = [[], []]
            else:
                bge_rerank_score = -8.0
                rank_score = result_dict_sorted[sorted_top_chunk[index][0]]['final_score'] + bge_score_weight + bge_rerank_score
                result_dict_sorted[sorted_top_chunk[index][0]]['rerank_score'] = rank_score
                result_dict_sorted[sorted_top_chunk[index][0]]['final_score'] = rank_score
            
            if sid_en_flag and query_en_targe_year in sorted_top_chunk[index][1]['doc_name'] and \
                query_en_targe_year_name in sorted_top_chunk[index][1]['doc_name'] and add_en_num == 0 :
                result_dict_sorted[sorted_top_chunk[index][0]]['final_score'] += 3
                result_dict_sorted[sorted_top_chunk[index][0]]['rerank_score'] = result_dict_sorted[sorted_top_chunk[index][0]]['final_score']
                add_en_num += 1 
            """
            elif add_en_num < 100 and (not has_chinese(sorted_top_chunk[index][1]['chunk_text'])):
                # add_en_num += 1
                bge_data = {
                             'query': query,
                             'chunk_text': sorted_top_chunk[index][1]['chunk_text'],
                        }
                if index == 0: print (bge_data)
                bge_rerank_score_data = requests.post(bge_server_url, data=json.dumps(bge_data))
                bge_rerank_score = bge_rerank_score_data.json()['score']
                rank_score = result_dict_sorted[sorted_top_chunk[index][0]]['final_score'] + bge_score_weight * bge_rerank_score
                result_dict_sorted[sorted_top_chunk[index][0]]['rerank_score'] = rank_score
                result_dict_sorted[sorted_top_chunk[index][0]]['final_score'] = rank_score
            """


    if len(bge_score_buff_dict[0]) > 0:
        bge_multi_data = {'type': 'multi', 'multi_data': bge_score_buff_dict[1]}
        bge_rerank_score_data_list = requests.post(bge_server_url, data=json.dumps(bge_multi_data))
        bge_rerank_score_list = bge_rerank_score_data_list.json()['score']
        assert len(bge_rerank_score_list) == len(bge_score_buff_dict[0])
        for score_index in range(len(bge_rerank_score_list)):
            try:
                rank_score = result_dict_sorted[bge_score_buff_dict[0][score_index]]['final_score'] + \
                                bge_score_weight * bge_rerank_score_list[score_index]
            except:
                rank_score = result_dict_sorted[bge_score_buff_dict[0][score_index]]['final_score'] + \
                                bge_score_weight * bge_rerank_score_list[score_index][0]
            result_dict_sorted[bge_score_buff_dict[0][score_index]]['rerank_score'] = rank_score
            result_dict_sorted[bge_score_buff_dict[0][score_index]]['final_score'] = rank_score
        bge_score_buff_dict = [[], []]
             
    result_dict = result_dict_sorted

    #query kw 匹配调权
    kw_zh = kwargs['kw_zh']
    kw_en = kwargs['kw_en']

    kw_num, weight_num = len(kw_zh), 0
    for (id, row) in result_dict.items():
        match_cnt = 0
        if row.get('final_score', 0) < 0.4 or 'chunk_text' not in row:
            continue
        weight_num += 1
        shard = row['chunk_text']
        if has_chinese(shard):
            for kw in kw_zh:
                if kw in shard: match_cnt += 1
        else:
            tokens_set = set([PORTER_STEMMER.stem(t) for t in re.split(REGEX_PATTERN, shard)])
            match_cnt = len(tokens_set.intersection(kw_en))

        match_score = (0.95 + pow(match_cnt / kw_num, 2)) if kw_num > 0 else 0.001
        row['match_cnt'] = match_cnt
        row['match_score'] = match_score
        row['final_score'] = row['final_score'] * match_score
        # print(row['final_score'])
        # if random.random() < 0.01:
        # print(f"doc_name=[{row['doc_name']}],match_cnt={match_cnt}, match_score=[{match_score}], final_score={row['final_score']}")

    print(f'keyword weight num = {weight_num}')
    kwargs['result_dict'] = result_dict
    # print("after rerank")
    # print(kwargs)
    return kwargs

def concat_shards_by_rank(**kwargs):
    """
    1 根据打分列表得到top相关文档;
    2 由top相关文档拼接相关文本;
    @ 2023.10.30
    """
    query = kwargs['query']
    query_expand = kwargs['query_expand']
    # query_rank_embed_expand = kwargs['query_rank_embed_expand'][-1]
    result_dict = kwargs['result_dict']
    top_doc_num: int = kwargs['top_doc_num']
    is_debug: bool = kwargs['is_debug']
    target_doc_name = kwargs['target_doc_name']
    # start_time = time.time()
    sorted_top_chunk = sorted(list(result_dict.values()), key=lambda d: -d['final_score'])
    # 按照rank_score排序[dict{index, doc_name, rank_score, shard, shard_id}]
    # print(sorted_top_chunk)
    #
    start_time =time.time()

    simMatrix = {}
    if top_doc_num <= 5 :
        sim_num = 5 * 5 + 30
    else:
        sim_num = top_doc_num * 5 + 30
    for i in range(sim_num):
        info_i = sorted_top_chunk[i]
        index_i = info_i['index']
        for j in range(i + 1, sim_num):
            info_j = sorted_top_chunk[j]
            index_j = info_j['index']
            bigger, smaller = max(index_i, index_j), min(index_i, index_j)
    print(f'[concat_shards_by_rank] , using time={time.time()-start_time}')
    #
    selectedDocs = []
    for i in range(sim_num):
        mmrStep(0.9, selectedDocs, sorted_top_chunk, simMatrix)
    
    
    print(f'[concat_shards_by_rank] mmrStep, using time={time.time() - start_time}')
    print(f"[concat_shards_by_rank]compare: before mmr top10={[t['index'] for t in sorted_top_chunk[:20]]}")
    print(f"[concat_shards_by_rank]compare: after mmr top10={selectedDocs[:20]}")
    # 按照rank_score排序[dict{index, doc_name, rank_score, shard, shard_id}]

    res_arr_dict = {}
    res_arr = []
    res_num = 0

    min_concat_score = 99.9
    min_concat_id = -1

    i=0
    this_chunk_num = 0
    this_index = []
    this_res = {}
    this_docs = []
    this_shard = []
    this_title = []
    this_num_token = 0
    this_scores = 0.0

    chunks_id = 0

    for index in selectedDocs:
        i+=1
        dict = result_dict[index]
        doc_name = dict['doc_name']
        shard = dict['chunk_text']
        index = dict['index']
        title= dict['title']

        tokens = ENCODING.encode(shard)
        # if this_num_token + len(tokens) >= TOKEN_LIMIT:
        #     # 超过token数时取截断
        #     cut_token = TOKEN_LIMIT - len(tokens)
        #     shard = ENCODING.decode(tokens[:cut_token])
        #     this_num_token = TOKEN_LIMIT
        # else:
        #     this_num_token += len(tokens)
        this_num_token += len(tokens)
        # print(shard)
        this_shard.append(shard)
        this_index.append(index)
        this_title.append(title)
        this_chunk_num += 1
        this_scores += dict['final_score']


        this_docs.append(doc_name)

        if this_num_token >= TOKEN_LIMIT:
            this_res['text'] = this_shard
            this_res['ans_id'] = res_num
            # this_res['doc_name'] = this_docs
            # """
            this_res['doc_name'] = this_docs
            this_res['index'] = this_index
            # for doc_name in this_docs[10:]:
            #     if (not has_chinese(doc_name)) and (has_chinese(this_res['doc_name'][-1])):
            #         this_res['doc_name'][-1] = doc_name
            # """
            this_res['title'] = this_title
            this_res['score'] = this_scores / this_chunk_num
            # concat_embed = RANK_MODEL.encode(this_shard).astype(np.float64)
            # concat_score = util.cos_sim(concat_embed, query_rank_embed_expand).item()
            concat_score = 0.5
            this_res['text'] = this_shard
            this_res['concat_score'] = concat_score
            if is_debug:
                print(f'[concat_shards_by_rank] chunks_id={chunks_id}, score={concat_score}')
            if concat_score < min_concat_score:
                min_concat_score = concat_score
                min_concat_id = chunks_id
            
            # 解决mmr排序以后得分不递减问题
            res_arr_dict[(this_res['score'], tuple(this_res['text']))] = this_res


            # res_arr.append(this_res)

            this_chunk_num = 0
            this_res = {}
            this_docs = []
            this_shard = []
            this_index = []
            this_num_token = 0
            this_scores = 0.0
            chunks_id += 1
            res_num += 1
        
        if res_num >= top_doc_num:
            break
    
    sorted_res_arr_dict = sorted(res_arr_dict.items(), key=lambda d: -d[0][0]) 
    for item_index, sorted_item in enumerate(sorted_res_arr_dict):
        sorted_item[1]['ans_id'] = item_index
        res_arr.append(sorted_item[1])
    print(f'[concat_shards_by_rank] min concat_scores={min_concat_score}, chunks_id={min_concat_id}')
    # res_arr[min_concat_id]['text'] += query_expand[-1]

    if random.random() < 0.1 or is_debug:
        for i, d in enumerate(res_arr):
            print(f'[concat_shards_by_rank] rk={i}, res={d["text"][:200]}')

    # print(f'generate final doc chunck spent time = {time.time() - start_time}')
    return res_arr

def query_expand_srv(query: str, query_expand: list):
    # step 1 search in history for gpt-4 answer
    query_contains = {
        'match': {
            'question': query
        }
    }
    es_res = ES.search(index="query_answer_history_v1023", query=query_contains, size=3)["hits"]["hits"]
    candidates = []
    for r in es_res:
        dct = {}
        score = r["_score"]
        dct['question'] = r["_source"]["question"]
        dct['answer'] = r["_source"]["answer"]
        dct['score'] = score
        candidates.append(dct)

    for dct in sorted(candidates, key=lambda x: -x['score']):
        editsim = levenshteinDistance(query, dct['question'])
        if editsim <= 2:
            # print(f"[query_expand_srv] find question in searcg history {dct['question']}-{dct['answer']}")
            query_expand.append(dct['answer'])
            return

    # step 2 find GPT-4 answer in config
    gpt4_df = pd.read_excel('/home/tcl/rqa_dir/models_lxl/test_0222_60.xlsx')
    if query in gpt4_df['问题'].values: 
        hanghao = gpt4_df[gpt4_df['问题'] == query].index.to_list()[0]
        query_expand.append(gpt4_df.iloc[hanghao, 1])
        return
    else :
        have_ans_querys = list(gpt4_df['问题'].values)
        for query_index in range(len(have_ans_querys)):
            querys_distance = levenshteinDistance(query, have_ans_querys[query_index])
            max_query_len = max(len(query), len(have_ans_querys[query_index]))
            if querys_distance/max_query_len < 0.2:
                query_expand.append(gpt4_df.iloc[query_index, 1])
                print(query, have_ans_querys[query_index], gpt4_df.iloc[query_index, 1])
                return
    
    # '''
    # step 3 find short answer
    url = 'https://tcl-ai-france.openai.azure.com/openai/deployments/gpt-4-0314/chat/completions?api-version=2023-03-15-preview'
    headers = {
        'Content-Type': 'application/json',
        'api-key': '98ff3b4afac846a7bede351bcec20ce8'
    }
    temperature = 0
    messages = [
        {
            "role": "system",
            "content": "You are an expert in the field of semiconductor displays technology. "
        },
        {
            "role": "assistant",
            "content": ""
        },
        {
            "role": "user",
            "content": f"""请用中文简短地回答问题: '{query}'。字数在30个字以内。"""
            # "content": f"""Please answer the question briefly :'{query}'.and try not to exceed 50 words"""
        }
    ]
    data = json.dumps({"messages": messages, "temperature": temperature})
    try:
        response = requests.post(url, data=data, headers=headers)
        res = response.json().get('choices')[0].get('message').get('content')
        query_expand.append(res.strip(' '))
    except Exception as e:
        print(f'cannot find query expand by gpt-4 engine')
    return
    # '''


@app.route('/api-rqa-search/test', methods=['GET'])
def hello_world():
    return json_result(0, '', 'Service available')

@app.route('/api-rqa-search/search', methods=['POST'])
def get_data():
    """
    搜索服务主函数
    """
    form = request.form
    query = form.get('query', '', str)
    # query = query+" "+baidu_translate(query,"zh","en")
    # query = baidu_translate(query,"zh","en") #translate query
    query_en = form.get('query_dst', '', str)
    id = form.get('id', 0, int)
    is_debug = form.get('debug', 0, int) == 1
    top_doc_num = form.get('top_doc_num', 0, int)
    target_doc_name = form.get('target_doc_name', '', str)
    # add delete doc_id serve
    is_delete = form.get('is_delete', 0, int)
    delete_doc_id = form.get('delete_doc_id', '', str)
    if query == '':
        return json_result(-1, 'query must not be null.', None)
    if id == 0:
        return json_result(-1, 'id must not be null.', None)
    if top_doc_num == 0:
        return json_result(-1, 'top_doc_num must not be null.', None)
    if is_delete == 1 and delete_doc_id:
        delete_doc_id_list = [int(t) for t in delete_doc_id.split(',')]
        res = MONGO_PIPELINE.collection.delete_many({
            'doc_id': {'$in': delete_doc_id_list}
        })
        # for r in res:
        #     print(r)
        return json_result(0, f'delete {delete_doc_id_list} successfully', None)

    start_time = time.time()
    code = 0
    msg = ''
    data = {}
    data['model'] = MODEL_NAME
    data['version'] = VERSION
    # print(f'[get_data] reading TOKEN_LIMIT={TOKEN_LIMIT}, SCORE_THREHOLD={SCORE_THREHOLD}, RETRIEVE_CHUNK_NUM={RETRIEVE_CHUNK_NUM}')

    # try:
    if 1:
        # return JsonLst
        json_arr = []
        query_embed = encode_from_net(query)
        query_rank_embed = encode_from_net(query)

        result_dict = {}
        start_time = time.time()
        params = {}
        params['id'] = id
        params['query'] = query
        if has_chinese(query):
            params['query_zh'] = query
            params['query_en'] = zhipu_translate(id, query, 'zh', 'en')
        else:
            params['query_en'] = query
            params['query_zh'] = zhipu_translate(id, query, 'en', 'zh')
        print(f'[get_data] zhipu translate using {time.time()-start_time} seconds')
        query_expand = [query]
        # query_expand_srv(query, query_expand)
        print(f'query expand spent time = {time.time() - start_time}')
        params['query_expand'] = query_expand
        query_embed = encode_from_net(query)
        query_embed_expand = encode_from_net(query_expand)
        query_rank_embed_expand = encode_from_net(query_expand)
        print(f'[get_data] expand queries = {query_expand}, emb_len={len(query_embed_expand)}')

        params['query_embed'] = query_embed
        params['query_embed_expand'] = query_embed_expand
        params['query_rank_embed'] = query_rank_embed
        params['query_rank_embed_expand'] = query_rank_embed_expand
        params['target_doc_name'] = target_doc_name
        params['flag_query_rel'] = True
        
        if len(query_en.strip()) > 0:
            params['query_en'] = query_en
        params['result_dict'] = result_dict
        params = recall_pipeline(**params)
        print(f'recall spent time = {time.time() - start_time}')
        if not params['flag_query_rel']:
            code = 1
            data['msg'] = 'query must be relevant'
            data['doc_num'] = 0
            data['arr'] = []
            return json_result(code, msg, data)
        # 调用前

        params = rank_pipeline(**params)

        # 调用后

        # params = rank_pipeline(**params)
        print(f'rank spent time = {time.time() - start_time}')
        params = rerank_pipeline(**params)
        print(f'rerank spent time = {time.time() - start_time}')
        print(f'In request: query={query}, top_doc_num={top_doc_num}')
        params['top_doc_num'] = top_doc_num
        params['concat_num'] = CONCAT_CHUNK_NUM
        params['is_debug'] = is_debug
        similar_shards = concat_shards_by_rank(**params)
        # print('similar_shards: ', similar_shards)
        # similar_shards: [rank, doc_name, text, score]
        for dict in similar_shards:
            score = dict['score']
            print(score)
            if score < SCORE_THREHOLD:
                break
            json_arr.append(dict)
            # json_str = json.dumps(dict, indent=4, ensure_ascii=False)
            # json_arr.append(json_str)

        # 构造response
        data['arr'] = json_arr
        data['doc_num'] = len(json_arr)

    # except Exception as e:
    else:
        code = -1
        msg = traceback.format_exc()
        data['msg'] = msg
        data['doc_num'] = 0
        data['arr'] = []

    now = datetime.datetime.now()
    data['ts'] = int(datetime.datetime.timestamp(now) * 1000)
    print(f'spent time = {time.time() - start_time}')
    return json_result(code, msg, data)


@app.route('/api-rqa-search/download', methods=['GET'])
def download():
    doc_name = request.args.get('docName', '', str)
    query_id = request.args.get('queryId', 0, int)
    username = request.args.get('username', '', str)
    try:
        save_download_record(query_id, username, doc_name)
    except Exception as err:
        print('Failed to save download record:', err)
    file = '/mnt/hdd1/rqa_dir/db/papers_pdf/' + doc_name
    with open(file, 'rb') as f:
        stream = f.read()
    f.close()
    response = Response(stream, content_type='application/octet-stream')
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,HEAD,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Referer,Accept,Origin,User-Agent'
    return response


def save_download_record(query_id: int, username: str, doc_name: str):
    if query_id == 0:
        return
    if username == '':
        return
    global DOWNLOAD_PIPELINE
    if DOWNLOAD_PIPELINE is None:
        DOWNLOAD_PIPELINE = mongodb(MONGO_URL, MONGO_DB, "download_record")
        DOWNLOAD_PIPELINE.connect()
    document = {
        "query_id": query_id,
        "username": username,
        "doc_name": doc_name,
        "created_at": datetime.datetime.now()
    }
    DOWNLOAD_PIPELINE.insert_one(document)


# 0表示正常 -1表示接口出错 可以使用其他code适用于不同场景
def json_result(code: int, msg: str, data):
    return jsonify({'code': code, 'msg': msg, 'data': data})

faulthandler.enable()
print(f'start server at {time.time()}')

# 服务启动参数填充
config = configparser.ConfigParser()
config.read('./config/search_srv_pipeline_l.ini', encoding='UTF-8')
# config.read( os.path.join(os.path.dirname( __file__ ),'config','search_srv_pipeline_multi.ini'), encoding='UTF-8')
# embed
# MODEL_NAME = config['embedding']['embed_model_name']
print(f'load embed model name = {MODEL_NAME}')

# mongo
# TABLE_NAME = config['mongo']['table_name']
# VERSION = config['mongo']['version']
print(f'load mongo tbl={TABLE_NAME} , {type(TABLE_NAME)}, version={VERSION}')

# vec_search

scheduler = BackgroundScheduler()
scheduler.add_job(crontab_update_config, 'interval', seconds=180, coalesce=True, replace_existing=True)
scheduler.start()

load_data(TABLE_NAME, MODEL_NAME)

if __name__ == '__main__':
    app.run('10.70.223.31', port=1508)
    # app.run()

    # Test环境
    # gunicorn -c config/gunicorn.conf.py -b 10.70.223.31:5002 search_srv_pipeline_multi:app -n query2doc

    # 正式环境
