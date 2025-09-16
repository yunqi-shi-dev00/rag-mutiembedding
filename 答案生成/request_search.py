import pandas as pd
import time
import hashlib
import requests
import json
import urllib
import random
import http

from time import sleep

def search_answer_online(query: str, port: int, top_doc_num: int = 5):
    """
    召回query相关的文本切片（chunk）
    """
    start_time = time.time()

    # 请求向量检索的flask服务，召回近似向量
    url = f'http://10.70.223.31:{port}/api-rqa-search/search'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Content-Length':'<calculated when request is sent>',
        'Accept-Encoding':'gzip, deflate, br'
    }
    sent_data = {'id': 102, 
                 'top_doc_num': top_doc_num,
                 'query': query}
    res_json = requests.post(url, verify=False, headers=headers, data=sent_data,  timeout=(3600, 3600)).content.decode('utf-8')

    res_data = json.loads(res_json).get('data')


    # print(res_data)
    res_num = res_data.get('doc_num')
    # res_num = 5
    results = []
    for i in range(res_num):
        ans = res_data.get('arr')[i]
        # print(f"rank={i}, score={ans['score']}, chunk={ans['text'][:100]}")
        # results.append([ans['ans_id'],ans['doc_name'],ans['text'],ans['score'], ans['kw_zh_all']])
        # results.append([ans['ans_id'],ans['doc_name'],ans['text'],ans['total_chunk_index']])
        # if ans.get('title'):
        results.append([ans['ans_id'],ans['title'],ans['text'],ans['index'], ans['score']])
        # else:
        #     results.append([ans['ans_id'],ans['doc_name'],ans['text'],ans['index'], ans['score']])
    if res_num < 5:
        for i in range(5-res_num):
            results.append(['null']*4+[0.0])

    sleep(3)
    # print(f'process query spent time = {time.time()-start_time}')
    return results

def baidu_translate(text: str, from_lang: str, to_lang: str, appid: str = '20180319000137450', key: str = 'gphHbEXij5p_v1c_syYz'):
    """
    调用百度api中译英
    """
    # 通用翻译API HTTP地址
    url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    salt = random.randint(3276, 65536)

    sign = appid + text + str(salt) + key
    sign = hashlib.md5(sign.encode()).hexdigest()
    url += f'?appid={appid}&q={urllib.parse.quote(text)}&from={from_lang}&to={to_lang}&salt={str(salt)}&sign={sign}'

    # 建立会话，返回结果
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', url)
        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)
        httpClient.close()
        # print(result)
        return result['trans_result'][0]['dst']

    except Exception as e:
        print(e)
        return 'null'
    
# def longest_common_substring(s1, s2):
#     m, n = len(s1), len(s2)
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
#     max_length = 0  # 最长公共子字符串的长度
#     end_index = 0  # 最长公共子字符串在 s1 中的结束位置

#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             if s1[i - 1] == s2[j - 1]:
#                 dp[i][j] = dp[i - 1][j - 1] + 1
#                 if dp[i][j] > max_length:
#                     max_length = dp[i][j]
#                     end_index = i
#             else:
#                 dp[i][j] = 0

#     # 提取最长公共子字符串
#     longest_common_sub = s1[end_index - max_length: end_index]
#     return longest_common_sub

if __name__=="__main__":
    query = '有关封框胶的问题。为了起到接着固定TFT和CF基板以及封闭液晶的作用，封框胶需要具备哪些特性'
    query_en = baidu_translate(query,"zh","en") #translate query
    data_re=search_answer_online(query, port = 1508)
    data_en_re=search_answer_online(query_en, port= 1508)[0]
    # data_collect=[]
    # for i,item in enumerate(data_re):
    #     data_collect.append(data_re[i]+data_en_re[i])

    # i=0
    # for item in data_collect:
    #     print(i)
    #     print(item)
    #     i+=1
    # print(data_re[1])
    print(data_re)
    print(data_en_re)

    print("done")

