import os
# 设置环境变量强制使用本地文件
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from FlagEmbedding import LayerWiseFlagLLMReranker
import numpy as np
from FlagEmbedding import FlagReranker
from flask import Flask, request, jsonify
import json

app = Flask(__name__)
reranker = LayerWiseFlagLLMReranker(
    '/data/user/rqa/lhy_playground/model/bge-reranker-v2-minicpm-layerwise', 
    use_fp16=True  
)


@app.route('/query_bge_reranker/', methods=['post'])
def post_http():
    if not request.data:  # 检测是否有数据
        return ('fail')
    params = request.data.decode('utf-8')
    # 获取到POST过来的数据，因为我这里传过来的数据需要转换一下编码。根据晶具体情况而定
    prams = json.loads(params)
    if 'type' not in prams or prams['type'] == 'single' :
        query = prams['query']
        shard = prams['shard']
        score = reranker.compute_score([query, shard], cutoff_layers=[28])
    else:
        data_list = prams['multi_data']
        score = reranker.compute_score(data_list, cutoff_layers=[28])
        if len(data_list) == 1:
            score = [score]

    prams['score'] = score 
    prams_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in prams.items()}

    return jsonify(prams_serializable)

    # 返回JSON数据。


if __name__ == '__main__':
    # app.run(host='10.120.105.223', port=35227)
    app.run(host='0.0.0.0', port=8004)