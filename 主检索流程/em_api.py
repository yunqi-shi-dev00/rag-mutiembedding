from flask import Flask, request, jsonify
from FlagEmbedding import FlagLLMModel
from FlagEmbedding import BGEM3FlagModel

# Initialize the model
# model = FlagLLMModel('/data/user/rqa/FlagEmbedding/data/bge-multilingual-gemma2', use_fp16=True)
model_dir="/data/user/rqa/FlagEmbedding/data/trained_em_model/fv6"
# 初始化新的嵌入模型
embed_model = BGEM3FlagModel(model_dir,  
                       use_fp16=True)
# Create Flask app
app = Flask(__name__)

@app.route('/encode', methods=['POST'])
def encode_queries():
    try:
        # Parse JSON request
        data = request.get_json()
        if not data or 'queries' not in data:
            return jsonify({'error': 'Invalid input, please provide a list of queries'}), 400

        queries = data['queries']
        if not isinstance(queries, list):
            return jsonify({'error': 'Queries should be a list'}), 400

        # Generate embeddings
        embeddings = embed_model.encode(queries)['dense_vecs']
        return jsonify({'embeddings': embeddings.tolist()[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
