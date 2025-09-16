import pandas as pd
import re
import request_search
import time
import threading
from openpyxl import Workbook
import requests
# 记录开始时间
start_time = time.time()
import os
from openai import OpenAI

data_en_re = None
data_re = None

sumres=0
# 读取Excel文件
# 将文件路径替换为你的文件路径

file_path ='/mnt/hdd1/haoyangliu/playground/data/100道全领域评测题.xlsx'
df = pd.read_excel(file_path)
query_array = df['Query'].to_numpy()
# query_array = df['question'].to_numpy()


# file_path = '/mnt/hdd1/haoyangliu/playground/data/微软问题&目标切片45.xlsx'  
# df = pd.read_excel(file_path,sheet_name=1)




def remove_specified_items(lst, items_to_remove):
    return [item for item in lst if item not in items_to_remove]

# 对“参考文献”列进行处理
def process_references(references):
    # 分割参考文献
    references_list = references.split('\n')
    if len(references_list)>1:
        references_list=remove_specified_items(references_list,'')
        for i, item in enumerate(references_list):
            match = re.search(r'[)\）]', item)
            if match:
                references_list[i] = item[match.end():]
    # 去除序号
    cleaned_references =  references_list
    return cleaned_references

# references_array = df['参考文献'].apply(process_references).to_numpy()
# chunk_lable=df['文献切片（充分非必要条件）'].to_numpy()

def check_intersection(arr1, arr2):
    for sub_arr1 in arr1:
        for sub_arr2 in arr2:
            if set(sub_arr1).intersection(set(sub_arr2)):
                print("t")
                print(sub_arr1)
                print(sub_arr2)
                return
    print("f")

def remove_file_extension(filename):
    # 使用 rsplit 方法从右边分割字符串，最多分割一次
    return filename.rsplit('.', 1)[0]

def get_chatglm_response(input_text):
    
    headers = {
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json',
    'Accept': '*/*',
    'Host': 'hz-model.bigmodel.cn',
    'Connection': 'keep-alive'
    }

    # data = {
    #     'inputs': f'<|user|>\n{input_text}<|assistant|>',
    #     'parameters': {
    #         'max_new_tokens': 900,
    #         'temperature': 0.01,
    #         'top_p': 0.01, 
    #         "stop":  ["<|endoftext|>","<|user|>","<|observation|>"] 
    #     },
    # }
    
    json_data = {
        'model': 'glm4',
        'stream': False,
        'top_p':0.01,
        'temperature':0.01,
        'messages':[
            {"role":"user",
             "content":input_text}
            ]
        }

    response = requests.post('http://10.70.223.193:8010/v1/chat/completions', headers=headers, json=json_data )

    # print(response.json())

    return response.json()['choices'][0]['message']['content']

def find_intersection(arr1, arr2):
    # 将数组转换为集合
    set1 = set(arr1)
    set2 = set(arr2)
    global sumres
    # print(set1)
    # print(set2)
    # 检查交集
    intersection = set1 & set2
    
    # 打印结果
    if intersection:
        sumres += len(intersection) 
        print('t')
        print(intersection)
    else:
        print('f')

def quest_step( query: str, passages: list):
    """
    Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
    :return: next thought
    """
    input_text = '''[角色]
    你是一名半导体显示技术专家，充分掌握半导体显示技术的复杂概念和细节，擅长对专业知识进行解答。


    [知识]
    """"""
    {know}
    """"""


    [问题]
    {question}


    [要求]
    1. [知识]存在对解答[问题]无关的内容，你需要对解答[问题]有效的内容进行提取和理解。
    2. 如果[知识]中未提供足够信息解答[问题]，则要回答需要相关背景知识。
    3. [知识]的内容庞杂，你需要把他们逻辑梳理准确，不得出现错误。
    4. 根据已有的[知识]，对问题进行详尽的回答，推荐分点回答格式。
    5. 对[问题]的解答要准确、无误。缺乏所需信息可以提出疑问。


    [回答]
    '''

    prompt_user = ''
    if passages[1]!="null":
        for i in range(len(passages[1])):
            prompt_user += f'Doc name: {passages[1][i]}\n'
            prompt_user += f'{passages[2][i]}\n\n'
    # print(prompt_user)
    input_text1=input_text.replace("{know}", prompt_user[:8000])
    input_text2=input_text1.replace("{question}", query)
    # answer=get_chatglm_response(input_text2)
    answer=get_deepseek_response(input_text2)
    return answer

def get_en_anser(item):
    print("en_start")
    query_en=request_search.baidu_translate(item,"zh","en")
    global data_en_re
    data_en_re=request_search.search_answer_online(query_en, port= 1508)[0]

def get_zh_anser(item):
    print("zh_start")
    global data_re
    data_re=request_search.search_answer_online(item, port = 1508)[0]

def get_deepseek_response(input_text):
    client = OpenAI(api_key="sk-261bf579de514b2a85eb81484391a437", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": input_text},
        ],
        stream=False
    )

    anwser=response.choices[0].message.content
    print(anwser)
    return anwser

def write_to_excel(data, index, filename):
    """
    Write the provided data to an Excel file.

    Parameters:
    - data: dict containing the data to write.
    - index: int, representing the index of the entry.
    - filename: str, the name of the output Excel file.

    Returns:
    - None. Saves the data to an Excel file.
    """
    # Extract relevant information from the data dictionary
    question = data.get("question", "")
    chunk = data.get("chunk", [])
    answer = data.get("anwser", "")  # Corrected typo 'anwser' to 'answer'
    
    # Create a DataFrame to organize the data
    df = pd.DataFrame({
        "Index": [index],
        "Question": [question],
        "Chunk": [chunk],  # Join chunk if it's a list
        "Answer": [answer]
    })
    
    # print(df)
    if os.path.exists(filename):
        # Read the existing data from the file
        existing_df = pd.read_excel(filename, engine='openpyxl')

        # Append the new data to the existing DataFrame
        updated_df = pd.concat([existing_df, df], ignore_index=True)

        # Write the updated DataFrame back to the file
        with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
            illegal_chars = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
            cleaned_df = updated_df.copy()
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].astype(str).apply(
                        lambda x: illegal_chars.sub('', x) if isinstance(x, str) else x
                    )

            cleaned_df.to_excel(writer, index=False, engine='openpyxl')
    else:
        # Create a new Excel file with the DataFrame
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            df.to_excel(writer, index=False, engine='openpyxl')

    return filename

for index, item in enumerate(query_array):
    if index>=0:
        data = {'question': [], 'chunk': [],'anwser':''}
        
        get_en_anser(item)
        get_zh_anser(item)
        
        # 处理null值
        for i_1,item_1 in enumerate(data_re):
            if item_1=="null":
                data_re[i_1]=[]
        for i_2,item_2 in enumerate(data_en_re):
            if item_2=="null":
                data_en_re[i_2]=[]

        seen_texts = set()
        deduplicated_zh = [[], [], [], []]  # [ans_id, doc_name, text, total_chunk_index]
        
        for i in range(len(data_re[2])):
            if data_re[2][i] and data_re[2][i] not in seen_texts:
                seen_texts.add(data_re[2][i])
                # deduplicated_zh[0].append(data_re[0][i])
                deduplicated_zh[1].append(data_re[1][i])
                deduplicated_zh[2].append(data_re[2][i])
                deduplicated_zh[3].append(data_re[3][i])
        
        # 第二步：合并英文结果，避免重复
        for i in range(len(data_en_re[3])):
            en_text = data_en_re[2][i]
            
            # 只要文本内容不重复就添加
            if en_text and en_text not in seen_texts:
                # deduplicated_zh[0].append(data_en_re[0][i])
                deduplicated_zh[1].append(data_en_re[1][i])
                deduplicated_zh[2].append(data_en_re[2][i])
                deduplicated_zh[3].append(data_en_re[3][i])
                seen_texts.add(en_text)
        
        # 更新data_re为去重后的结果
        data_re = deduplicated_zh
        # 后续处理保持不变
        docname=data_re[1]+data_en_re[1]
        for j,item2 in enumerate(docname):
            docname[j]=remove_file_extension(item2)
            
        data['question']=item
        restr=""
        if data_re[2]:
            for dr_i,dr_item in enumerate(data_re[2]):   
                addstr=f"chunk:{dr_i},source:{data_re[1][dr_i]}\n{dr_item}\n"  
                print(addstr)
                restr+=addstr
        print(restr)
        data['chunk']=restr
        
        write_to_excel(data,index,"/mnt/hdd1/haoyangliu/playground/chunk_case/new_em_ocr_chunkv0.xlsx")
        print(index)



end_time=time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time} 秒")
print(f"find {sumres}篇文献")
