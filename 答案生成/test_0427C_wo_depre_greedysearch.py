# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import json
from tqdm import tqdm    
        
def load_json(file_path):
    with open(file_path, "r+", encoding="utf8") as load_f:
        dicts = json.load(load_f)
    return dicts

def dcts2json(dcts, save_path):
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(dcts , f, indent=4, ensure_ascii=False) 
        
def rag_prompt(q, contexts):

    prompt2_c="""
{
  "instruction":"你是一个半导体显示领域的资深专家，你掌握TFT、OLED、LCD、QLED、EE、Design等显示半导体显示领域内的相关知识。请根据输入中的切片信息和问题进行回答。切片信息是可能相关的资料，切片信息的内容庞杂，不一定会包含目标答案，可能含有与问题相近的干扰信息，请仔细阅读每个切片后再作答，不得出现错误。"
  "input": {
    "context": "{context}"
    "question": "{question}"
  },
  "output": {
    "answer": "根据切片中提供的有效信息和自身知识对问题进行详尽的回答，推荐分点回答格式。"
  },
  "requirements": {
    "criteria": "根据提供的切片信息提取有效信息，同时结合自身已有的半导体显示知识进行完整、准确的回答",
    "format": "1、输出内容必须用中文作答且有逻辑条理性；2、输出内容不要显示引用切片，与模型原生回答格式一致。"
  }
}
"""

    contexts = contexts.strip()
    inputs = prompt2_c.replace('{context}', contexts.replace("\n", "\\n"))
    inputs = inputs.replace('{question}', q.replace("\n", "\\n"))
    return inputs

def rag_prompt_v2(word):


    prompt3_c="""
你是一个大模型关键词安全审核官，现在有一个关键词，请你按照如下几种类别进行准确分类：
类别：（1）煽动颠覆国家政权、推翻社会主义制度；（2）危害国家安全和利益、损害国家形象；（3）煽动分裂国家、破坏国家统一和社会稳定；（4）宣扬恐怖主义、极端主义；（5）宣扬民族仇恨；（6）宣扬暴力、淫秽色情；（7）传播虚假有害信息；（8）其他法律、行政法规禁止的内容；(9)民族歧视内容；(10)信仰歧视内容；（11）国别歧视内容；（12）地域歧视内容；（13）性别歧视内容；（14）年龄歧视内容；（15）职业歧视内容；（16）健康歧视内容；（17）其他方面歧视内容；（18）侵犯他人知识产权；（19）违反商业道德；（20）泄露他人商业秘密；（21）利用算法、数据、平台等优势，实施垄断和不正当竞争行为；（22）其他商业违法违规行为。（23）危害他人身心健康；（24）侵害他人肖像权；（25）侵害他人名誉权；（26）侵害他人荣誉权；（27）侵害他人隐私权；（28）侵害他人个人信息权益；（29）侵犯他人其他合法权益。
你需要分类的敏感词为{word}
只需要输出最终分类的类别名称即可，不需要输出其他额外信息
"""
    word = word.strip()
    inputs = prompt3_c.replace('{word}', word.replace("\n", "\\n"))
    return inputs

def get_model_response(input_text, model, tokenizer):
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]   
    messages = [{"role": "system", "content": "你是一个半导体显示技术领域的专家。"}, 
                {"role": "user", "content": input_text}]
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=8192, do_sample = False, eos_token_id=terminators)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    response = response.split("<|im_end|>")[0]
    response = response.split("<|endoftext|>")[0]
    return response
       
def test(model_path, query_path, save_file_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
    results = []
    input_file = query_path
    output_file = save_file_path
    data = pd.read_excel(input_file)
    answers = []
    
    """
    for i in range(99999999):
        question = "阐述墨家 “兼爱”“非攻” 思想的意义，以及在当代国际关系中的启示。"
        inputs = question
        answer = get_model_response(inputs, model, tokenizer)
        print("响应结果：", answer)
    """
    for index, row in data.iterrows():
        flag=0
        if(index>=0):
            # """
            context = row["Chunk"]
            question = row["Question"]
            if type(context)!=float:
                inputs = rag_prompt(question, context,)
            else:
                inputs = question
            answer = get_model_response(inputs, model, tokenizer)
            """
            word = row["word"]
            if type(word) != int and type(word)!=float:
  
                inputs = rag_prompt_v2(word)
                print(inputs) 
                answer = get_model_response(inputs, model, tokenizer)
            else:
                answer = ''
            """
            
            print("响应结果：", answer)
      
            # 将当前回答添加到 answers 列表
            data.at[index, "Answer"] = answer
            # 将当前结果写入 Excel 文件中
            data.to_excel(output_file, index=False)
            print(f"{index}当前结果已保存到 {output_file}")
      
        else:
            data.at[index, "Answer"] = "NULL_CHUNK"

    print(f"最终结果已保存到 {output_file}")


    print(f"最终结果已保存到 {output_file}")
    """
    samples = load_json(query_path)
    for sample in samples:
        sample['response'] = get_model_response(sample["question"], model, tokenizer)
        results.append(sample)
        dcts2json(results, save_file_path)
    """


if __name__ == "__main__":
    test("/mnt/data/MLLM/liuchi/trained_models/Qwen3-32B-dpo-5w_retrain","/mnt/data/LLM/xuleliu/qwne3_rag_answer/new_em_ocr_chunkv0.xlsx","new_em_ocr_chunkv0_qwen3_2_3.xlsx")
    """
    test("/data/lc/factory0929/LLaMA-Factory/out_merge/qwen2.5_14_base_pt0409_sft_merge2k_alpaca_3ep_batch72_template_default_final_preference_ct_dpo_batch120_lora64_beta0.05_template_qwen_cutoff4096", 
         "/data/lc/factory250506/tests/whole_domain100.json", 
         "domain100_0427C.json")
    """
    
